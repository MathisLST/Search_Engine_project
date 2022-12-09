import re

def load_data(): 

    all_path = "CISI/CISI.ALL"
    rel_path = "CISI/CISI.REL"
    qry_path = "CISI/CISI.QRY"

    expr_all = "\.I\s+(\d+)\s+\.T\s+([\s\S]+?)^\.A\s+([\s\S]+?)^\.W\s+([\s\S]+?)^\.X\s+([\d\s]+)"
    expr_rel = "^\s+(\d+)\s+(\d+)"

    with open(all_path) as f:
        lines = f.read()
        all_file = re.findall(expr_all, lines, re.M) 

    with open(rel_path) as f:
        lines = f.read()
        rel_file_brut = re.findall(expr_rel, lines, re.M)
        rel_file = {}
        for p in rel_file_brut:
            if(int(p[0]) in rel_file):
                rel_file[int(p[0])].append(int(p[1]))
            else:
                rel_file[int(p[0])] = [int(p[1])]
    qry_file = []
    with open(qry_path) as f:
        lines = f.readlines()
        x = []
        for i in range (0,len(lines)):
            w = ""
            if(lines[i].startswith(".I")):
                 x.append(int(lines[i].split()[1]))

            if(lines[i].startswith(".W")):
                i+=1
                while(not lines[i].startswith(".")):
                    w += lines[i]
                    i+=1
                x.append(w)
                qry_file.append(x)
                x = []

    return all_file,qry_file,rel_file