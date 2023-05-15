def modify_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

#Find the index of second string, that contains "addCall" substring
    index = -1
    count = 0
    for i, line in enumerate(lines):
        if 'addCall' in line:
            count += 1
            if count == 2:
                index = i
                break

    if index == -1:
        print("Can't find second string with 'addCall' as a substring in file")
        return

#Change two strings before second string with 'addCall'
    for i in range(index - 2, index):
        if i >= 0:
            line = lines[i]
            if "=" in line:
                parts = line.split("=")
                parts[1] = " call i64* @llvm.returnaddress(i32 " + str(index - i - 1) + ")\n"
                line = "=".join(parts)
                lines[i] = line

    next_index = index + 3
    if next_index < len(lines):
        lines.insert(next_index, "declare i64* @llvm.returnaddress(i32)\n")

    with open(filename, 'w') as file:
        file.writelines(lines)

    print("File changed successfully.")


filename = 'Logger.ll'
modify_file(filename)
