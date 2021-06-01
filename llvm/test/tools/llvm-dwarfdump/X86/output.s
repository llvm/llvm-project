# RUN: rm -f %t1.txt %t2.txt %t3.txt
# RUN: llvm-mc %S/brief.s -filetype obj -triple x86_64-apple-darwin -o %t.o

# RUN: llvm-dwarfdump -o=- %t.o | FileCheck %s

# RUN: llvm-dwarfdump -o=%t1.txt %t.o
# RUN: FileCheck %s --input-file %t1.txt

# RUN: touch %t2.txt
# RUN: llvm-dwarfdump -o=%t2.txt %t.o
# RUN: FileCheck %s --input-file %t2.txt

# RUN: touch %t3.txt
# RUN: chmod 444 %t3.txt
# RUN: not llvm-dwarfdump -o=%t3.txt %t.o 2>&1 | FileCheck %s  --check-prefix=ERROR1 -DFILE=%t3.txt

# RUN: not llvm-dwarfdump -o= %t.o 2>&1 | FileCheck %s  --check-prefix=ERROR2

# CHECK: DW_TAG_compile_unit
# ERROR1: unable to open output file [[FILE]]: {{[pP]}}ermission denied
# ERROR2: unable to open output file : {{[nN]}}o such file or directory
