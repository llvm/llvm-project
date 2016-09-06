# RUN: llvm-mc -arch=mips64el -filetype=obj -mcpu=mips64r2 -target-abi=n64 %s -o - \
# RUN:   | llvm-objdump -disassemble - | FileCheck %s

        dext $2, $4, 5, 10   # CHECK: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
        dextu $2, $4, 34, 6  # CHECK: dextu ${{[0-9]+}}, ${{[0-9]+}}, 34, 6
        dextm $2, $4, 5, 34  # CHECK: dextm ${{[0-9]+}}, ${{[0-9]+}}, 5, 34
        dins $4, $5, 8, 10   # CHECK: dins ${{[0-9]+}}, ${{[0-9]+}}, 8, 10
        dinsm $4, $5, 10, 1  # CHECK: dinsm ${{[0-9]+}}, ${{[0-9]+}}, 10, 1
        dinsu $4, $5, 40, 13 # CHECK: dinsu ${{[0-9]+}}, ${{[0-9]+}}, 40, 13
