# RUN: llvm-mc -arch=mips64el -filetype=obj -mcpu=mips64r2 -target-abi=n64 %s -o - \
# RUN:   | llvm-objdump -disassemble - | FileCheck --check-prefix=OBJ %s
# RUN: llvm-mc -arch=mips64el -filetype=obj -mcpu=mips64r6 -mattr=+micromips \
# RUN:         -target-abi=n64 %s -o - | llvm-objdump -disassemble - \
# RUN:   | FileCheck --check-prefix=OBJ %s

# RUN: llvm-mc -arch=mips64el -mcpu=mips64r2 -target-abi=n64 %s -o - \
# RUN:   | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc -arch=mips64el -mcpu=mips64r6 -mattr=+micromips -target-abi=n64 \
# RUN:     %s -o - | FileCheck --check-prefix=ASM %s

        dext $2, $4, 5, 10   # OBJ: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
        dextu $2, $4, 34, 6  # OBJ: dext ${{[0-9]+}}, ${{[0-9]+}}, 34, 6
        dextm $2, $4, 5, 34  # OBJ: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 34
        dins $4, $5, 8, 10   # OBJ: dins ${{[0-9]+}}, ${{[0-9]+}}, 8, 10
        dinsm $4, $5, 30, 6  # OBJ: dins ${{[0-9]+}}, ${{[0-9]+}}, 30, 6
        dinsu $4, $5, 40, 13 # OBJ: dins ${{[0-9]+}}, ${{[0-9]+}}, 40, 13
        # check the aliases
        dins $2, $4, 5, 10   # OBJ: dins ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
        dins $2, $4, 34, 6   # OBJ: dins ${{[0-9]+}}, ${{[0-9]+}}, 34, 6
        dins $2, $4, 5, 34   # OBJ: dins ${{[0-9]+}}, ${{[0-9]+}}, 5, 34
        dext $2, $4, 5, 10   # OBJ: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 10
        dext $2, $4, 34, 6   # OBJ: dext ${{[0-9]+}}, ${{[0-9]+}}, 34, 6
        dext $2, $4, 5, 34   # OBJ: dext ${{[0-9]+}}, ${{[0-9]+}}, 5, 34
        # check the edge values
        dins $3, $4, 31, 1   # ASM: dins $3, $4, 31, 1
        dins $3, $4, 31, 33  # ASM: dinsm $3, $4, 31, 33
        dins $3, $4, 32, 32  # ASM: dinsu $3, $4, 32, 32
        dext $3, $4, 31, 32  # ASM: dext $3, $4, 31, 32
        dext $3, $4, 31, 33  # ASM: dextm $3, $4, 31, 33
        dext $3, $4, 32, 32  # ASM: dextu $3, $4, 32, 32

