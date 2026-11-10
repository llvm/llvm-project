## Test human-readable annotations for dbar hints (`-M annotate`), in the
## same format as GNU objdump.

# RUN: llvm-mc --triple=loongarch64 -M annotate %s \
# RUN:     | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc --triple=loongarch64 --loongarch-annotate %s \
# RUN:     | FileCheck --check-prefix=ASM %s
# RUN: llvm-mc --triple=loongarch64 --filetype=obj %s -o %t
# RUN: llvm-objdump -d --no-print-imm-hex -M annotate --no-show-raw-insn %t \
# RUN:     | FileCheck --check-prefix=OBJ --match-full-lines --strict-whitespace %s
# RUN: llvm-objdump -d --no-print-imm-hex --no-show-raw-insn %t \
# RUN:     | FileCheck --check-prefix=NOANN --implicit-check-not=# %s

# NOANN: dbar 0
# ASM:      dbar 0 #  RW|RW
# OBJ: {{.*}}dbar{{[[:space:]]+}}0{{[[:space:]]+}}#  RW|RW
dbar 0x0

# ASM:      dbar 1 #  RW|R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}1{{[[:space:]]+}}#  RW|R
dbar 0x1

# ASM:      dbar 2 #  RW| W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}2{{[[:space:]]+}}#  RW| W
dbar 0x2

# ASM:      dbar 3 #  RW|
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}3{{[[:space:]]+}}#  RW|
dbar 0x3

# ASM:      dbar 4 #  R |RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}4{{[[:space:]]+}}#  R |RW
dbar 0x4

# ASM:      dbar 5 #  R |R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}5{{[[:space:]]+}}#  R |R
dbar 0x5

# ASM:      dbar 6 #  R | W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}6{{[[:space:]]+}}#  R | W
dbar 0x6

# ASM:      dbar 7 #  R |
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}7{{[[:space:]]+}}#  R |
dbar 0x7

# ASM:      dbar 8 #   W|RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}8{{[[:space:]]+}}#   W|RW
dbar 0x8

# ASM:      dbar 9 #   W|R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}9{{[[:space:]]+}}#   W|R
dbar 0x9

# ASM:      dbar 10 #   W| W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}10{{[[:space:]]+}}#   W| W
dbar 0xa

# ASM:      dbar 11 #   W|
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}11{{[[:space:]]+}}#   W|
dbar 0xb

# ASM:      dbar 12 #    |RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}12{{[[:space:]]+}}#    |RW
dbar 0xc

# ASM:      dbar 13 #    |R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}13{{[[:space:]]+}}#    |R
dbar 0xd

# ASM:      dbar 14 #    | W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}14{{[[:space:]]+}}#    | W
dbar 0xe

# ASM:      dbar 15{{$}}
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}15{{$}}
dbar 0xf

# ASM:      dbar 16 # CRW|RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}16{{[[:space:]]+}}# CRW|RW
dbar 0x10

# ASM:      dbar 17 # CRW|R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}17{{[[:space:]]+}}# CRW|R
dbar 0x11

# ASM:      dbar 18 # CRW| W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}18{{[[:space:]]+}}# CRW| W
dbar 0x12

# ASM:      dbar 19 # CRW|
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}19{{[[:space:]]+}}# CRW|
dbar 0x13

# ASM:      dbar 20 # CR |RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}20{{[[:space:]]+}}# CR |RW
dbar 0x14

# ASM:      dbar 21 # CR |R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}21{{[[:space:]]+}}# CR |R
dbar 0x15

# ASM:      dbar 22 # CR | W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}22{{[[:space:]]+}}# CR | W
dbar 0x16

# ASM:      dbar 23 # CR |
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}23{{[[:space:]]+}}# CR |
dbar 0x17

# ASM:      dbar 24 # C W|RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}24{{[[:space:]]+}}# C W|RW
dbar 0x18

# ASM:      dbar 25 # C W|R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}25{{[[:space:]]+}}# C W|R
dbar 0x19

# ASM:      dbar 26 # C W| W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}26{{[[:space:]]+}}# C W| W
dbar 0x1a

# ASM:      dbar 27 # C W|
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}27{{[[:space:]]+}}# C W|
dbar 0x1b

# ASM:      dbar 28 # C  |RW
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}28{{[[:space:]]+}}# C  |RW
dbar 0x1c

# ASM:      dbar 29 # C  |R
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}29{{[[:space:]]+}}# C  |R
dbar 0x1d

# ASM:      dbar 30 # C  | W
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}30{{[[:space:]]+}}# C  | W
dbar 0x1e

# ASM:      dbar 31{{$}}
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}31{{$}}
dbar 0x1f

# ASM:      dbar 1792 # SA-RAR
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}1792{{[[:space:]]+}}# SA-RAR
dbar 0x700

# ASM:      dbar 28672{{$}}
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}28672{{$}}
dbar 0x7000

# ASM:      dbar 1808{{$}}
# OBJ-NEXT: {{.*}}dbar{{[[:space:]]+}}1808{{$}}
dbar 0x710

## ibar is not annotated.
# ASM:      ibar 0{{$}}
# OBJ-NEXT: {{.*}}ibar{{[[:space:]]+}}0{{$}}
ibar 0
