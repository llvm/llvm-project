# RUN: llvm-mc -triple s390x-linux-gnu -filetype=asm %s | \
# RUN: FileCheck %s --check-prefix=ASM
# RUN: llvm-mc -triple s390x-linux-gnu -filetype=obj %s | \
# RUN: llvm-objdump --mcpu=z14 -D - | FileCheck %s --check-prefix=OBJ

#ASM:      .gnu_attribute 8, 2

#OBJ:  0000000000000000 <.gnu.attributes>:
#OBJ:       0: 41 00 00 00
#OBJ:       4: 0f 67
#OBJ:       6: 6e 75 00 01
#OBJ:       a: 00 00
#OBJ:       c: 00 07
#OBJ:       e: 08 02

  .gnu_attribute 8, 2
