# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=-relax %s -o %t
# RUN: llvm-readobj -r -x .alloc_w %t| FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r -x .alloc_w %t.relax | FileCheck %s --check-prefixes=CHECK,RELAX

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax %s -o %t
# RUN: llvm-readobj -r -x .alloc_w %t | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r -x .alloc_w %t.relax | FileCheck %s --check-prefixes=CHECK,RELAX

## Test temporary workaround for suppressting relocations for actually-non-foldable
## DWARF v5 DW_LLE_offset_pair/DW_RLE_offset_pair.
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax -riscv-uleb128-reloc=0 %s -o %t0
# RUN: llvm-readobj -r -x .alloc_w %t0 | FileCheck %s --check-prefix=CHECK0
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax -riscv-uleb128-reloc=0 %s -o %t0.relax
# RUN: llvm-readobj -r -x .alloc_w %t0.relax | FileCheck %s --check-prefixes=CHECK0,RELAX0

# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=-relax --defsym ERR=1 %s -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR
# RUN: not llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax --defsym ERR=1 %s -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR

# CHECK0:      Relocations [
# CHECK0-NEXT:   .rela.alloc_w {
# CHECK0-NEXT:     0x2 R_RISCV_CALL_PLT foo 0x0
# RELAX0-NEXT:     0x2 R_RISCV_RELAX - 0x0
# CHECK0-NEXT:   }
# CHECK0-NEXT: ]

# CHECK:      Relocations [
# CHECK-NEXT:   .rela.alloc_w {
# CHECK-NEXT:     0x0 R_RISCV_SET_ULEB128 w1 0x0
# CHECK-NEXT:     0x0 R_RISCV_SUB_ULEB128 w 0x0
# RELAX-NEXT:     0x1 R_RISCV_SET_ULEB128 w2 0x0
# RELAX-NEXT:     0x1 R_RISCV_SUB_ULEB128 w1 0x0
# CHECK-NEXT:     0x2 R_RISCV_CALL_PLT foo 0x0
# RELAX-NEXT:     0x2 R_RISCV_RELAX - 0x0
# RELAX-NEXT:     0xA R_RISCV_SET_ULEB128 w2 0x0
# RELAX-NEXT:     0xA R_RISCV_SUB_ULEB128 w1 0x0
# RELAX-NEXT:     0xB R_RISCV_SET_ULEB128 w2 0x78
# RELAX-NEXT:     0xB R_RISCV_SUB_ULEB128 w1 0x0
# RELAX-NEXT:     0xD R_RISCV_SET_ULEB128 w1 0x0
# RELAX-NEXT:     0xD R_RISCV_SUB_ULEB128 w2 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

## R_RISCV_SET_ULEB128 relocated locations contain values not accounting for linker relaxation.
# CHECK:      Hex dump of section '.alloc_w':
# CHECK-NEXT: 0x00000000 02089700 0000e780 00000880 01f8ffff ................
# CHECK-NEXT: 0x00000010 ffffffff ffff01                     .......

.section .alloc_w,"ax",@progbits; w:
.uleb128 w1-w       # w1 is later defined in the same section
.uleb128 w2-w1      # w1 and w2 are separated by a linker relaxable instruction
w1:
  call foo
w2:
.uleb128 w2-w1      # 0x08
.uleb128 w2-w1+120  # 0x0180
.uleb128 -(w2-w1)   # 0x01fffffffffffffffff8

.ifdef ERR
# ERR: :[[#@LINE+1]]:16: error: .uleb128 expression is not absolute
.uleb128 extern-w   # extern is undefined
# ERR: :[[#@LINE+1]]:11: error: .uleb128 expression is not absolute
.uleb128 w-extern
# ERR: :[[#@LINE+1]]:11: error: .uleb128 expression is not absolute
.uleb128 x-w        # x is later defined in another section

.section .alloc_x,"aw",@progbits; x:
# ERR: :[[#@LINE+1]]:11: error: .uleb128 expression is not absolute
.uleb128 y-x
.section .alloc_y,"aw",@progbits; y:
# ERR: :[[#@LINE+1]]:11: error: .uleb128 expression is not absolute
.uleb128 x-y

# ERR: :[[#@LINE+1]]:10: error: .uleb128 expression is not absolute
.uleb128 extern
# ERR: :[[#@LINE+1]]:10: error: .uleb128 expression is not absolute
.uleb128 y
.endif
