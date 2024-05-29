# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax %s -o %t
# RUN: llvm-readobj -r -x .alloc_w %t | FileCheck --check-prefixes=CHECK,NORELAX %s
# RUN: llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax %s -o %t.relax
# RUN: llvm-readobj -r -x .alloc_w %t.relax | FileCheck --check-prefixes=CHECK,RELAX %s

# RUN: not llvm-mc --filetype=obj --triple=loongarch64 --mattr=-relax --defsym ERR=1 %s -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR
# RUN: not llvm-mc --filetype=obj --triple=loongarch64 --mattr=+relax --defsym ERR=1 %s -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR

# CHECK:      Relocations [
# CHECK-NEXT:   .rela.alloc_w {
# RELAX-NEXT:      0x0 R_LARCH_ADD_ULEB128 w1 0x0
# RELAX-NEXT:      0x0 R_LARCH_SUB_ULEB128 w 0x0
# RELAX-NEXT:      0x1 R_LARCH_ADD_ULEB128 w2 0x0
# RELAX-NEXT:      0x1 R_LARCH_SUB_ULEB128 w1 0x0
# CHECK-NEXT:      0x2 R_LARCH_PCALA_HI20 foo 0x0
# RELAX-NEXT:      0x2 R_LARCH_RELAX - 0x0
# CHECK-NEXT:      0x6 R_LARCH_PCALA_LO12 foo 0x0
# RELAX-NEXT:      0x6 R_LARCH_RELAX - 0x0
# RELAX-NEXT:      0xA R_LARCH_ADD_ULEB128 w2 0x0
# RELAX-NEXT:      0xA R_LARCH_SUB_ULEB128 w1 0x0
# RELAX-NEXT:      0xB R_LARCH_ADD_ULEB128 w2 0x78
# RELAX-NEXT:      0xB R_LARCH_SUB_ULEB128 w1 0x0
# RELAX-NEXT:      0xD R_LARCH_ADD_ULEB128 w1 0x0
# RELAX-NEXT:      0xD R_LARCH_SUB_ULEB128 w2 0x0
# RELAX-NEXT:      0x17 R_LARCH_ADD_ULEB128 w3 0x6F
# RELAX-NEXT:      0x17 R_LARCH_SUB_ULEB128 w2 0x0
# RELAX-NEXT:      0x18 R_LARCH_ADD_ULEB128 w3 0x71
# RELAX-NEXT:      0x18 R_LARCH_SUB_ULEB128 w2 0x0
# CHECK-NEXT:   }
# CHECK-NEXT: ]

# CHECK:        Hex dump of section '.alloc_w':
# NORELAX-NEXT: 0x00000000 02080c00 001a8c01 c0020880 01f8ffff
# NORELAX-NEXT: 0x00000010 ffffffff ffff017f 8101
# RELAX-NEXT:   0x00000000 00000c00 001a8c01 c0020080 00808080
# RELAX-NEXT:   0x00000010 80808080 80800000 8000

.section .alloc_w,"ax",@progbits; w:
.uleb128 w1-w       # w1 is later defined in the same section
.uleb128 w2-w1      # w1 and w2 are separated by a linker relaxable instruction
w1:
  la.pcrel $t0, foo
w2:
.uleb128 w2-w1      # 0x08
.uleb128 w2-w1+120  # 0x0180
.uleb128 -(w2-w1)   # 0x01fffffffffffffffff8
.uleb128 w3-w2+111  # 0x7f
.uleb128 w3-w2+113  # 0x0181
w3:

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
