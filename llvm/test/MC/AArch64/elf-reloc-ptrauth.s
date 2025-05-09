// RUN: llvm-mc -triple=aarch64 %s | FileCheck %s --check-prefix=ASM

// RUN: llvm-mc -triple=aarch64 -filetype=obj %s | \
// RUN:   llvm-readelf -S -r -x .test - | FileCheck %s --check-prefix=RELOC

// RELOC: Relocation section '.rela.test' at offset {{.*}} contains 9 entries:
// RELOC-NEXT:  Offset Info Type Symbol's Value Symbol's Name + Addend
// RELOC-NEXT: 0000000000000000 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 .helper + 0
// RELOC-NEXT: 0000000000000010 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g1 + 0
// RELOC-NEXT: 0000000000000020 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g2 + 0
// RELOC-NEXT: 0000000000000030 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g3 + 0
// RELOC-NEXT: 0000000000000040 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g4 + 7
// RELOC-NEXT: 0000000000000050 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g5 - 3
// RELOC-NEXT: 0000000000000060 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g 6 + 0
// RELOC-NEXT: 0000000000000070 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g 7 + 7
// RELOC-NEXT: 0000000000000080 {{.*}} R_AARCH64_AUTH_ABS64 0000000000000000 _g4 + 7

// RELOC: Hex dump of section '.test':
//                VVVVVVVV addend, not needed for rela
//                             VV reserved
// RELOC-NEXT: 00 00000000 2a000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT: 10 00000000 00000010
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 01 ib key 0000 reserved
// RELOC-NEXT: 20 00000000 050000a0
//                         ^^^^ discriminator
//                               ^^ 1    addr diversity 0 reserved 10 da key 0000 reserved
// RELOC-NEXT: 30 00000000 ffff00b0
//                         ^^^^ discriminator
//                               ^^ 1    addr diversity 0 reserved 11 db key 0000 reserved
// RELOC-NEXT: 40 00000000 00000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT: 50 00000000 00de0010
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 01 ib key 0000 reserved
// RELOC-NEXT: 60 00000000 ff0000b0
//                         ^^^^ discriminator
//                               ^^ 1    addr diversity 0 reserved 11 db key 0000 reserved
// RELOC-NEXT: 70 00000000 10000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved
// RELOC-NEXT: 80 00000000 00000000
//                         ^^^^ discriminator
//                               ^^ 0 no addr diversity 0 reserved 00 ia key 0000 reserved

.section    .helper
.local "_g 6"
.type _g0, @function
_g0:
  ret
.type _g8, @function
_g8:
  ret
.type _g9, @function
_g9:
  ret

.section	.test
.p2align	3

// ASM:          .xword _g0@AUTH(ia,42)
.quad _g0@AUTH(ia,42)
.quad 0

// ASM:          .xword (+_g1)@AUTH(ib,0)
.quad +_g1@AUTH(ib,0)
.quad 0

// ASM:          .xword _g2@AUTH(da,5,addr)
.quad _g2 @ AUTH(da,5,addr)
.quad 0

// ASM:          .xword _g3@AUTH(db,65535,addr)
.quad _g3@AUTH(db,0xffff,addr)
.quad 0

// ASM:          .xword (_g4+7)@AUTH(ia,0)
.quad (_g4 + 7)@AUTH(ia,0)
.quad 0

// ASM:          .xword (_g5-3)@AUTH(ib,56832)
.quad (_g5 - 3)@AUTH(ib,0xde00)
.quad 0

// ASM:          .xword "_g 6"@AUTH(db,255,addr)
.quad "_g 6"@AUTH(db,0xff,addr)
.quad 0

// ASM:          .xword ("_g 7"+7)@AUTH(ia,16)
.quad ("_g 7" + 7)@AUTH(ia,16)
.quad 0

.quad 7 + _g4@AUTH(ia,0)
.quad 0

// RUN: not llvm-mc -triple=aarch64 --defsym=ERR=1 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ERR --implicit-check-not=error:

.ifdef ERR

// ERR: :[[#@LINE+1]]:15: error: expected '('
.quad sym@AUTH)ia,42)

// ERR: :[[#@LINE+1]]:16: error: expected key name
.quad sym@AUTH(42,42)

// ERR: :[[#@LINE+1]]:16: error: invalid key 'ic'
.quad sym@AUTH(ic,42)

// ERR: :[[#@LINE+1]]:19: error: expected ','
.quad sym@AUTH(ia 42)

// ERR: :[[#@LINE+1]]:19: error: expected integer discriminator
.quad sym@AUTH(ia,xxx)

// ERR: :[[#@LINE+1]]:19: error: integer discriminator 65536 out of range [0, 0xFFFF]
.quad sym@AUTH(ia,65536)

// ERR: :[[#@LINE+1]]:22: error: expected 'addr'
.quad sym@AUTH(ia,42,add)

// ERR: :[[#@LINE+1]]:21: error: expected ')'
.quad sym@AUTH(ia,42(

// ERR: :[[#@LINE+1]]:14: error: unexpected token
.quad sym@PLT@AUTH(ia,42)

// ERR: :[[#@LINE+1]]:15: error: expected '('
.quad sym@AUTH@GOT(ia,42)

// ERR: :[[#@LINE+1]]:22: error: expected '('
.quad "long sym"@AUTH@PLT(ia,42)

// ERR: :[[#@LINE+1]]:17: error: invalid relocation specifier
.quad (sym - 5)@GOT@AUTH(ia,42)

// ERR: :[[#@LINE+1]]:23: error: unexpected token
.quad sym@AUTH(ia,42) + 1

// ERR: :[[#@LINE+1]]:27: error: unexpected token
.quad 1 + sym@AUTH(ia,42) + 1

/// @AUTH applies to the whole operand instead of an individual term.
/// Trailing expression parts are not allowed even if the logical subtraction
/// result might make sense.
// ERR: :[[#@LINE+1]]:23: error: unexpected token
.quad _g9@AUTH(ia,42) - _g8

// ERR: :[[#@LINE+1]]:23: error: unexpected token
.quad _g9@AUTH(ia,42) - _g8@AUTH(ia,42)

// ERR: :[[#@LINE+1]]:24: error: unexpected token
.quad _g13@AUTH(ia,42) + _g14@AUTH(ia,42)

.endif // ERR
