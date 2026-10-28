// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+pauth -show-encoding %s | FileCheck %s
// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+pauth -filetype=obj %s -o - | \
// RUN:   llvm-readelf -r -s - | FileCheck --check-prefix=CHECK-ELF %s

adrp x8, :tlsdesc_auth:var
ldr x7, [x6, :tlsdesc_auth_lo12:var]
add x5, x4, #:tlsdesc_auth_lo12:var
.tlsauthdesccall var
blraa x3, x2

// CHECK:      adrp   x8, :tlsdesc_auth:var            // encoding: [0x08'A',A,A,0x90'A']
// CHECK-NEXT:                                         // fixup A - offset: 0, value: :tlsdesc_auth:var, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK:      ldr    x7, [x6, :tlsdesc_auth_lo12:var] // encoding: [0xc7,0bAAAAAA00,0b01AAAAAA,0xf9]
// CHECK-NEXT:                                         // fixup A - offset: 0, value: :tlsdesc_auth_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK:      add    x5, x4, :tlsdesc_auth_lo12:var   // encoding: [0x85,0bAAAAAA00,0b00AAAAAA,0x91]
// CHECK-NEXT:                                         // fixup A - offset: 0, value: :tlsdesc_auth_lo12:var, kind: fixup_aarch64_add_imm12
// CHECK:      .tlsauthdesccall var                    // encoding: []
// CHECK-NEXT:                                         // fixup A - offset: 0, value: var, relocation type: 598
// CHECK:      blraa  x3, x2                           // encoding: [0x62,0x08,0x3f,0xd7]

// CHECK-ELF:      Relocation section '.rela.text' at offset 0x98 contains 4 entries:
// CHECK-ELF-NEXT: Offset           Info             Type                              Symbol's Value   Symbol's Name + Addend
// CHECK-ELF-NEXT: 0000000000000000 0000000200000253 R_AARCH64_AUTH_TLSDESC_ADR_PAGE21 0000000000000000 var + 0
// CHECK-ELF-NEXT: 0000000000000004 0000000200000254 R_AARCH64_AUTH_TLSDESC_LD64_LO12  0000000000000000 var + 0
// CHECK-ELF-NEXT: 0000000000000008 0000000200000255 R_AARCH64_AUTH_TLSDESC_ADD_LO12   0000000000000000 var + 0
// CHECK-ELF-NEXT: 000000000000000c 0000000200000256 R_AARCH64_AUTH_TLSDESC_CALL       0000000000000000 var + 0

// Make sure symbol has type STT_TLS:

// CHECK-ELF:      Symbol table '.symtab' contains [[#]] entries:
// CHECK-ELF-NEXT: Num:   Value            Size Type Bind   Vis     Ndx Name
// CHECK-ELF:      [[#]]: 0000000000000000 0    TLS  GLOBAL DEFAULT UND var
