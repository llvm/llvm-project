// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+pauth -show-encoding < %s | FileCheck %s
// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+pauth -filetype=obj < %s -o - | \
// RUN:   llvm-readobj -r --symbols - | FileCheck --check-prefix=CHECK-ELF %s

        adrp x8, :tlsdesc_auth:var
        ldr x7, [x6, :tlsdesc_auth_lo12:var]
        add x5, x4, #:tlsdesc_auth_lo12:var
        .tlsdescauthcall var
        blraa x3, x2

// CHECK:      adrp   x8, :tlsdesc_auth:var            // encoding: [0x08'A',A,A,0x90'A']
// CHECK-NEXT:                                         // fixup A - offset: 0, value: :tlsdesc_auth:var, kind: fixup_aarch64_pcrel_adrp_imm21
// CHECK:      ldr    x7, [x6, :tlsdesc_auth_lo12:var] // encoding: [0xc7,0bAAAAAA00,0b01AAAAAA,0xf9]
// CHECK-NEXT:                                         // fixup A - offset: 0, value: :tlsdesc_auth_lo12:var, kind: fixup_aarch64_ldst_imm12_scale8
// CHECK:      add    x5, x4, :tlsdesc_auth_lo12:var   // encoding: [0x85,0bAAAAAA00,0b00AAAAAA,0x91]
// CHECK-NEXT:                                         // fixup A - offset: 0, value: :tlsdesc_auth_lo12:var, kind: fixup_aarch64_add_imm12
// CHECK:      .tlsdescauthcall var                    // encoding: []
// CHECK-NEXT:                                         // fixup A - offset: 0, value: var, relocation type: 598
// CHECK:      blraa  x3, x2                           // encoding: [0x62,0x08,0x3f,0xd7]

// CHECK-ELF:      Relocations [
// CHECK-ELF-NEXT:   Section {{.*}} .rela.text {
// CHECK-ELF-NEXT:     0x0 R_AARCH64_AUTH_TLSDESC_ADR_PAGE21 [[VARSYM:[^ ]+]]
// CHECK-ELF-NEXT:     0x4 R_AARCH64_AUTH_TLSDESC_LD64_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0x8 R_AARCH64_AUTH_TLSDESC_ADD_LO12 [[VARSYM]]
// CHECK-ELF-NEXT:     0xC R_AARCH64_AUTH_TLSDESC_CALL [[VARSYM]]

// Make sure symbol has type STT_TLS:

// CHECK-ELF:      Symbols [
// CHECK-ELF:        Symbol {
// CHECK-ELF:          Name: var
// CHECK-ELF-NEXT:     Value:
// CHECK-ELF-NEXT:     Size:
// CHECK-ELF-NEXT:     Binding: Global
// CHECK-ELF-NEXT:     Type: TLS
