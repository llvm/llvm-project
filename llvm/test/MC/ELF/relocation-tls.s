// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - | llvm-readobj -S --sr | FileCheck  %s

// Test that we produce the correct relocation.

        leaq    _ZL3ccc@TLSDESC(%rip), %rax
        call    *_ZL3ccc@TLSCALL(%rax)
        addq    %fs:0, %rax

// CHECK: Section {
// CHECK:   Index:
// CHECK:   Name: .rela.text
// CHECK-NEXT:   Type: SHT_RELA
// CHECK-NEXT:   Flags [
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x0
// CHECK-NEXT:   Offset:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Link:
// CHECK-NEXT:   Info:
// CHECK-NEXT:   AddressAlignment: 8
// CHECK-NEXT:   EntrySize: 24
// CHECK-NEXT:   Relocations [
// CHECK-NEXT:     0x3 R_X86_64_GOTPC32_TLSDESC _ZL3ccc 0xFFFFFFFFFFFFFFFC
// CHECK-NEXT:     0x9 R_X86_64_TLSDESC_CALL _ZL3ccc 0x0
// CHECK-NEXT:   ]
// CHECK-NEXT: }
