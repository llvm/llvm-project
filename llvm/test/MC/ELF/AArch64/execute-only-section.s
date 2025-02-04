// RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux-gnu %s -o - \
// RUN: | llvm-readobj -S --symbols - | FileCheck %s

        .section        .text,"axy",@progbits,unique,0
        .globl  foo
        .p2align  2
        .type   foo,@function
foo:
        .cfi_startproc
        ret
.Lfunc_end0:
        .size   foo, .Lfunc_end0-foo
        .cfi_endproc

        .section        ".note.GNU-stack","",@progbits


// CHECK:      Section {
// CHECK:        Name: .text
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x20000006)
// CHECK-NEXT:     SHF_AARCH64_PURECODE (0x20000000)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_EXECINSTR (0x4)
// CHECK-NEXT:   ]
// CHECK:        Size: 0
// CHECK:      }

// CHECK:      Section {
// CHECK:        Name: .text
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x20000006)
// CHECK-NEXT:     SHF_AARCH64_PURECODE (0x20000000)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_EXECINSTR (0x4)
// CHECK-NEXT:   ]
// CHECK:        Size: 2
// CHECK:      }

// CHECK: Symbol {
// CHECK:   Name: foo
// CHECK:   Section: .text (0x3)
// CHECK: }
