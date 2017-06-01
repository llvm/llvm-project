// RUN: llvm-mc -filetype=obj -triple thumbv7m-arm-linux-gnu %s -o - \
// RUN: | llvm-readobj -s -t | FileCheck %s

        .section        .text,"axy",%progbits,unique,0
        .globl  foo
        .align  2
        .type   foo,%function
        .code   16
        .thumb_func
foo:
        .fnstart
        bx      lr
.Lfunc_end0:
        .size   foo, .Lfunc_end0-foo
        .fnend

        .section        ".note.GNU-stack","",%progbits


// CHECK:      Section {
// CHECK:        Name: .text (16)
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x6)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_EXECINSTR (0x4)
// CHECK-NEXT:   ]
// CHECK:        Size: 0
// CHECK:      }

// CHECK:      Section {
// CHECK:        Name: .text (16)
// CHECK-NEXT:   Type: SHT_PROGBITS (0x1)
// CHECK-NEXT:   Flags [ (0x20000006)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_ARM_PURECODE (0x20000000)
// CHECK-NEXT:     SHF_EXECINSTR (0x4)
// CHECK-NEXT:   ]
// CHECK:        Size: 2
// CHECK:      }

// CHECK: Symbol {
// CHECK:   Name: foo (22)
// CHECK:   Section: .text (0x3)
// CHECK: }
