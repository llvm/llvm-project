// Check that unwind info emission can compute the function length and epilog
// offsets even when an alignment directive (e.g. from an inline asm blob)
// creates alignment padding in the middle of the function. This used to fail
// with "Failed to evaluate function length in SEH unwind info".

// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s -o %t.o
// RUN: llvm-readobj -u %t.o | FileCheck %s

// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func
// CHECK-NEXT:     ExceptionRecord: .xdata
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 28
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK-NEXT:       EpilogueOffset: 0
// CHECK-NEXT:       ByteCodeLength: 4
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0x83                ; stp x29, x30, [sp, #-32]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func2
// CHECK-NEXT:     ExceptionRecord: .xdata
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 48
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: No
// CHECK-NEXT:       EpilogueScopes: 2
// CHECK-NEXT:       ByteCodeLength: 4
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0x83                ; stp x29, x30, [sp, #-32]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:         EpilogueScope {
// CHECK-NEXT:           StartOffset: 2
// CHECK-NEXT:           EpilogueStartIndex: 0
// CHECK-NEXT:           Opcodes [
// CHECK-NEXT:             0x83                ; ldp x29, x30, [sp], #32
// CHECK-NEXT:             0xe4                ; end
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:         EpilogueScope {
// CHECK-NEXT:           StartOffset: 10
// CHECK-NEXT:           EpilogueStartIndex: 0
// CHECK-NEXT:           Opcodes [
// CHECK-NEXT:             0x83                ; ldp x29, x30, [sp], #32
// CHECK-NEXT:             0xe4                ; end
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

    .text
    .globl func
    .seh_proc func
func:
    stp x29, x30, [sp, #-32]!       // offset 0
    .seh_save_fplr_x 32
    .seh_endprologue
    nop                             // offset 4
    // 8 bytes of padding; the section alignment is raised to 16, so the
    // amount of padding is known before layout.
    .p2align 4
    nop                             // offset 16
    .seh_startepilogue
    ldp x29, x30, [sp], #32         // offset 20
    .seh_save_fplr_x 32
    .seh_endepilogue
    ret                             // offset 24
    .seh_endproc                    // function length 28

// Multiple epilogs (so that the epilog scopes can't be packed), with the
// second epilog's start offset crossing the alignment padding. The function
// itself does not start at the requested alignment (it starts at section
// offset 28), so this also checks that padding is computed from the start of
// the section rather than the start of the function.
    .globl func2
    .seh_proc func2
func2:
    stp x29, x30, [sp, #-32]!       // function offset 0, section offset 28
    .seh_save_fplr_x 32
    .seh_endprologue
    cbz x0, 1f                      // offset 4
    .seh_startepilogue
    ldp x29, x30, [sp], #32         // offset 8
    .seh_save_fplr_x 32
    .seh_endepilogue
    ret                             // offset 12
1:
    .p2align 5                      // 20 bytes of padding (to section offset 64)
    nop                             // offset 36
    .seh_startepilogue
    ldp x29, x30, [sp], #32         // offset 40
    .seh_save_fplr_x 32
    .seh_endepilogue
    ret                             // offset 44
    .seh_endproc                    // function length 48
