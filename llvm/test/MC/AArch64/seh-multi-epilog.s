// This test checks that we emit allow multiple consecutive epilogs without
// triggering failed asserts.unwind info correctly for epilogs that:
// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s -o %t.o
// RUN: llvm-readobj -u %t.o | FileCheck %s

// CHECK-LABEL:UnwindInformation [
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: multi_epilog (0x0)
// CHECK-NEXT:    ExceptionRecord: .xdata (0x0)
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength: 20
// CHECK-NEXT:      Version: 0
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: No
// CHECK-NEXT:      EpilogueScopes: 2
// CHECK-NEXT:      ByteCodeLength: 4
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0x81                ; stp x29, x30, [sp, #-16]!
// CHECK-NEXT:        0xe4                ; end
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 2
// CHECK-NEXT:          EpilogueStartIndex: 0
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0x81                ; ldp x29, x30, [sp], #16
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 3
// CHECK-NEXT:          EpilogueStartIndex: 0
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0x81                ; ldp x29, x30, [sp], #16
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:]

	.text
	.global	multi_epilog
	.p2align	2
	.seh_proc multi_epilog
multi_epilog:
	stp	x29, lr, [sp, #-16]!
	.seh_save_fplr_x 16
	.seh_endprologue
        nop
	.seh_startepilogue
	ldp	x29, lr, [sp], #16
	.seh_save_fplr_x 16
	.seh_endepilogue
	.seh_startepilogue
	ldp	x29, lr, [sp], #16
	.seh_save_fplr_x 16
	.seh_endepilogue
	ret
	.seh_endfunclet
	.seh_endproc
