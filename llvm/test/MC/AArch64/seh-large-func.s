// This test checks that we emit unwind info correctly for functions
// larger than 1MB.
// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s -o %t.o
// RUN: llvm-readobj -S -r -u %t.o | FileCheck %s

// CHECK:        Section {
// CHECK:          Number: 4
// CHECK-NEXT:     Name: .xdata (2E 78 64 61 74 61 00 00)
// CHECK-NEXT:     VirtualSize: 0x0
// CHECK-NEXT:     VirtualAddress: 0x0
// CHECK-NEXT:     RawDataSize: 52
// CHECK-NEXT:     PointerToRawData: 0x3D0A20
// CHECK-NEXT:     PointerToRelocations: 0x0
// CHECK-NEXT:     PointerToLineNumbers: 0x0
// CHECK-NEXT:     RelocationCount: 0
// CHECK-NEXT:     LineNumberCount: 0
// CHECK-NEXT:     Characteristics [ (0x40300040)
// CHECK-NEXT:       IMAGE_SCN_ALIGN_4BYTES (0x300000)
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// CHECK-NEXT:       IMAGE_SCN_MEM_READ (0x40000000)
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK-NEXT:     Number: 5
// CHECK-NEXT:     Name: .pdata (2E 70 64 61 74 61 00 00)
// CHECK-NEXT:     VirtualSize: 0x0
// CHECK-NEXT:     VirtualAddress: 0x0
// CHECK-NEXT:     RawDataSize: 40
// CHECK-NEXT:     PointerToRawData: 0x3D0A54
// CHECK-NEXT:     PointerToRelocations: 0x3D0A7C
// CHECK-NEXT:     PointerToLineNumbers: 0x0
// CHECK-NEXT:     RelocationCount: 10
// CHECK-NEXT:     LineNumberCount: 0
// CHECK-NEXT:     Characteristics [ (0x40300040)
// CHECK-NEXT:       IMAGE_SCN_ALIGN_4BYTES (0x300000)
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// CHECK-NEXT:       IMAGE_SCN_MEM_READ (0x40000000)
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-LABEL: Relocations [
// CHECK-NEXT:   Section (1) .text {
// CHECK-NEXT:     0x186A04 IMAGE_REL_ARM64_BRANCH26 foo (14)
// CHECK-NEXT:     0x3D091C IMAGE_REL_ARM64_BRANCH26 foo (14)
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .pdata {
// CHECK-NEXT:     0x0 IMAGE_REL_ARM64_ADDR32NB .text (0)
// CHECK-NEXT:     0x4 IMAGE_REL_ARM64_ADDR32NB .xdata (9)
// CHECK-NEXT:     0x8 IMAGE_REL_ARM64_ADDR32NB .text (0)
// CHECK-NEXT:     0xC IMAGE_REL_ARM64_ADDR32NB .xdata (9)
// CHECK-NEXT:     0x10 IMAGE_REL_ARM64_ADDR32NB $L.text_1 (2)
// CHECK-NEXT:     0x14 IMAGE_REL_ARM64_ADDR32NB .xdata (9)
// CHECK-NEXT:     0x18 IMAGE_REL_ARM64_ADDR32NB $L.text_2 (3)
// CHECK-NEXT:     0x1C IMAGE_REL_ARM64_ADDR32NB .xdata (9)
// CHECK-NEXT:     0x20 IMAGE_REL_ARM64_ADDR32NB $L.text_3 (4)
// CHECK-NEXT:     0x24 IMAGE_REL_ARM64_ADDR32NB .xdata (9)
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK-LABEL: UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: a (0x0)
// CHECK-NEXT:     ExceptionRecord: .xdata (0x0)
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 1048572
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: No
// CHECK-NEXT:       EpilogueScopes: 0
// CHECK-NEXT:       ByteCodeLength: 4
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xd561              ; str x30, [sp, #-16]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: a +0xFFFFC (0xFFFFC)
// CHECK-NEXT:     ExceptionRecord: .xdata +0x8 (0x8)
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 551444
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK-NEXT:       EpilogueOffset: 1
// CHECK-NEXT:       ByteCodeLength: 4
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xe5                ; end_c
// CHECK-NEXT:         0xd561              ; str x30, [sp, #-16]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       Epilogue [
// CHECK-NEXT:         0xd561              ; ldr x30, [sp], #16
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: b (0x186A10)
// CHECK-NEXT:     ExceptionRecord: .xdata +0x10 (0x10)
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 1048572
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: No
// CHECK-NEXT:       EpilogueScopes: 0
// CHECK-NEXT:       ByteCodeLength: 8
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xe1                ; mov fp, sp
// CHECK-NEXT:         0xc81e              ; stp x19, x20, [sp, #240]
// CHECK-NEXT:         0x9f                ; stp x29, x30, [sp, #-256]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: $L.text_2 +0x86A0C (0x286A0C)
// CHECK-NEXT:     ExceptionRecord: .xdata +0x1C (0x1C)
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 1048572
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK-NEXT:       EpilogueOffset: 0
// CHECK-NEXT:       ByteCodeLength: 8
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xe5                ; end_c
// CHECK-NEXT:         0xe1                ; mov fp, sp
// CHECK-NEXT:         0xc81e              ; stp x19, x20, [sp, #240]
// CHECK-NEXT:         0x9f                ; stp x29, x30, [sp, #-256]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: $L.text_3 +0x86A08 (0x386A08)
// CHECK-NEXT:     ExceptionRecord: .xdata +0x28 (0x28)
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 302888
// CHECK-NEXT:       Version: 0
// CHECK-NEXT:       ExceptionData: No
// CHECK-NEXT:       EpiloguePacked: Yes
// CHECK-NEXT:       EpilogueOffset: 1
// CHECK-NEXT:       ByteCodeLength: 8
// CHECK-NEXT:       Prologue [
// CHECK-NEXT:         0xe5                ; end_c
// CHECK-NEXT:         0xe1                ; mov fp, sp
// CHECK-NEXT:         0xc81e              ; stp x19, x20, [sp, #240]
// CHECK-NEXT:         0x9f                ; stp x29, x30, [sp, #-256]!
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:       Epilogue [
// CHECK-NEXT:         0xe1                ; mov sp, fp
// CHECK-NEXT:         0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:         0x9f                ; ldp x29, x30, [sp], #256
// CHECK-NEXT:         0xe4                ; end
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

	.text
// A simple function with an single epilog mirroring the prolog.
	.global	a
	.p2align	2
	.seh_proc a
a:
	str	x30, [sp, #-16]!
	.seh_save_reg_x	x30, 16
	.seh_endprologue
        .rept 400000
        nop
        .endr
	bl	foo
	.seh_startepilogue
	ldr	x30, [sp], #16
	.seh_save_reg_x	x30, 16
	.seh_endepilogue
	ret
	.seh_endfunclet
	.seh_endproc

// Example 1 from https://docs.microsoft.com/en-us/cpp/build/arm64-exception-handling#function-fragments
	.global	b
	.p2align	2
	.seh_proc b
b:
	stp	x29, lr, [sp, #-256]!
	.seh_save_fplr_x 256
	stp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	mov	x29, fp
	.seh_set_fp
	.seh_endprologue
        .rept 600000
        nop
        .endr
	bl	foo
	.seh_startepilogue
	mov	sp, x29
	.seh_set_fp
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
	.seh_endfunclet
	.seh_endproc
