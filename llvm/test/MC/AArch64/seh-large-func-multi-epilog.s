// This test checks that we emit unwind info correctly for epilogs that:
// 1. mirror the prolog; or
// 2. are subsequence at the end of the prolog; or
// 3. neither of above two.
// in the same segment. 
// RUN: llvm-mc -triple aarch64-pc-win32 -filetype=obj %s -o %t.o
// RUN: llvm-readobj -S -r -u %t.o | FileCheck %s

// CHECK:       Section {
// CHECK:         Number: 4
// CHECK-NEXT:    Name: .xdata (2E 78 64 61 74 61 00 00)
// CHECK-NEXT:    VirtualSize: 0x0
// CHECK-NEXT:    VirtualAddress: 0x0
// CHECK-NEXT:    RawDataSize: 80
// CHECK-NEXT:    PointerToRawData: 0x1251AC
// CHECK-NEXT:    PointerToRelocations: 0x0
// CHECK-NEXT:    PointerToLineNumbers: 0x0
// CHECK-NEXT:    RelocationCount: 0
// CHECK-NEXT:    LineNumberCount: 0
// CHECK-NEXT:    Characteristics [ (0x40300040)
// CHECK-NEXT:      IMAGE_SCN_ALIGN_4BYTES (0x300000)
// CHECK-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// CHECK-NEXT:      IMAGE_SCN_MEM_READ (0x40000000)
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  Section {
// CHECK-NEXT:    Number: 5
// CHECK-NEXT:    Name: .pdata (2E 70 64 61 74 61 00 00)
// CHECK-NEXT:    VirtualSize: 0x0
// CHECK-NEXT:    VirtualAddress: 0x0
// CHECK-NEXT:    RawDataSize: 16
// CHECK-NEXT:    PointerToRawData: 0x1251FC
// CHECK-NEXT:    PointerToRelocations: 0x12520C
// CHECK-NEXT:    PointerToLineNumbers: 0x0
// CHECK-NEXT:    RelocationCount: 4
// CHECK-NEXT:    LineNumberCount: 0
// CHECK-NEXT:    Characteristics [ (0x40300040)
// CHECK-NEXT:      IMAGE_SCN_ALIGN_4BYTES (0x300000)
// CHECK-NEXT:      IMAGE_SCN_CNT_INITIALIZED_DATA (0x40)
// CHECK-NEXT:      IMAGE_SCN_MEM_READ (0x40000000)
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:]
// CHECK-LABEL:Relocations [
// CHECK-NEXT:  Section (1) .text {
// CHECK-NEXT:    0x94 IMAGE_REL_ARM64_BRANCH26 foo (12)
// CHECK-NEXT:    0x125068 IMAGE_REL_ARM64_BRANCH26 foo (12)
// CHECK-NEXT:  }
// CHECK-NEXT:  Section (5) .pdata {
// CHECK-NEXT:    0x0 IMAGE_REL_ARM64_ADDR32NB .text (0)
// CHECK-NEXT:    0x4 IMAGE_REL_ARM64_ADDR32NB .xdata (7)
// CHECK-NEXT:    0x8 IMAGE_REL_ARM64_ADDR32NB .text (0)
// CHECK-NEXT:    0xC IMAGE_REL_ARM64_ADDR32NB .xdata (7)
// CHECK-NEXT:  }
// CHECK-NEXT:]
// CHECK-LABEL:UnwindInformation [
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: multi_epilog (0x0)
// CHECK-NEXT:    ExceptionRecord: .xdata (0x0)
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength: 1048572
// CHECK-NEXT:      Version: 0
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: No
// CHECK-NEXT:      EpilogueScopes: 3
// CHECK-NEXT:      ByteCodeLength: 24
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xe1                ; mov fp, sp
// CHECK-NEXT:        0xca16              ; stp x27, x28, [sp, #176]
// CHECK-NEXT:        0xc998              ; stp x25, x26, [sp, #192]
// CHECK-NEXT:        0xc91a              ; stp x23, x24, [sp, #208]
// CHECK-NEXT:        0xc89c              ; stp x21, x22, [sp, #224]
// CHECK-NEXT:        0xc81e              ; stp x19, x20, [sp, #240]
// CHECK-NEXT:        0x9f                ; stp x29, x30, [sp, #-256]!
// CHECK-NEXT:        0xe4                ; end
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 38
// CHECK-NEXT:          EpilogueStartIndex: 0
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xe1                ; mov sp, fp
// CHECK-NEXT:            0xca16              ; ldp x27, x28, [sp, #176]
// CHECK-NEXT:            0xc998              ; ldp x25, x26, [sp, #192]
// CHECK-NEXT:            0xc91a              ; ldp x23, x24, [sp, #208]
// CHECK-NEXT:            0xc89c              ; ldp x21, x22, [sp, #224]
// CHECK-NEXT:            0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:            0x9f                ; ldp x29, x30, [sp], #256
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 46
// CHECK-NEXT:          EpilogueStartIndex: 3
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xc998              ; ldp x25, x26, [sp, #192]
// CHECK-NEXT:            0xc91a              ; ldp x23, x24, [sp, #208]
// CHECK-NEXT:            0xc89c              ; ldp x21, x22, [sp, #224]
// CHECK-NEXT:            0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:            0x9f                ; ldp x29, x30, [sp], #256
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 52
// CHECK-NEXT:          EpilogueStartIndex: 13
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xe1                ; mov sp, fp
// CHECK-NEXT:            0xc91a              ; ldp x23, x24, [sp, #208]
// CHECK-NEXT:            0xc89c              ; ldp x21, x22, [sp, #224]
// CHECK-NEXT:            0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:            0x9f                ; ldp x29, x30, [sp], #256
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: multi_epilog +0xFFFFC (0xFFFFC)
// CHECK-NEXT:    ExceptionRecord: .xdata +0x28 (0x28)
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength: 151744
// CHECK-NEXT:      Version: 0
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: No
// CHECK-NEXT:      EpilogueScopes: 3
// CHECK-NEXT:      ByteCodeLength: 24
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xe5                ; end_c
// CHECK-NEXT:        0xe1                ; mov fp, sp
// CHECK-NEXT:        0xca16              ; stp x27, x28, [sp, #176]
// CHECK-NEXT:        0xc998              ; stp x25, x26, [sp, #192]
// CHECK-NEXT:        0xc91a              ; stp x23, x24, [sp, #208]
// CHECK-NEXT:        0xc89c              ; stp x21, x22, [sp, #224]
// CHECK-NEXT:        0xc81e              ; stp x19, x20, [sp, #240]
// CHECK-NEXT:        0x9f                ; stp x29, x30, [sp, #-256]!
// CHECK-NEXT:        0xe4                ; end
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 37916
// CHECK-NEXT:          EpilogueStartIndex: 1
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xe1                ; mov sp, fp
// CHECK-NEXT:            0xca16              ; ldp x27, x28, [sp, #176]
// CHECK-NEXT:            0xc998              ; ldp x25, x26, [sp, #192]
// CHECK-NEXT:            0xc91a              ; ldp x23, x24, [sp, #208]
// CHECK-NEXT:            0xc89c              ; ldp x21, x22, [sp, #224]
// CHECK-NEXT:            0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:            0x9f                ; ldp x29, x30, [sp], #256
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 37924
// CHECK-NEXT:          EpilogueStartIndex: 4
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xc998              ; ldp x25, x26, [sp, #192]
// CHECK-NEXT:            0xc91a              ; ldp x23, x24, [sp, #208]
// CHECK-NEXT:            0xc89c              ; ldp x21, x22, [sp, #224]
// CHECK-NEXT:            0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:            0x9f                ; ldp x29, x30, [sp], #256
// CHECK-NEXT:            0xe4                ; end
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 37930
// CHECK-NEXT:          EpilogueStartIndex: 14
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xe1                ; mov sp, fp
// CHECK-NEXT:            0xc91a              ; ldp x23, x24, [sp, #208]
// CHECK-NEXT:            0xc89c              ; ldp x21, x22, [sp, #224]
// CHECK-NEXT:            0xc81e              ; ldp x19, x20, [sp, #240]
// CHECK-NEXT:            0x9f                ; ldp x29, x30, [sp], #256
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
	stp	x29, lr, [sp, #-256]!
	.seh_save_fplr_x 256
	stp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x25, x26, [sp, #192]
	.seh_save_regp x25, 192
	stp	x27, x28, [sp, #176]
	.seh_save_regp x27, 176
	mov	x29, fp
	.seh_set_fp
	.seh_endprologue
        .rept 30
        nop
        .endr
	bl	foo
// Epilogs 1, 2 and 3 are in the same segment as prolog.
// epilog1 - mirroring prolog
	.seh_startepilogue
	mov	sp, x29
	.seh_set_fp
	stp	x27, x28, [sp, #176]
	.seh_save_regp x27, 176
	stp	x25, x26, [sp, #192]
	.seh_save_regp x25, 192
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
// epilog2 - a subsequence at the end of prolog, can use prolog's opcodes.
	.seh_startepilogue
	stp	x25, x26, [sp, #192]
	.seh_save_regp x25, 192
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
// epilog3 - cannot use prolog's opcode.
	.seh_startepilogue
	mov	sp, x29
	.seh_set_fp
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
        .rept 300000
        nop
        .endr
	bl	foo
// Epilogs below are in a segment without prolog
// epilog4 - mirroring prolog, its start index should be 1, counting the end_c. 
	.seh_startepilogue
	mov	sp, x29
	.seh_set_fp
	stp	x27, x28, [sp, #176]
	.seh_save_regp x27, 176
	stp	x25, x26, [sp, #192]
	.seh_save_regp x25, 192
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
// epilog5 - same as epilog2, its start index should be: 1 + epilog2's index.
	.seh_startepilogue
	stp	x25, x26, [sp, #192]
	.seh_save_regp x25, 192
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
// epilog6 - same as epilog3, cannot use prolog's opcode. Again its start index
//           should be: 1 + epilog3's index.
	.seh_startepilogue
	mov	sp, x29
	.seh_set_fp
	stp	x23, x24, [sp, #208]
	.seh_save_regp x23, 208
	stp	x21, x22, [sp, #224]
	.seh_save_regp x21, 224
	ldp	x19, x20, [sp, #240]
	.seh_save_regp x19, 240
	ldp	x29, lr, [sp], #256
	.seh_save_fplr_x 256
	.seh_endepilogue
	ret
	.seh_endfunclet
	.seh_endproc
