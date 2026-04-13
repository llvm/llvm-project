// This test checks that the SEH directives emit the correct unwind data.

// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -S -u -r - | FileCheck %s

// CHECK:      Sections [
// CHECK:        Section {
// CHECK:          Name: .text
// CHECK:          RelocationCount: 0
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_CODE
// CHECK-NEXT:       MEM_EXECUTE
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .xdata
// CHECK:          RawDataSize: 80
// CHECK:          RelocationCount: 7
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .pdata
// CHECK:          RelocationCount: 15
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK-NEXT: Relocations [
// CHECK-NEXT:   Section (4) .xdata {
// CHECK-NEXT:     0x14 IMAGE_REL_AMD64_ADDR32NB __C_specific_handler
// CHECK-NEXT:     0x20 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x24 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x28 IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     0x44 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x48 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x4C IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .pdata {
// CHECK-NEXT:     0x0 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x4 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x8 IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     0xC IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x10 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x14 IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     0x18 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x1C IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x20 IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     0x24 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x28 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x2C IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     0x30 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x34 IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:     0x38 IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:   }
// CHECK-NEXT: ]


// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     StartAddress: [[CodeSect1:[^ ]+]] [[BeginDisp1:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     EndAddress: [[CodeSect1]] [[EndDisp1:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     UnwindInfoAddress:
// CHECK-NEXT:     UnwindInfo {
// CHECK-NEXT:       Version: 1
// CHECK-NEXT:       Flags [
// CHECK-NEXT:         ExceptionHandler
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 18
// CHECK-NEXT:       FrameRegister: RBX
// CHECK-NEXT:       FrameOffset: 0x0
// CHECK-NEXT:       UnwindCodeCount: 8
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:         0x12: SET_FPREG reg=RBX, offset=0x0
// CHECK-NEXT:         0x0F: PUSH_NONVOL reg=RBX
// CHECK-NEXT:         0x0E: SAVE_XMM128 reg=XMM8, offset=0x0
// CHECK-NEXT:         0x09: SAVE_NONVOL reg=RSI, offset=0x10
// CHECK-NEXT:         0x04: ALLOC_SMALL size=24
// CHECK-NEXT:         0x00: PUSH_MACHFRAME errcode=yes
// CHECK-NEXT:       ]
// CHECK-NEXT:       Handler: __C_specific_handler
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     StartAddress: [[CodeSect1]] [[BeginDisp2:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     EndAddress: [[CodeSect1]] [[EndDisp2:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     UnwindInfoAddress:
// CHECK-NEXT:     UnwindInfo {
// CHECK-NEXT:       Version: 1
// CHECK-NEXT:       Flags [
// CHECK-NEXT:         ChainInfo
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 0
// CHECK-NEXT:       FrameRegister: -
// CHECK-NEXT:       FrameOffset: -
// CHECK-NEXT:       UnwindCodeCount: 0
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:       ]
// CHECK-NEXT:       Chained {
// CHECK-NEXT:         StartAddress: [[CodeSect1]] [[BeginDisp1]]
// CHECK-NEXT:         EndAddress: [[CodeSect1]] [[EndDisp1]]
// CHECK-NEXT:         UnwindInfoAddress:
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     StartAddress: [[CodeSect3:[^ ]+]] [[BeginDisp3:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     EndAddress: [[CodeSect3]] [[EndDisp3:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     UnwindInfoAddress:
// CHECK-NEXT:     UnwindInfo {
// CHECK-NEXT:       Version: 1
// CHECK-NEXT:       Flags [
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 0
// CHECK-NEXT:       FrameRegister: -
// CHECK-NEXT:       FrameOffset: -
// CHECK-NEXT:       UnwindCodeCount: 0
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     StartAddress: [[CodeSect4:[^ ]+]] [[BeginDisp4:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     EndAddress: [[CodeSect4]] [[EndDisp4:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     UnwindInfoAddress:
// CHECK-NEXT:     UnwindInfo {
// CHECK-NEXT:       Version: 1
// CHECK-NEXT:       Flags [ (0x0)
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 1
// CHECK-NEXT:       FrameRegister: -
// CHECK-NEXT:       FrameOffset: -
// CHECK-NEXT:       UnwindCodeCount: 1
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:         0x01: PUSH_NONVOL reg=RBX
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     StartAddress: [[CodeSect4]] [[BeginDisp5:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     EndAddress: [[CodeSect4]] [[EndDisp5:(\+0x[A-F0-9]+)?]]
// CHECK-NEXT:     UnwindInfoAddress:
// CHECK-NEXT:     UnwindInfo {
// CHECK-NEXT:       Version: 1
// CHECK-NEXT:       Flags [ (0x4)
// CHECK-NEXT:         ChainInfo (0x4)
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 1
// CHECK-NEXT:       FrameRegister: -
// CHECK-NEXT:       FrameOffset: -
// CHECK-NEXT:       UnwindCodeCount: 1
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:         0x01: PUSH_NONVOL reg=RSI
// CHECK-NEXT:       ]
// CHECK-NEXT:       Chained {
// CHECK-NEXT:         StartAddress: [[CodeSect4]] [[BeginDisp4]]
// CHECK-NEXT:         EndAddress: [[CodeSect4]] [[EndDisp4]]
// CHECK-NEXT:         UnwindInfoAddress:
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
func:
    .seh_pushframe @code
    subq $24, %rsp
    .seh_stackalloc 24
    movq %rsi, 16(%rsp)
    .seh_savereg %rsi, 16
    movups %xmm8, (%rsp)
    .seh_savexmm %xmm8, 0
    pushq %rbx
    .seh_pushreg %rbx
    mov %rsp, %rbx
    .seh_setframe 3, 0
    .seh_endprologue
    .seh_handler __C_specific_handler, @except
    .seh_handlerdata
    .long 0
    .text
    .seh_endprologue
    .seh_splitchained
    .seh_endprologue
    .seh_startepilogue
    lea (%rbx), %rsp
    pop %rbx
    addq $24, %rsp
    .seh_endepilogue
    ret
    .seh_endproc

// Test emission of small functions.
    .globl smallFunc
    .def smallFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
smallFunc:
    ret
    .seh_endproc

// Test chaining prologs together.
    .globl split_prolog
    .def split_prolog; .scl 2; .type 32; .endef
    .seh_proc split_prolog
split_prolog:
    pushq %rbx
    .seh_pushreg %rbx
    .seh_endprologue
    .seh_splitchained
    pushq %rsi
    .seh_pushreg %rsi
    .seh_endprologue
    .seh_startepilogue
    pop %rsi
    pop %rbx
    .seh_endepilogue
    ret
    .seh_endproc
