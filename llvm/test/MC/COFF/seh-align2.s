// This test checks the alignment and padding of the unwind info.

// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -S --sd --sr -u - | FileCheck %s

// CHECK:      Sections [
// CHECK:        Section {
// CHECK:          Name: .xdata
// CHECK:          RawDataSize: 16
// CHECK:          RelocationCount: 1
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK:          Relocations [
// CHECK-NEXT:       [[HandlerDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB __C_specific_handler
// CHECK-NEXT:     ]
// CHECK:          SectionData (
// CHECK-NEXT:       0000: 09000100 04220000 00000000 BEBAFECA
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT:   Section {
// CHECK:          Name: .pdata
// CHECK:          RawDataSize: 12
// CHECK:          RelocationCount: 3
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_4BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:     ]
// CHECK:          Relocations [
// CHECK-NEXT:       [[BeginDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB func
// CHECK-NEXT:       [[EndDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB func
// CHECK-NEXT:       [[UnwindDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     ]
// CHECK:          SectionData (
// CHECK-NEXT:       0000: FCFFFFFF 05000000 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK-NEXT: ]
// CHECK:      UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     StartAddress: func {{(\+0x[A-F0-9]+ )?}}([[BeginDisp]])
// CHECK-NEXT:     EndAddress: func {{(\+0x[A-F0-9]+ )?}}([[EndDisp]])
// CHECK-NEXT:     UnwindInfoAddress: .xdata {{(\+0x[A-F0-9]+ )?}}([[UnwindDisp]])
// CHECK-NEXT:     UnwindInfo {
// CHECK-NEXT:       Version: 1
// CHECK-NEXT:       Flags [
// CHECK-NEXT:         ExceptionHandler
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 0
// CHECK-NEXT:       FrameRegister: -
// CHECK-NEXT:       FrameOffset: -
// CHECK-NEXT:       UnwindCodeCount: 1
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:         0x04: ALLOC_SMALL size=24
// CHECK-NEXT:       ]
// CHECK-NEXT:       Handler: __C_specific_handler ([[HandlerDisp]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Generates only one unwind code.
// Requires padding of the unwind code array.
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
    subq $24, %rsp
    .seh_stackalloc 24
    .seh_handler __C_specific_handler, @except
    .seh_handlerdata
    .long 0xcafebabe
    .text
    .seh_endprologue
func:
    addq $24, %rsp
    ret
    .seh_endproc
