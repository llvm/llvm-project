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
// CHECK-NEXT:       0000: 19000200 04D002C0 00000000 BEBAFECA
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
// CHECK-NEXT:         TerminateHandler
// CHECK-NEXT:       ]
// CHECK-NEXT:       PrologSize: 0
// CHECK-NEXT:       FrameRegister: -
// CHECK-NEXT:       FrameOffset: -
// CHECK-NEXT:       UnwindCodeCount: 2
// CHECK-NEXT:       UnwindCodes [
// CHECK-NEXT:         0x04: PUSH_NONVOL reg=R13
// CHECK-NEXT:         0x02: PUSH_NONVOL reg=R12
// CHECK-NEXT:       ]
// CHECK-NEXT:       Handler: __C_specific_handler ([[HandlerDisp]])
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Generates two unwind codes.
// Requires no padding of the unwind code array.
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
    push %r12
    .seh_pushreg %r12
    push %r13
    .seh_pushreg %r13
    .seh_handler __C_specific_handler, @except, @unwind
    .seh_handlerdata
    .long 0xcafebabe
    .text
    .seh_endprologue
func:
    pop %r13
    pop %r12
    ret
    .seh_endproc
