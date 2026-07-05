// This test checks the alignment and padding of the unwind info.

// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -S --sd --sr -u - | FileCheck %s

// CHECK:      Sections [
// CHECK:        Section {
// CHECK:          Name: .xdata
// CHECK:          RawDataSize: 8
// CHECK:          RelocationCount: 0
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK:          Relocations [
// CHECK-NEXT:     ]
// CHECK:          SectionData (
// CHECK-NEXT:       0000: 01000000 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .pdata
// CHECK:          RawDataSize: 12
// CHECK:          RelocationCount: 3
// CHECK:          Characteristics [
// CHECK-NEXT:       IMAGE_SCN_ALIGN_4BYTES
// CHECK-NEXT:       IMAGE_SCN_CNT_INITIALIZED_DATA
// CHECK-NEXT:       IMAGE_SCN_MEM_READ
// CHECK-NEXT:     ]
// CHECK:          Relocations [
// CHECK-NEXT:       [[BeginDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:       [[EndDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB .text
// CHECK-NEXT:       [[UnwindDisp:0x[A-F0-9]+]] IMAGE_REL_AMD64_ADDR32NB .xdata
// CHECK-NEXT:     ]
// CHECK:          SectionData (
// CHECK-NEXT:       0000: 00000000 01000000 00000000
// CHECK-NEXT:     )
// CHECK-NEXT:   }
// CHECK:        UnwindInformation [
// CHECK-NEXT:     RuntimeFunction {
// CHECK-NEXT:     StartAddress: smallFunc {{(\+0x[A-F0-9]+ )?}}([[BeginDisp]])
// CHECK-NEXT:     EndAddress: smallFunc {{(\+0x[A-F0-9]+ )?}}([[EndDisp]])
// CHECK-NEXT:     UnwindInfoAddress: .xdata {{(\+0x[A-F0-9]+ )?}}([[UnwindDisp]])
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
// CHECK-NEXT: ]

// Generate the minimal unwind info.
// It contains only the version set to 1. All other bytes are 0.
    .globl smallFunc
    .def smallFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
smallFunc:
    ret
    .seh_endproc
