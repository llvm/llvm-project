// RUN: llvm-mc -triple x86_64-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// CHECK:       UnwindInformation [

.text

single_epilog_atend:
    .seh_proc stack_alloc_no_pushes
    .seh_unwindversion 2
    subq    $40, %rsp
    .seh_stackalloc 40
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: single_epilog_atend
// CHECK-NEXT:   EndAddress: single_epilog_atend +0xF
// CHECK-NEXT:   UnwindInfoAddress: .xdata
// CHECK-NEXT:   UnwindInfo {
// CHECK-NEXT:     Version: 2
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     PrologSize: 4
// CHECK-NEXT:     FrameRegister: -
// CHECK-NEXT:     FrameOffset: -
// CHECK-NEXT:     UnwindCodeCount: 3
// CHECK-NEXT:     UnwindCodes [
// CHECK-NEXT:       0x01: EPILOG atend=yes, length=0x1
// CHECK-NEXT:       0x00: EPILOG padding
// CHECK-NEXT:       0x04: ALLOC_SMALL size=40
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

single_epilog_notatend:
    .seh_proc stack_alloc_no_pushes
    .seh_unwindversion 2
    subq    $40, %rsp
    .seh_stackalloc 40
    .seh_endprologue
    callq   a
    nop
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    retq
    nop
    .seh_endproc
// CHECK-LABEL:  StartAddress: single_epilog_notatend
// CHECK-NEXT:   EndAddress: single_epilog_notatend +0x10
// CHECK-NEXT:   UnwindInfoAddress: .xdata +0xC
// CHECK-NEXT:   UnwindInfo {
// CHECK-NEXT:     Version: 2
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     PrologSize: 4
// CHECK-NEXT:     FrameRegister: -
// CHECK-NEXT:     FrameOffset: -
// CHECK-NEXT:     UnwindCodeCount: 3
// CHECK-NEXT:     UnwindCodes [
// CHECK-NEXT:       0x01: EPILOG atend=no, length=0x1
// CHECK-NEXT:       0x02: EPILOG offset=0x2
// CHECK-NEXT:       0x04: ALLOC_SMALL size=40
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

multiple_epilogs:
    .seh_proc multiple_epilogs
    .seh_unwindversion 2
    subq    $40, %rsp
    .seh_stackalloc 40
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_ELSE_1
    movl    %eax, %ecx
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     c
.L_ELSE_1:
    nop
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     b
    .seh_endproc
// CHECK-LABEL:  StartAddress: multiple_epilogs
// CHECK-NEXT:   EndAddress: multiple_epilogs +0x22
// CHECK-NEXT:   UnwindInfoAddress: .xdata +0x18
// CHECK-NEXT:   UnwindInfo {
// CHECK-NEXT:     Version: 2
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     PrologSize: 4
// CHECK-NEXT:     FrameRegister: -
// CHECK-NEXT:     FrameOffset: -
// CHECK-NEXT:     UnwindCodeCount: 5
// CHECK-NEXT:     UnwindCodes [
// CHECK-NEXT:       0x01: EPILOG atend=no, length=0x1
// CHECK-NEXT:       0x05: EPILOG offset=0x5
// CHECK-NEXT:       0x0F: EPILOG offset=0xF
// CHECK-NEXT:       0x00: EPILOG padding
// CHECK-NEXT:       0x04: ALLOC_SMALL size=40
// CHECK-NEXT:     ]
// CHECK-NEXT:   }

mismatched_terminators:
    .seh_proc mismatched_terminators
    .seh_unwindversion 2
    subq    $40, %rsp
    .seh_stackalloc 40
    .seh_endprologue
    callq   b
    testl   %eax, %eax
    jle     .L_ELSE_1
# %bb.2:
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     b
.L_ELSE_2:
    nop
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    retq
    .seh_endproc
// CHECK-LABEL:  StartAddress: mismatched_terminators
// CHECK-NEXT:   EndAddress: mismatched_terminators +0x1C
// CHECK-NEXT:   UnwindInfoAddress: .xdata +0x28
// CHECK-NEXT:   UnwindInfo {
// CHECK-NEXT:     Version: 2
// CHECK-NEXT:     Flags [ (0x0)
// CHECK-NEXT:     ]
// CHECK-NEXT:     PrologSize: 4
// CHECK-NEXT:     FrameRegister: -
// CHECK-NEXT:     FrameOffset: -
// CHECK-NEXT:     UnwindCodeCount: 3
// CHECK-NEXT:     UnwindCodes [
// CHECK-NEXT:       0x01: EPILOG atend=yes, length=0x1
// CHECK-NEXT:       0x0B: EPILOG offset=0xB
// CHECK-NEXT:       0x04: ALLOC_SMALL size=40
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
