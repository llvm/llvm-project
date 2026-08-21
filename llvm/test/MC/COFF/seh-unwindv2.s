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

chained:
    .seh_proc chained
    .seh_unwindversion 2
    subq    $40, %rsp
    .seh_stackalloc 40
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_ELSE_3
    movl    %eax, %ecx
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     c
    .seh_splitchained
    .seh_endprologue
.L_ELSE_3:
    nop
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     b
    .seh_endproc

// CHECK:         RuntimeFunction {
// CHECK-NEXT:    StartAddress: chained (0x30)
// CHECK-NEXT:    EndAddress: chained [[EndDisp1:\+0x[A-F0-9]+]] (0x34)
// CHECK-NEXT:    UnwindInfoAddress: .xdata [[InfoDisp1:\+0x[A-F0-9]+]] (0x38)
// CHECK-NEXT:    UnwindInfo {
// CHECK-NEXT:      Version: 2
// CHECK-NEXT:      Flags [ (0x0)
// CHECK-NEXT:      ]
// CHECK-NEXT:      PrologSize: 4
// CHECK-NEXT:      FrameRegister: -
// CHECK-NEXT:      FrameOffset: -
// CHECK-NEXT:      UnwindCodeCount: 3
// CHECK-NEXT:      UnwindCodes [
// CHECK-NEXT:        0x01: EPILOG atend=no, length=0x1
// CHECK-NEXT:        0x05: EPILOG offset=0x5
// CHECK-NEXT:        0x04: ALLOC_SMALL size=40
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    StartAddress: chained [[EndDisp1]] (0x3C)
// CHECK-NEXT:    EndAddress: chained +0x22 (0x40)
// CHECK-NEXT:    UnwindInfoAddress: .xdata +0x40 (0x44)
// CHECK-NEXT:    UnwindInfo {
// CHECK-NEXT:      Version: 2
// CHECK-NEXT:      Flags [ (0x4)
// CHECK-NEXT:        ChainInfo (0x4)
// CHECK-NEXT:      ]
// CHECK-NEXT:      PrologSize: 0
// CHECK-NEXT:      FrameRegister: -
// CHECK-NEXT:      FrameOffset: -
// CHECK-NEXT:      UnwindCodeCount: 2
// CHECK-NEXT:      UnwindCodes [
// CHECK-NEXT:        0x01: EPILOG atend=no, length=0x1
// CHECK-NEXT:        0x05: EPILOG offset=0x5
// CHECK-NEXT:      ]
// CHECK-NEXT:      Chained {
// CHECK-NEXT:        StartAddress: chained (0x48)
// CHECK-NEXT:        EndAddress: chained [[EndDisp1]] (0x4C)
// CHECK-NEXT:        UnwindInfoAddress: .xdata [[InfoDisp1]] (0x50)
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

has_ex_handler_data:
    .seh_proc has_ex_handler_data
    .seh_handler __C_specific_handler, @unwind, @except
    .seh_unwindversion 2
    pushq   %rbp
    .seh_pushreg %rbp
    subq    $32, %rsp
    .seh_stackalloc 32
    leaq    32(%rsp), %rbp
    .seh_setframe %rbp, 32
    .seh_endprologue
.has_ex_handler_data_callsite:
    callq   *__imp_callme(%rip)
    nop
.has_ex_handler_data_finish:
    .seh_startepilogue
    addq    $32, %rsp
    .seh_unwindv2start
    popq    %rbp
    .seh_endepilogue
    retq
.has_ex_handler_data_handler:
    jmp     .has_ex_handler_data_finish
    .seh_handlerdata
    .long   1  # Number of call sites
    .long   .has_ex_handler_data_callsite@IMGREL    # LabelStart
    .long   .has_ex_handler_data_finish@IMGREL      # LabelEnd
    .long   1                                       # CatchAll
    .long   .has_ex_handler_data_handler@IMGREL     # ExceptionHandler
    .text
    .seh_endproc
// CHECK-LABEL:  StartAddress: has_ex_handler_data
// CHECK-NEXT:   EndAddress: .has_ex_handler_data_handler +0x2
// CHECK-NEXT:   UnwindInfoAddress: .xdata +0x54
// CHECK-NEXT:   UnwindInfo {
// CHECK-NEXT:     Version: 2
// CHECK-NEXT:     Flags [ (0x3)
// CHECK-NEXT:      ExceptionHandler (0x1)
// CHECK-NEXT:      TerminateHandler (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     PrologSize: 10
// CHECK-NEXT:     FrameRegister: RBP
// CHECK-NEXT:     FrameOffset: 0x2
// CHECK-NEXT:     UnwindCodeCount: 5
// CHECK-NEXT:     UnwindCodes [
// CHECK-NEXT:       0x02: EPILOG atend=no, length=0x2
// CHECK-NEXT:       0x04: EPILOG offset=0x4
// CHECK-NEXT:       0x0A: SET_FPREG reg=RBP, offset=0x20
// CHECK-NEXT:       0x05: ALLOC_SMALL size=32
// CHECK-NEXT:       0x01: PUSH_NONVOL reg=RBP
// CHECK-NEXT:     ]
// CHECK-NEXT:     Handler: __C_specific_handler
// CHECK-NEXT:   }

has_ex_handler_data_and_chaining:
    .seh_proc chained
    .seh_handler __C_specific_handler, @unwind, @except
    .seh_unwindversion 2
    subq    $40, %rsp
    .seh_stackalloc 40
    .seh_endprologue
    callq   c
    testl   %eax, %eax
    jle     .L_ELSE_4
.has_ex_handler_data_and_chaining_callsite:
    callq   *__imp_callme(%rip)
    nop
.has_ex_handler_data_and_chaining_finish:
    movl    %eax, %ecx
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     c
    .seh_splitchained
    .seh_endprologue
.L_ELSE_4:
    nop
    .seh_startepilogue
    addq    $40, %rsp
    .seh_unwindv2start
    .seh_endepilogue
    jmp     b
.has_ex_handler_data_and_chaining_handler:
    jmp     .has_ex_handler_data_and_chaining_finish
    .seh_handlerdata
    .long   1  # Number of call sites
    .long   .has_ex_handler_data_and_chaining_callsite@IMGREL   # LabelStart
    .long   .has_ex_handler_data_and_chaining_finish@IMGREL     # LabelEnd
    .long   1                                                   # CatchAll
    .long   .has_ex_handler_data_and_chaining_handler@IMGREL    # ExceptionHandler
    .text
    .seh_endproc

// CHECK:         RuntimeFunction {
// CHECK-NEXT:    StartAddress: has_ex_handler_data_and_chaining (0x54)
// CHECK-NEXT:    EndAddress: .has_ex_handler_data_and_chaining_finish [[EndDisp2:\+0x[A-F0-9]+]] (0x58)
// CHECK-NEXT:    UnwindInfoAddress: .xdata [[InfoDisp2:\+0x[A-F0-9]+]] (0x5C)
// CHECK-NEXT:    UnwindInfo {
// CHECK-NEXT:      Version: 2
// CHECK-NEXT:      Flags [ (0x3)
// CHECK-NEXT:       ExceptionHandler (0x1)
// CHECK-NEXT:       TerminateHandler (0x2)
// CHECK-NEXT:      ]
// CHECK-NEXT:      PrologSize: 4
// CHECK-NEXT:      FrameRegister: -
// CHECK-NEXT:      FrameOffset: -
// CHECK-NEXT:      UnwindCodeCount: 3
// CHECK-NEXT:      UnwindCodes [
// CHECK-NEXT:        0x01: EPILOG atend=no, length=0x1
// CHECK-NEXT:        0x05: EPILOG offset=0x5
// CHECK-NEXT:        0x04: ALLOC_SMALL size=40
// CHECK-NEXT:      ]
// CHECK-NEXT:      Handler: __C_specific_handler
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    StartAddress: .has_ex_handler_data_and_chaining_finish [[EndDisp2]] (0x60)
// CHECK-NEXT:    EndAddress: .has_ex_handler_data_and_chaining_handler +0x2 (0x64)
// CHECK-NEXT:    UnwindInfoAddress: .xdata +0xA0 (0x68)
// CHECK-NEXT:    UnwindInfo {
// CHECK-NEXT:      Version: 2
// CHECK-NEXT:      Flags [ (0x4)
// CHECK-NEXT:        ChainInfo (0x4)
// CHECK-NEXT:      ]
// CHECK-NEXT:      PrologSize: 0
// CHECK-NEXT:      FrameRegister: -
// CHECK-NEXT:      FrameOffset: -
// CHECK-NEXT:      UnwindCodeCount: 2
// CHECK-NEXT:      UnwindCodes [
// CHECK-NEXT:        0x01: EPILOG atend=no, length=0x1
// CHECK-NEXT:        0x07: EPILOG offset=0x7
// CHECK-NEXT:      ]
// CHECK-NEXT:      Chained {
// CHECK-NEXT:        StartAddress: has_ex_handler_data_and_chaining (0xA8)
// CHECK-NEXT:        EndAddress: .has_ex_handler_data_and_chaining_finish [[EndDisp2]] (0xAC)
// CHECK-NEXT:        UnwindInfoAddress: .xdata [[InfoDisp2]] (0xB0)
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
