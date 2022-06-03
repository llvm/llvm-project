// This test checks various cases around generating packed unwind info.

// RUN: llvm-mc -triple thumbv7-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: func6
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 8
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func7
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 8
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r4}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r4}
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func8
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 10
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r4, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r4, lr}
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func9
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 24
// CHECK-NEXT:    ReturnType: b.w <target>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 32
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #32
// CHECK-NEXT:      vpush {d8}
// CHECK-NEXT:      push {lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #32
// CHECK-NEXT:      vpop {d8}
// CHECK-NEXT:      pop {lr}
// CHECK-NEXT:      b.w <target>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func10
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 26
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 1
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: Yes
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #16
// CHECK-NEXT:      vpush {d8-d9}
// CHECK-NEXT:      mov r11, sp
// CHECK-NEXT:      push {r11, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      vpop {d8-d9}
// CHECK-NEXT:      pop {r11, lr}
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func11
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 24
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 1
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: Yes
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #16
// CHECK-NEXT:      vpush {d8-d9}
// CHECK-NEXT:      mov r11, sp
// CHECK-NEXT:      push {r11, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      vpop {d8-d9}
// CHECK-NEXT:      pop {r11, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func12
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 18
// CHECK-NEXT:    ReturnType: b.w <target>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 6
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #16
// CHECK-NEXT:      vpush {d8-d14}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      vpop {d8-d14}
// CHECK-NEXT:      b.w <target>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func13
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 18
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 6
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: Yes
// CHECK-NEXT:    StackAdjustment: 20
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #20
// CHECK-NEXT:      add.w r11, sp, #28
// CHECK-NEXT:      push {r4-r11, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #20
// CHECK-NEXT:      pop {r4-r11, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func14
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 14
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 20
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #20
// CHECK-NEXT:      push {r4-r11, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #20
// CHECK-NEXT:      pop {r4-r11, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func15
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 20
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 512
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #512
// CHECK-NEXT:      push {r4, lr}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #512
// CHECK-NEXT:      pop {r4}
// CHECK-NEXT:      ldr pc, [sp], #20
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func16
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 20
// CHECK-NEXT:    ReturnType: b.w <target>
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: Yes
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      mov r11, sp
// CHECK-NEXT:      push {r11, lr}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r11, lr}
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      b.w <target>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func17
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 20
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 512
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #512
// CHECK-NEXT:      push {r4}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #512
// CHECK-NEXT:      pop {r4}
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func18
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 6
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 4
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r3, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r3, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func19
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 12
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r0-r4}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r0-r4}
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func20
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 14
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r0-r4}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      pop {r4}
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func21
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 14
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #16
// CHECK-NEXT:      push {r4}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r0-r4}
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func22
// CHECK-NEXT:    Fragment: Yes
// CHECK-NEXT:    FunctionLength: 14
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 512
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #512
// CHECK-NEXT:      push {r4, lr}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #512
// CHECK-NEXT:      pop {r4}
// CHECK-NEXT:      ldr pc, [sp], #20
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func24
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 16
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 3
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: Yes
// CHECK-NEXT:    StackAdjustment: 8
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      add.w r11, sp, #24
// CHECK-NEXT:      push {r2-r7, r11, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #8
// CHECK-NEXT:      pop {r4-r7, r11, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func25
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 16
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 3
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: Yes
// CHECK-NEXT:    StackAdjustment: 8
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #8
// CHECK-NEXT:      add.w r11, sp, #16
// CHECK-NEXT:      push {r4-r7, r11, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r2-r7, r11, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func26
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 8
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 12
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r1-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #12
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func27
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 8
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 12
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #12
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r1-r3}
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func28
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 8
// CHECK-NEXT:    ReturnType: bx <reg>
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func29
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 10
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: Yes
// CHECK-NEXT:    Reg: 7
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {lr}
// CHECK-NEXT:      push {r0-r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      ldr pc, [sp], #20
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func30
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 6
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 2
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r4-r6, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      pop {r4-r6, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func31
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 4
// CHECK-NEXT:    ReturnType: (no epilogue)
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 2
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 0
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      push {r4-r6, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }

// CHECK:         Function: notpacked1
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked2
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked3
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked4
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked5
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked6
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked7
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked8
// CHECK-NEXT:    ExceptionRecord:
// CHECK:         Function: notpacked9
// CHECK-NEXT:    ExceptionRecord:

        .text
        .syntax unified

        .seh_proc func6
func6:
        .seh_endprologue
        nop
        nop
        nop
        .seh_startepilogue
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func7
func7:
        push {r4}
        .seh_save_regs {r4}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4}
        .seh_save_regs {r4}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func8
func8:
        push {r4,lr}
        .seh_save_regs {r4,lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop.w {r4,lr}
        .seh_save_regs_w {r4,lr}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func9
func9:
        push {lr}
        .seh_save_regs {lr}
        vpush {d8}
        .seh_save_fregs {d8}
        sub sp, sp, #32
        .seh_stackalloc 32
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #32
        .seh_stackalloc 32
        vpop {d8}
        .seh_save_fregs {d8}
        pop.w {lr}
        .seh_save_regs_w {lr}
        b.w tailcall
        .seh_nop_w
        .seh_endepilogue
        .seh_endproc

        .seh_proc func10
func10:
        push.w {r11,lr}
        .seh_save_regs_w {r11,lr}
        mov r11, sp
        .seh_nop
        vpush {d8-d9}
        .seh_save_fregs {d8-d9}
        sub sp, sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #16
        .seh_stackalloc 16
        vpop {d8-d9}
        .seh_save_fregs {d8-d9}
        pop.w {r11,lr}
        .seh_save_regs_w {r11,lr}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func11
func11:
        push.w {r11,lr}
        .seh_save_regs_w {r11,lr}
        mov r11, sp
        .seh_nop
        vpush {d8-d9}
        .seh_save_fregs {d8-d9}
        sub sp, sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #16
        .seh_stackalloc 16
        vpop {d8-d9}
        .seh_save_fregs {d8-d9}
        pop.w {r11,pc}
        .seh_save_regs_w {r11,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func12
func12:
        vpush {d8-d14}
        .seh_save_fregs {d8-d14}
        sub sp, sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #16
        .seh_stackalloc 16
        vpop {d8-d14}
        .seh_save_fregs {d8-d14}
        b.w tailcall
        .seh_nop_w
        .seh_endepilogue
        .seh_endproc

        .seh_proc func13
func13:
        push.w {r4-r11,lr}
        .seh_save_regs_w {r4-r11,lr}
        add.w r11, sp, #0x1c
        .seh_nop_w
        sub sp, sp, #20
        .seh_stackalloc 20
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #20
        .seh_stackalloc 20
        pop.w {r4-r11,pc}
        .seh_save_regs_w {r4-r11,lr}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func14
func14:
        push.w {r4-r11,lr}
        .seh_save_regs_w {r4-r11,lr}
        sub sp, sp, #20
        .seh_stackalloc 20
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #20
        .seh_stackalloc 20
        pop.w {r4-r11,pc}
        .seh_save_regs_w {r4-r11,lr}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func15
func15:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push {r4,lr}
        .seh_save_regs {r4,lr}
        sub.w sp, sp, #512
        .seh_stackalloc_w 512
        .seh_endprologue
        nop
        .seh_startepilogue
        add.w sp, sp, #512
        .seh_stackalloc_w 512
        pop {r4}
        .seh_save_regs {r4}
        ldr pc, [sp], #20
        .seh_save_lr 20
        .seh_endepilogue
        .seh_endproc

        .seh_proc func16
func16:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push.w {r11,lr}
        .seh_save_regs_w {r11,lr}
        mov r11, sp
        .seh_nop
        .seh_endprologue
        nop
        .seh_startepilogue
        pop.w {r11, lr}
        .seh_save_regs_w {r11,lr}
        add sp, sp, #16
        .seh_stackalloc 16
        b.w tailcall
        .seh_nop_w
        .seh_endepilogue
        .seh_endproc

        .seh_proc func17
func17:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push {r4}
        .seh_save_regs {r4}
        sub.w sp, sp, #512
        .seh_stackalloc_w 512
        .seh_endprologue
        nop
        .seh_startepilogue
        add.w sp, sp, #512
        .seh_stackalloc_w 512
        pop {r4}
        .seh_save_regs {r4}
        add sp, sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func18
func18:
        push {r3,lr}
        .seh_save_regs {r3,lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r3,pc}
        .seh_save_regs {r3,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func19
func19:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push {r0-r4}
        .seh_save_regs {r0-r4}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r0-r4}
        .seh_save_regs {r0-r4}
        add sp, sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func20
func20:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push {r0-r4}
        .seh_save_regs {r0-r4}
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #16
        .seh_stackalloc 16
        pop {r4}
        .seh_save_regs {r4}
        add sp, sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func21
func21:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push {r4}
        .seh_save_regs {r4}
        sub sp, sp, #16
        .seh_stackalloc 16
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r0-r4}
        .seh_save_regs {r0-r4}
        add sp, sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func22
func22:
        .seh_save_regs {r0-r3}
        .seh_save_regs {r4,lr}
        .seh_stackalloc_w 512
        .seh_endprologue_fragment
        nop
        nop
        .seh_startepilogue
        add.w sp, sp, #512
        .seh_stackalloc_w 512
        pop {r4}
        .seh_save_regs {r4}
        ldr pc, [sp], #20
        .seh_save_lr 20
        .seh_endepilogue
        .seh_endproc

        .seh_proc func24
func24:
        push.w {r2-r7,r11,lr}
        .seh_save_regs_w {r2-r7,r11,lr}
        add.w r11, sp, #24
        .seh_nop_w
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #8
        .seh_stackalloc 8
        pop.w {r4-r7,r11,pc}
        .seh_save_regs_w {r4-r7,r11,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func25
func25:
        push.w {r4-r7,r11,lr}
        .seh_save_regs_w {r4-r7,r11,lr}
        add.w r11, sp, #16
        .seh_nop_w
        sub sp, sp, #8
        .seh_stackalloc 8
        .seh_endprologue
        nop
        .seh_startepilogue
        pop.w {r2-r7,r11,pc}
        .seh_save_regs_w {r2-r7,r11,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func26
func26:
        push {r1-r3}
        .seh_save_regs {r1-r3}
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #12
        .seh_stackalloc 12
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func27
func27:
        sub sp, sp, #12
        .seh_stackalloc 12
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r1-r3}
        .seh_save_regs {r1-r3}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func28
func28:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func29
func29:
        push {r0-r3}
        .seh_save_regs {r0-r3}
        push {lr}
        .seh_save_regs {lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        ldr pc, [sp], #20
        .seh_save_lr 20
        .seh_endepilogue
        .seh_endproc

        .seh_proc func30
func30:
        push {r4-r6,lr}
        .seh_save_regs {r4-r6,lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4-r6,pc}
        .seh_save_regs {r4-r6,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc func31
func31:
        push {r4-r6,lr}
        .seh_save_regs {r4-r6,lr}
        .seh_endprologue
        nop
        .seh_endproc

        .seh_proc notpacked1
notpacked1:
        push {r1-r3}
        .seh_save_regs {r1-r3}
        .seh_endprologue
        nop
        .seh_startepilogue
        // Mismatch with the folded prologue
        add sp, sp, #8
        .seh_stackalloc 8
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked2
notpacked2:
        sub sp, sp, #8
        .seh_stackalloc 8
        .seh_endprologue
        nop
        .seh_startepilogue
        // Folded epilogue is a mismatch to the regular stack adjust in the prologue
        pop {r1-r3}
        .seh_save_regs {r1-r3}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked3
notpacked3:
        // Can't represent d8-d15 in the packed form
        vpush {d8-d15}
        .seh_save_fregs {d8-d15}
        .seh_endprologue
        nop
        .seh_startepilogue
        vpop {d8-d15}
        .seh_save_fregs {d8-d15}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked4
notpacked4:
        push {r2-r7}
        .seh_save_regs {r2-r7}
        sub sp, sp, #16
        .seh_stackalloc 16
        // Can't have both a folded stack adjustment and a separate one
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #16
        .seh_stackalloc 16
        pop {r2-r7}
        .seh_save_regs {r2-r7}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked5
notpacked5:
        // Can't represent r11 in packed form when it's not contiguous
        // with the rest and when it's not chained (missing "add.w r11, sp, #.."
        // and .seh_nop_w).
        push.w {r4-r7,r11,lr}
        .seh_save_regs_w {r4-r7,r11,lr}
        sub sp, sp, #8
        .seh_stackalloc 8
        .seh_endprologue
        nop
        .seh_startepilogue
        pop.w {r2-r7,r11,pc}
        .seh_save_regs_w {r2-r7,r11,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked6
notpacked6:
        // Can't pack non-contiguous registers
        push {r4,r7}
        .seh_save_regs {r4,r7}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4,r7}
        .seh_save_regs {r4,r7}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked7
notpacked7:
        // Can't pack float registers ouside of d8-d14
        vpush {d0-d3}
        .seh_save_fregs {d0-d3}
        .seh_endprologue
        nop
        .seh_startepilogue
        vpop {d0-d3}
        .seh_save_fregs {d0-d3}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked8
notpacked8:
        push {r4-r7,lr}
        .seh_save_regs {r4-r7,lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4-r7,pc}
        .seh_save_regs {r4-r7,pc}
        .seh_endepilogue
        // Epilogue isn't at the end of the function; can't be packed.
        nop
        .seh_endproc

        .seh_proc notpacked9
notpacked9:
        push {r4-r7,lr}
        .seh_save_regs {r4-r7,lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4-r7,pc}
        .seh_save_regs {r4-r7,pc}
        .seh_endepilogue
        // Multiple epilogues, can't be packed
        nop
        .seh_startepilogue
        pop {r4-r7,pc}
        .seh_save_regs {r4-r7,pc}
        .seh_endepilogue
        .seh_endproc
