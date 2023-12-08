// REQUIRES: arm-registered-target
// RUN: llvm-mc -filetype=obj -triple thumbv7-windows-gnu %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck --strict-whitespace %s

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
// CHECK-NEXT:    Function: func23
// CHECK-NEXT:    Fragment: No
// CHECK-NEXT:    FunctionLength: 12
// CHECK-NEXT:    ReturnType: (no epilogue)
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
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 0
// CHECK-NEXT:    R: 1
// CHECK-NEXT:    LinkRegister: No
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 4
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      vpush {d8}
// CHECK-NEXT:      push {r3}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      vpop {d8}
// CHECK-NEXT:      pop {r3}
// CHECK-NEXT:      bx <reg>
// CHECK-NEXT:    ]
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func29
// CHECK-NEXT:    Fragment: Yes
// CHECK-NEXT:    FunctionLength: 6
// CHECK-NEXT:    ReturnType: pop {pc}
// CHECK-NEXT:    HomedParameters: No
// CHECK-NEXT:    Reg: 2
// CHECK-NEXT:    R: 0
// CHECK-NEXT:    LinkRegister: Yes
// CHECK-NEXT:    Chaining: No
// CHECK-NEXT:    StackAdjustment: 16
// CHECK-NEXT:    Prologue [
// CHECK-NEXT:      sub sp, sp, #16
// CHECK-NEXT:      push {r4-r6, lr}
// CHECK-NEXT:    ]
// CHECK-NEXT:    Epilogue [
// CHECK-NEXT:      add sp, sp, #16
// CHECK-NEXT:      pop {r4-r6, pc}
// CHECK-NEXT:    ]
// CHECK-NEXT:  }

        .thumb
        .syntax unified

func6:
        nop
        nop
        nop
        bx lr

func7:
        push {r4}
        nop
        pop {r4}
        bx lr

func8:
        push {r4,lr}
        nop
        pop {r4,lr}
        bx lr

func9:
        push {lr}
        vpush {d8}
        sub sp, sp, #32
        nop
        add sp, sp, #32
        vpop {d8}
        pop {lr}
        b tailcall

func10:
        push {r11,lr}
        mov r11, sp
        vpush {d8-d9}
        sub sp, sp, #16
        nop
        add sp, sp, #16
        vpop {d8-d9}
        pop {r11,lr}
        bx lr

func11:
        push {r11,lr}
        mov r11, sp
        vpush {d8-d9}
        sub sp, sp, #16
        nop
        add sp, sp, #16
        vpop {d8-d9}
        pop {r11,pc}

func12:
        vpush {d8-d14}
        sub sp, sp, #16
        nop
        add sp, sp, #16
        vpop {d8-d14}
        b tailcall

func13:
        push {r4-r11,lr}
        add r11, sp, #0x1c
        sub sp, sp, #20
        nop
        add sp, sp, #20
        pop {r4-r11,pc}

func14:
        push {r4-r11,lr}
        sub sp, sp, #20
        nop
        add sp, sp, #20
        pop {r4-r11,pc}

func15:
        push {r0-r3}
        push {r4,lr}
        sub sp, sp, #512
        nop
        add sp, sp, #512
        pop {r4}
        ldr pc, [sp], #20

func16:
        push {r0-r3}
        push {r11,lr}
        mov r11, sp
        nop
        pop {r11, lr}
        add sp, sp, #16
        b tailcall

func17:
        push {r0-r3}
        push {r4}
        sub sp, sp, #512
        nop
        add sp, sp, #512
        pop {r4}
        add sp, sp, #16
        bx lr

func18:
        push {r3,lr}
        nop
        pop {r3,pc}

func19:
        push {r0-r3}
        push {r0-r4}
        nop
        pop {r0-r4}
        add sp, sp, #16
        bx lr

func20:
        push {r0-r3}
        push {r0-r4}
        nop
        add sp, sp, #16
        pop {r4}
        add sp, sp, #16
        bx lr

func21:
        push {r0-r3}
        push {r4}
        sub sp, sp, #16
        nop
        pop {r0-r4}
        add sp, sp, #16
        bx lr

func22:
        nop
        nop
        add sp, sp, #512
        pop {r4}
        ldr pc, [sp], #20

func23:
        push {r0-r3}
        push {r4,lr}
        sub sp, sp, #512
        nop
        nop

func24:
        push {r2-r7,r11,lr}
        add r11, sp, #24
        nop
        add sp, sp, #8
        pop {r4-r7,r11,pc}

func25:
        push {r4-r7,r11,lr}
        add r11, sp, #16
        sub sp, sp, #8
        nop
        pop {r2-r7,r11,pc}

func26:
        push {r1-r3}
        nop
        add sp, sp, #12
        bx lr

func27:
        sub sp, sp, #12
        nop
        pop {r1-r3}
        bx lr

func28:
        push {r3}
        vpush {d8}
        nop
        vpop {d8}
        pop {r3}
        bx lr

func29:
        nop
        pop {r4-r11,pc}

        .section .pdata,"dr"
        .rva func6
        .long 0x000f2011
        .rva func7
        .long 0x00002011
        .rva func8
        .long 0x00102015
        .rva func9
        .long 0x02184031
        .rva func10
        .long 0x01392035
        .rva func11
        .long 0x01390031
        .rva func12
        .long 0x010e4025
        .rva func13
        .long 0x01760025
        .rva func14
        .long 0x0157001d
        .rva func15
        .long 0x20108029
        .rva func16
        .long 0x003fc029
        .rva func17
        .long 0x2000a029
        .rva func18
        .long 0xff1f000d
        .rva func19
        .long 0xffc0a019
        .rva func20
        .long 0xfdc0a01d
        .rva func21
        .long 0xfec0a01d
        .rva func22
        .long 0x2010801e
        .rva func23
        .long 0x2010e019
        .rva func24
        .long 0xfd730021
        .rva func25
        .long 0xfe730021
        .rva func26
        .long 0xfd8f2011
        .rva func27
        .long 0xfe8f2011
        .rva func28
        .long 0xff082011
        .rva func29
        .long 0x0112000e
