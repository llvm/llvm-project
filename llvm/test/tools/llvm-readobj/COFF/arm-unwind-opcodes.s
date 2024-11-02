// REQUIRES: arm-registered-target
// RUN: llvm-mc -filetype=obj -triple thumbv7-windows-gnu %s -o %t.o
// RUN: llvm-readobj --unwind %t.o | FileCheck --strict-whitespace %s

// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: func0
// CHECK:           Prologue [
// CHECK-NEXT:        0xcb                ; mov r11, sp
// CHECK-NEXT:        0x95 0x00           ; push.w {r8, r10, r12}
// CHECK-NEXT:        0xf6 0x13           ; vpush {d17-d19}
// CHECK-NEXT:        0xfc                ; nop.w
// CHECK-NEXT:        0xf5 0x35           ; vpush {d3-d5}
// CHECK-NEXT:        0xfb                ; nop
// CHECK-NEXT:        0xe2                ; vpush {d8-d10}
// CHECK-NEXT:        0x08                ; sub sp, #(8 * 4)
// CHECK-NEXT:        0xd6                ; push {r4-r6, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 15
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 13
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xe2                ; vpop {d8-d10}
// CHECK-NEXT:            0xcb                ; mov sp, r11
// CHECK-NEXT:            0x08                ; add sp, #(8 * 4)
// CHECK-NEXT:            0xd6                ; pop {r4-r6, pc}
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: func1
// CHECK:           Prologue [
// CHECK-NEXT:        0xef 0x08           ; str.w lr, [sp, #-32]!
// CHECK-NEXT:        0xd1                ; push {r4-r5}
// CHECK-NEXT:        0xfd                ; bx <reg>
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 4
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 4
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xef 0x08           ; ldr.w lr, [sp], #32
// CHECK-NEXT:            0xd1                ; pop {r4-r5}
// CHECK-NEXT:            0xfd                ; bx <reg>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: func2
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: No
// CHECK-NEXT:      EpilogueOffset: 0
// CHECK-NEXT:      ByteCodeLength:
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0x04                ; sub sp, #(4 * 4)
// CHECK-NEXT:        0xec 0x80           ; push {r7}
// CHECK-NEXT:        0xc7                ; mov r7, sp
// CHECK-NEXT:        0xfe                ; b.w <target>
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func3
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: Yes
// CHECK-NEXT:      EpilogueOffset: 1
// CHECK-NEXT:      ByteCodeLength:
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0x04                ; sub sp, #(4 * 4)
// CHECK-NEXT:        0xdf                ; push.w {r4-r11, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:      Epilogue [
// CHECK-NEXT:        0xdf                ; pop.w {r4-r11, pc}
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func4
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: No
// CHECK-NEXT:      EpilogueOffset: 0
// CHECK-NEXT:      ByteCodeLength:
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xec 0x50           ; push {r4, r6}
// CHECK-NEXT:        0xb5 0x00           ; push.w {r8, r10, r12, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func5
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData: No
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: No
// CHECK-NEXT:      EpilogueOffset: 16
// CHECK-NEXT:      ByteCodeLength:
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xfa 0x00 0x00 0x20 ; sub.w sp, sp, #(32 * 4)
// CHECK-NEXT:        0xf9 0x00 0x10      ; sub.w sp, sp, #(16 * 4)
// CHECK-NEXT:        0xf8 0x00 0x00 0x08 ; sub sp, sp, #(8 * 4)
// CHECK-NEXT:        0xf7 0x00 0x04      ; sub sp, sp, #(4 * 4)
// CHECK-NEXT:        0xe8 0x02           ; sub.w sp, #(2 * 4)
// CHECK-NEXT:        0xed 0x50           ; push {r4, r6, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:      Epilogue [
// CHECK-NEXT:        0xed 0x50           ; pop {r4, r6, pc}
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }

        .thumb
        .syntax unified
func0:
        push {r4-r6, lr}
        sub sp, sp, #32
        vpush {d8-d10}
        nop
        vpush {d3-d5}
        nop.w
        vpush {d17-d19}
        push {r8, r10, r12}
        mov r11, sp
        nop
        vpop {d8-d10}
        mov sp, r11
        add sp, sp, #32
        pop {r4-r6, pc}

func1:
        push {r4-r5}
        str lr, [sp, #-32]!
        nop
        ldr lr, [sp], #32
        pop {r4-r5}
        bx lr

func2:
        mov r7, sp
        push {r7}
        sub sp, sp, #16
        nop
        add sp, sp, #16
        pop {r7}
        mov sp, r7
        b tailcall

func3:
        nop.w
        nop
        nop
        add sp, sp, #16
        pop {r4-r11, pc}

func4:
        push {r8, r10, r12, lr}
        push {r4, r6}
        nop
        pop {r4, r6}
        pop {r8, r10, r12, pc}

func5:
        push {r4, r6, lr}
        subw sp, sp, #8
        sub sp, sp, #16
        sub sp, sp, #32
        subw sp, sp, #64
        subw sp, sp, #128
        nop
        pop {r4, r6, pc}

        .section .pdata,"dr"
        .rva func0
        .rva .Lunwind_func0
        .rva func1
        .rva .Lunwind_func1
        .rva func2
        .rva .Lunwind_func2
        .rva func3
        .rva .Lunwind_func3
        .rva func4
        .rva .Lunwind_func4
        .rva func5
        .rva .Lunwind_func5

        .section .xdata,"dr"
.Lunwind_func0:
.byte 0x14, 0x00, 0x80, 0x50
.byte 0x0f, 0x00, 0xe0, 0x0d
.byte 0xcb, 0x95, 0x00, 0xf6
.byte 0x13, 0xfc, 0xf5, 0x35
.byte 0xfb, 0xe2, 0x08, 0xd6
.byte 0xff, 0xe2, 0xcb, 0x08
.byte 0xd6, 0xff, 0x00, 0x00

.Lunwind_func1:
.byte 0x08, 0x00, 0x00, 0x00
.byte 0x01, 0x00, 0x02, 0x00
.byte 0x04, 0x00, 0xe0, 0x04
.byte 0xef, 0x08, 0xd1, 0xfd
.byte 0xef, 0x08, 0xd1, 0xfd

.Lunwind_func2:
.byte 0x09, 0x00, 0x20, 0x20
.byte 0x04, 0xec, 0x80, 0xc7
.byte 0xfe, 0x00, 0x00, 0x00
.Lunwind_func3:
.byte 0x07, 0x00, 0xe0, 0x10
.byte 0x04, 0xdf, 0xff, 0x00
.Lunwind_func4:
.byte 0x07, 0x00, 0x20, 0x20
.byte 0xec, 0x50, 0xb5, 0x00
.byte 0xff, 0x00, 0x00, 0x00
.Lunwind_func5:
.byte 0x0b, 0x00, 0x20, 0x58
.byte 0xfa, 0x00, 0x00, 0x20
.byte 0xf9, 0x00, 0x10, 0xf8
.byte 0x00, 0x00, 0x08, 0xf7
.byte 0x00, 0x04, 0xe8, 0x02
.byte 0xed, 0x50, 0xff, 0x00
