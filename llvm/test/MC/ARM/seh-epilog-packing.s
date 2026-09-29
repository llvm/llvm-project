// This test checks various cases around sharing opcodes between epilogue and prologue

// RUN: llvm-mc -triple thumbv7-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: func1
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: No
// CHECK-NEXT:      EpilogueOffset: 2
// CHECK-NEXT:      ByteCodeLength:
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xf5 0x15           ; vpush {d1-d5}
// CHECK-NEXT:        0x05                ; sub sp, #(5 * 4)
// CHECK-NEXT:        0xa0 0xf0           ; push.w {r4-r7, lr}
// CHECK-NEXT:        0xfe                ; b.w <target>
// CHECK-NEXT:      ]
// CHECK-NEXT:      Epilogue [
// CHECK-NEXT:        0x05                ; add sp, #(5 * 4)
// CHECK-NEXT:        0xa0 0xf0           ; pop.w {r4-r7, pc}
// CHECK-NEXT:        0xfe                ; b.w <target>
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func2
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: No
// CHECK-NEXT:      EpilogueOffset: 0
// CHECK-NEXT:      ByteCodeLength: 4
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xd2                ; push {r4-r6}
// CHECK-NEXT:        0x04                ; sub sp, #(4 * 4)
// CHECK-NEXT:        0xfd                ; bx <reg>
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: func3
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment: No
// CHECK-NEXT:      EpilogueOffset: 0
// CHECK-NEXT:      ByteCodeLength: 4
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xe1                ; vpush {d8-d9}
// CHECK-NEXT:        0xdf                ; push.w {r4-r11, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  RuntimeFunction {
// CHECK-NEXT:    Function: notshared1
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: Yes
// CHECK-NEXT:      Fragment:
// CHECK-NEXT:      EpilogueOffset: 2
// CHECK-NEXT:      ByteCodeLength: 4
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xdf                ; push.w {r4-r11, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:      Epilogue [
// CHECK-NEXT:        0xdb                ; pop.w {r4-r11}
// CHECK-NEXT:        0xfd                ; bx <reg>
// CHECK-NEXT:      ]
// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: notpacked2
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: No
// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: notpacked3
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: No

        .text
        .syntax unified
        .seh_proc func1
func1:
        push.w {r4-r7,lr}
        .seh_save_regs_w {r4-r7,lr}
        sub sp, sp, #20
        .seh_stackalloc 20
        vpush {d1-d5}
        .seh_save_fregs {d1-d5}
        .seh_endprologue
        nop
        .seh_startepilogue
        add sp, sp, #20
        .seh_stackalloc 20
        // As we're popping into lr instead of directly into pc, this pop
        // becomes a wide instruction. To match prologue vs epilogue, the
        // push in the prologue has been made wide too.
        pop.w {r4-r7,lr}
        .seh_save_regs_w {r4-r7,lr}
        b.w tailcall
        .seh_nop_w
        .seh_endepilogue
        .seh_endproc

        .seh_proc func2
func2:
        sub sp, sp, #16
        .seh_stackalloc 16
        push {r4-r6}
        .seh_save_regs {r4-r6}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4-r6}
        .seh_save_regs {r4-r6}
        add sp, sp, #16
        .seh_stackalloc 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func3
func3:
        push {r4-r11,lr}
        .seh_save_regs_w {r4-r11,lr}
        vpush {d8-d9}
        .seh_save_fregs {d8-d9}
        .seh_endprologue
        nop
        .seh_startepilogue
        vpop {d8-d9}
        .seh_save_fregs {d8-d9}
        pop {r4-r11,pc}
        .seh_save_regs_w {r4-r11,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc notshared1
notshared1:
        push {r4-r11,lr}
        .seh_save_regs_w {r4-r11,lr}
        .seh_endprologue
        nop
        .seh_startepilogue
        // Packed, but not shared as this opcode doesn't match the prolog
        pop {r4-r11}
        .seh_save_regs_w {r4-r11}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc notpacked2
notpacked2:
        push {r4-r11}
        .seh_save_regs_w {r4-r11}
        vpush {d8-d9}
        .seh_save_fregs {d8-d9}
        .seh_endprologue
        nop
        .seh_startepilogue
        vpop {d8-d9}
        .seh_save_fregs {d8-d9}
        pop {r4-r11}
        .seh_save_regs_w {r4-r11}
        bx lr
        .seh_nop
        .seh_endepilogue
        // Not packed, as the epilog isn't at the end of the function
        nop
        .seh_endproc

        .seh_proc notpacked3
notpacked3:
        push {r4-r11,lr}
        .seh_save_regs_w {r4-r11,lr}
        .seh_endprologue
        nop
        it ge
        // Not packed, as the epilog is conditional
        .seh_startepilogue_cond ge
        popge {r4-r11,pc}
        .seh_save_regs_w {r4-r11,pc}
        .seh_endepilogue
        .seh_endproc
