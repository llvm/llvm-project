// This test checks various cases around sharing opcodes between epilogue and prologue with more than one epilogue.

// RUN: llvm-mc -triple thumbv7-pc-win32 -filetype=obj %s | llvm-readobj -u - | FileCheck %s

// CHECK:       RuntimeFunction {
// CHECK-NEXT:    Function: func1
// CHECK-NEXT:    ExceptionRecord:
// CHECK-NEXT:    ExceptionData {
// CHECK-NEXT:      FunctionLength:
// CHECK-NEXT:      Version:
// CHECK-NEXT:      ExceptionData:
// CHECK-NEXT:      EpiloguePacked: No
// CHECK-NEXT:      Fragment:
// CHECK-NEXT:      EpilogueScopes: 3
// CHECK-NEXT:      ByteCodeLength: 12
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xf5 0x15           ; vpush {d1-d5}
// CHECK-NEXT:        0x05                ; sub sp, #(5 * 4)
// CHECK-NEXT:        0xa0 0xf0           ; push.w {r4-r7, lr}
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 6
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 6
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0x08                ; add sp, #(8 * 4)
// CHECK-NEXT:            0xfd                ; bx <reg>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 9
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 8
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0x10                ; add sp, #(16 * 4)
// CHECK-NEXT:            0xfd                ; bx <reg>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 13
// CHECK-NEXT:          Condition: 10
// CHECK-NEXT:          EpilogueStartIndex: 6
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0x08                ; add sp, #(8 * 4)
// CHECK-NEXT:            0xfd                ; bx <reg>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
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
// CHECK-NEXT:      EpiloguePacked: No
// CHECK-NEXT:      Fragment:
// CHECK-NEXT:      EpilogueScopes: 3
// CHECK-NEXT:      ByteCodeLength: 12
// CHECK-NEXT:      Prologue [
// CHECK-NEXT:        0xf5 0x15           ; vpush {d1-d5}
// CHECK-NEXT:        0x05                ; sub sp, #(5 * 4)
// CHECK-NEXT:        0xa0 0xf0           ; push.w {r4-r7, lr}
// CHECK-NEXT:        0xfe                ; b.w <target>
// CHECK-NEXT:      ]
// CHECK-NEXT:      EpilogueScopes [
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 6
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 2
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0x05                ; add sp, #(5 * 4)
// CHECK-NEXT:            0xa0 0xf0           ; pop.w {r4-r7, pc}
// CHECK-NEXT:            0xfe                ; b.w <target>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 11
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 3
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xa0 0xf0           ; pop.w {r4-r7, pc}
// CHECK-NEXT:            0xfe                ; b.w <target>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:        EpilogueScope {
// CHECK-NEXT:          StartOffset: 15
// CHECK-NEXT:          Condition: 14
// CHECK-NEXT:          EpilogueStartIndex: 6
// CHECK-NEXT:          Opcodes [
// CHECK-NEXT:            0xa0 0xf0           ; pop.w {r4-r7, pc}
// CHECK-NEXT:            0xfd                ; bx <reg>
// CHECK-NEXT:          ]
// CHECK-NEXT:        }
// CHECK-NEXT:      ]
// CHECK-NEXT:    }
// CHECK-NEXT:  }

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

        // Entirely different epilogue; can't be shared with the prologue.
        .seh_startepilogue
        add sp, sp, #32
        .seh_stackalloc 32
        bx lr
        .seh_nop
        .seh_endepilogue

        nop

        // Also a differing epilogue.
        .seh_startepilogue
        add sp, sp, #64
        .seh_stackalloc 64
        bx lr
        .seh_nop
        .seh_endepilogue

        nop

        // Epilogue matches the first one; will reuse that epilogue's opcodes,
        // even if they differ in conditionality.
        itt ge
        .seh_startepilogue_cond ge
        addge sp, sp, #32
        .seh_stackalloc 32
        bxge lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc

        .seh_proc func2
func2:
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
        // Ending with a different end opcode, but can still be shared with
        // the prolog.
        .seh_nop_w
        .seh_endepilogue

        // Another epilogue, matching the end of the previous epilogue.
        .seh_startepilogue
        pop.w {r4-r7,lr}
        .seh_save_regs_w {r4-r7,lr}
        b.w tailcall
        .seh_nop_w
        .seh_endepilogue

        // This epilogue differs in the end opcode, and can't be shared with
        // the prologue.
        .seh_startepilogue
        pop.w {r4-r7,lr}
        .seh_save_regs_w {r4-r7,lr}
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_endproc
