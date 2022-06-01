// This test checks that the SEH directives emit the correct unwind data.

// RUN: llvm-mc -triple thumbv7-pc-win32 -filetype=obj %s | llvm-readobj -S -r -u - | FileCheck %s

// Check that the output assembler directives also can be parsed, and
// that they produce equivalent output:

// RUN: llvm-mc -triple thumbv7-pc-win32 -filetype=asm %s | llvm-mc -triple thumbv7-pc-win32 -filetype=obj - | llvm-readobj -S -r -u - | FileCheck %s

// CHECK:      Sections [
// CHECK:        Section {
// CHECK:          Name: .text
// CHECK:          RelocationCount: 1
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_CODE
// CHECK-NEXT:       MEM_16BIT
// CHECK-NEXT:       MEM_EXECUTE
// CHECK-NEXT:       MEM_PURGEABLE
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .xdata
// CHECK:          RawDataSize: 120
// CHECK:          RelocationCount: 1
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK:        Section {
// CHECK:          Name: .pdata
// CHECK:          RelocationCount: 10
// CHECK:          Characteristics [
// CHECK-NEXT:       ALIGN_4BYTES
// CHECK-NEXT:       CNT_INITIALIZED_DATA
// CHECK-NEXT:       MEM_READ
// CHECK-NEXT:     ]
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK-NEXT: Relocations [
// CHECK-NEXT:   Section (1) .text {
// CHECK-NEXT:     0x5C IMAGE_REL_ARM_BRANCH24T tailcall
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (4) .xdata {
// CHECK-NEXT:     0x38 IMAGE_REL_ARM_ADDR32NB __C_specific_handler
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .pdata {
// CHECK-NEXT:     0x0 IMAGE_REL_ARM_ADDR32NB .text
// CHECK-NEXT:     0x4 IMAGE_REL_ARM_ADDR32NB .xdata
// CHECK-NEXT:     0x8 IMAGE_REL_ARM_ADDR32NB .text
// CHECK-NEXT:     0xC IMAGE_REL_ARM_ADDR32NB .xdata
// CHECK-NEXT:     0x10 IMAGE_REL_ARM_ADDR32NB .text
// CHECK-NEXT:     0x14 IMAGE_REL_ARM_ADDR32NB .xdata
// CHECK-NEXT:     0x18 IMAGE_REL_ARM_ADDR32NB .text
// CHECK-NEXT:     0x1C IMAGE_REL_ARM_ADDR32NB .xdata
// CHECK-NEXT:     0x20 IMAGE_REL_ARM_ADDR32NB .text
// CHECK-NEXT:     0x24 IMAGE_REL_ARM_ADDR32NB .xdata
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK-NEXT: UnwindInformation [
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func
// CHECK-NEXT:     ExceptionRecord: .xdata
// CHECK-NEXT:     ExceptionData {
// CHECK-NEXT:       FunctionLength: 86
// CHECK:            Fragment: No
// CHECK:            Prologue [
// CHECK-NEXT:         0xed 0xf8           ; push {r3-r7, lr}
// CHECK-NEXT:         0xf6 0x27           ; vpush {d18-d23}
// CHECK-NEXT:         0xf5 0x7e           ; vpush {d7-d14}
// CHECK-NEXT:         0xfb                ; nop
// CHECK-NEXT:         0xce                ; mov r14, sp
// CHECK-NEXT:         0xe3                ; vpush {d8-d11}
// CHECK-NEXT:         0xe6                ; vpush {d8-d14}
// CHECK-NEXT:         0xed 0xf8           ; push {r3-r7, lr}
// CHECK-NEXT:         0xbd 0x50           ; push.w {r4, r6, r8, r10-r12, lr}
// CHECK-NEXT:         0xd7                ; push {r4-r7, lr}
// CHECK-NEXT:         0xdd                ; push.w {r4-r9, lr}
// CHECK-NEXT:         0xfa 0x01 0x00 0x00 ; sub.w sp, sp, #(65536 * 4)
// CHECK-NEXT:         0xfc                ; nop.w
// CHECK-NEXT:         0xfc                ; nop.w
// CHECK-NEXT:         0xf9 0x04 0x00      ; sub.w sp, sp, #(1024 * 4)
// CHECK-NEXT:         0xe8 0x80           ; sub.w sp, #(128 * 4)
// CHECK-NEXT:         0xe8 0x80           ; sub.w sp, #(128 * 4)
// CHECK-NEXT:         0x06                ; sub sp, #(6 * 4)
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:         EpilogueScope {
// CHECK-NEXT:           StartOffset: 31
// CHECK-NEXT:           Condition: 14
// CHECK-NEXT:           EpilogueStartIndex: 31
// CHECK-NEXT:           Opcodes [
// CHECK-NEXT:             0xfc                ; nop.w
// CHECK-NEXT:             0xf7 0x00 0x80      ; add sp, sp, #(128 * 4)
// CHECK-NEXT:             0xfc                ; nop.w
// CHECK-NEXT:             0xfc                ; nop.w
// CHECK-NEXT:             0xf8 0x01 0x00 0x00 ; add sp, sp, #(65536 * 4)
// CHECK-NEXT:             0x06                ; add sp, #(6 * 4)
// CHECK-NEXT:             0xef 0x04           ; ldr.w lr, [sp], #16
// CHECK-NEXT:             0xfd                ; bx <reg>
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:       ExceptionHandler [
// CHECK-NEXT:         Routine: __C_specific_handler
// CHECK-NEXT:         Parameter: 0x0
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func2
// CHECK:            Prologue [
// CHECK-NEXT:         0xd3                ; push {r4-r7}
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:         EpilogueScope {
// CHECK:                Opcodes [
// CHECK-NEXT:             0xd2                ; pop {r4-r6}
// CHECK-NEXT:             0xfe                ; b.w <target>
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: func3
// CHECK:            FunctionLength: 8
// CHECK:            Prologue [
// CHECK-NEXT:         0xd5                ; push {r4-r5, lr}
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:         EpilogueScope {
// CHECK-NEXT:           StartOffset: 3
// CHECK-NEXT:           Condition: 14
// CHECK-NEXT:           EpilogueStartIndex: 2
// CHECK-NEXT:           Opcodes [
// CHECK-NEXT:             0xd6                ; pop {r4-r6, pc}
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: fragment
// CHECK:            FunctionLength: 6
// CHECK:            Fragment: Yes
// CHECK:            Prologue [
// CHECK-NEXT:         0xcb                ; mov r11, sp
// CHECK-NEXT:         0x10                ; sub sp, #(16 * 4)
// CHECK-NEXT:         0xd5                ; push {r4-r5, lr}
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:         EpilogueScope {
// CHECK-NEXT:           StartOffset: 1
// CHECK-NEXT:           Condition: 14
// CHECK-NEXT:           EpilogueStartIndex: 4
// CHECK-NEXT:           Opcodes [
// CHECK-NEXT:             0x10                ; add sp, #(16 * 4)
// CHECK-NEXT:             0xd5                ; pop {r4-r5, pc}
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   RuntimeFunction {
// CHECK-NEXT:     Function: condepilog
// CHECK:            FunctionLength: 8
// CHECK:            Prologue [
// CHECK-NEXT:         0xd5                ; push {r4-r5, lr}
// CHECK-NEXT:       ]
// CHECK-NEXT:       EpilogueScopes [
// CHECK-NEXT:         EpilogueScope {
// CHECK-NEXT:           StartOffset: 3
// CHECK-NEXT:           Condition: 10
// CHECK-NEXT:           EpilogueStartIndex: 2
// CHECK-NEXT:           Opcodes [
// CHECK-NEXT:             0xd5                ; pop {r4-r5, pc}
// CHECK-NEXT:           ]
// CHECK-NEXT:         }
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

        .text
        .syntax unified
        .globl func
        .def func
        .scl 2
        .type 32
        .endef
        .seh_proc func
func:
        sub sp, sp, #24
        .seh_stackalloc 24
        sub sp, sp, #512
        .seh_stackalloc_w 512
        sub sp, sp, #512
        .seh_stackalloc_w 512
        sub sp, sp, #4096
        .seh_stackalloc_w 4096
        movw r7, #0
        .seh_nop_w
        movt r7, #0x4 // 0x40000
        .seh_nop_w
        sub sp, sp, r7
        .seh_stackalloc_w 0x40000
        push {r4-r8,lr}
        .seh_save_regs_w {r4-r9,lr}
        push {r4-r7,lr}
        .seh_save_regs {r4-r7,lr}
        push {r4,r6,r8,r10,r11,r12,lr}
        .seh_save_regs_w {r4,r6,r8,r10,r11,r12,lr}
        push {r3-r7,lr}
        .seh_save_regs {r3-r7,lr}
        vpush {d8-d14}
        .seh_save_fregs {d8-d14}
        vpush {q4-q5}
        .seh_save_fregs {q4-q5}
        mov lr, sp
        .seh_save_sp lr
        nop
        .seh_nop
        vpush {d7-d14}
        .seh_save_fregs {d7-d14}
        vpush {d18-d23}
        .seh_save_fregs {d18-d23}
        push {r3-r7,lr}
        .seh_custom 0xed, 0xf8
        .seh_endprologue
        nop
        .seh_startepilogue
        mov r7, #512
        .seh_nop_w
        add sp, sp, r7
        .seh_stackalloc 512
        movw r7, #0
        .seh_nop_w
        movt r7, #0x4 // 0x40000
        .seh_nop_w
        add sp, sp, r7
        .seh_stackalloc 0x40000
        add sp, sp, #24
        .seh_stackalloc 24
        ldr lr, [sp], #16
        .seh_save_lr 16
        bx lr
        .seh_nop
        .seh_endepilogue
        .seh_handler __C_specific_handler, %except
        .seh_handlerdata
        .long 0
        .text
        .seh_endproc

        .seh_proc func2
func2:
        push {r4-r7}
        .seh_save_regs {r4-r7}
        .seh_endprologue
        nop
        .seh_startepilogue
        pop {r4-r6}
        .seh_save_regs {r4-r6}
        b.w tailcall
        .seh_nop_w
        .seh_endepilogue
        .seh_endproc

        .seh_proc func3
func3:
        push {r4-r5,lr}
        .seh_save_regs {r4-r5,lr}
        .seh_endprologue
        nop
        // The p2align causes the length of the function to be unknown.
        .p2align 1
        nop
        .seh_startepilogue
        pop {r4-r6,pc}
        .seh_save_regs {r4-r6,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc fragment
fragment:
        // Prologue opcodes without matching instructions
        .seh_save_regs {r4-r5,lr}
        .seh_stackalloc 64
        .seh_save_sp r11
        .seh_endprologue_fragment
        nop
        .seh_startepilogue
        add sp, sp, #64
        .seh_stackalloc 64
        pop {r4-r5,pc}
        .seh_save_regs {r4-r5,pc}
        .seh_endepilogue
        .seh_endproc

        .seh_proc condepilog
condepilog:
        push {r4-r5,lr}
        .seh_save_regs {r4-r5,lr}
        .seh_endprologue
        nop
        it ge
        .seh_startepilogue_cond ge
        popge {r4-r5,pc}
        .seh_save_regs {r4-r5,pc}
        .seh_endepilogue
        .seh_endproc

        // Function with no .seh directives; no pdata/xdata entries are
        // generated.
        .globl smallFunc
        .def smallFunc
        .scl 2
        .type 32
        .endef
        .seh_proc smallFunc
smallFunc:
        bx lr
        .seh_endproc

        // Function with no .seh directives, but with .seh_handlerdata.
        // No xdata/pdata entries are generated, but the custom handler data
        // (the .long after .seh_handlerdata) is left orphaned in the xdata
        // section.
        .globl handlerFunc
        .def handlerFunc
        .scl 2
        .type 32
        .endef
        .seh_proc handlerFunc
handlerFunc:
        bx lr
        .seh_handler __C_specific_handler, %except
        .seh_handlerdata
        .long 0
        .text
        .seh_endproc
