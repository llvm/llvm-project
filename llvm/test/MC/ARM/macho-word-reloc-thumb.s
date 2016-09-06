@ RUN: llvm-mc -triple thumbv7-apple-ios %s -filetype=obj -o %t
@ RUN: llvm-objdump -macho -d %t -triple thumbv7-apple-ios | FileCheck %s

@ ARM relocatable object files try to look like they're pre-linked, so the
@ offsets in the instructions are a best-guess. I suspect the "-3" should b

@ CHECK: movw r1, :lower16:((_bar-8)-3)
@ [...]
@ CHECK: .long {{[0-9]*[13579]}}

        .thumb
        .thumb_func _foo
_foo:
        movw r1, :lower16:(_bar-(LPC2_0+4))
        movt r1, :upper16:(_bar-(LPC2_0+4))
LPC2_0:
        add r1, pc
        ldr r0, Lconstpool
        bx lr
Lconstpool:
        .data_region
        .word _bar
        .end_data_region

        .thumb_func _bar
_bar:
        bx lr

        .subsections_via_symbols
