@ RUN: llvm-mc -triple thumbv6m-apple-macho %s -filetype=obj -o %t
@ RUN: llvm-objdump --macho --section=__DATA,__data %t | FileCheck %s

@ CHECK: 00000000 00000003
        .data
        .align 2
        .global _foo
_foo:
        .long _bar
        .long _baz

        .text
        .thumb_func _bar
        .weak_definition _bar
_bar:
        bx lr

        .thumb_func _baz
        .global _baz
 _baz:
        bx lr
