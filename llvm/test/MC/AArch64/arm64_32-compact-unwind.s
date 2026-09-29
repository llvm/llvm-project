; RUN: llvm-mc -triple=arm64_32-ios7.0 -filetype=obj %s -o %t
; RUN: llvm-objdump -s %t | FileCheck %s

; The compact unwind format in ILP32 mode is pretty much the same, except
; references to addresses (function, personality, LSDA) are pointer-sized.

; CHECK: Contents of section __LD,__compact_unwind:
; CHECK-NEXT:  0004 00000000 04000000 00000002 00000000
; CHECK-NEXT:  0014 00000000
        .globl  _test_compact_unwind
        .align  2
_test_compact_unwind:
        .cfi_startproc
        ret
        .cfi_endproc
