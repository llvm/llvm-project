; RUN: split-file %s %t
; RUN: not llvm-as < %t/not-integer.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NOTINTEGER
; RUN: not llvm-as < %t/zero.ll -disable-output 2>&1 | FileCheck %s --check-prefix=ZERO
; RUN: not llvm-as < %t/negative.ll -disable-output 2>&1 | FileCheck %s --check-prefix=NEGATIVE
; RUN: llvm-as < %t/valid.ll -disable-output

;--- not-integer.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", !"4"}
; NOTINTEGER: wchar_size metadata requires constant positive integer argument

;--- zero.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 0}
; ZERO: wchar_size metadata requires constant positive integer argument

;--- negative.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 -1}
; NEGATIVE: wchar_size metadata requires constant positive integer argument

;--- valid.ll
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}
