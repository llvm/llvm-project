; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t %s
; RUN: llvm-modextract -b -n 0 -o %t0 %t
; RUN: llvm-modextract -b -n 1 -o %t1 %t
; RUN: not llvm-modextract -b -n 2 -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-dis -o - %t0 | FileCheck --check-prefix=M0 %s
; RUN: llvm-dis -o - %t1 | FileCheck --check-prefix=M1 %s
; RUN: llvm-bcanalyzer -dump %t0 | FileCheck --check-prefix=BCA0 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck --check-prefix=BCA1 %s

; ERROR: llvm-modextract: error: module index out of range; bitcode file contains 2 module(s)

; BCA0: <GLOBALVAL_SUMMARY_BLOCK
; BCA1-NOT: <GLOBALVAL_SUMMARY_BLOCK

; M0: @g = external global i8{{$}}
; M1: @g = global i8 42, !type !0, !type !1, !type !2
@g = global i8 42, !type !1, !type !2, !type !4

; M0: define void @f()
; M1-NOT: @f()
define void @f() {
  ; M0: llvm.type.test{{.*}}metadata !"1.f50b51a12bb012bebbeff978335e34cf"
  %p = call i1 @llvm.type.test(i8* null, metadata !0)
  ; M0: llvm.type.checked.load{{.*}}metadata !"2.f50b51a12bb012bebbeff978335e34cf"
  %q = call {i8*, i1} @llvm.type.checked.load(i8* null, i32 0, metadata !3)
  ret void
}

declare i1 @llvm.type.test(i8*, metadata)
declare {i8*, i1} @llvm.type.checked.load(i8*, i32, metadata)

!0 = distinct !{}
; M1: !0 = !{i32 0, !"1.f50b51a12bb012bebbeff978335e34cf"}
!1 = !{i32 0, !0}
; M1: !1 = !{i32 1, !"1.f50b51a12bb012bebbeff978335e34cf"}
!2 = !{i32 1, !0}

!3 = distinct !{}
; M1: !2 = !{i32 0, !"2.f50b51a12bb012bebbeff978335e34cf"}
!4 = !{i32 0, !3}
