; Test "llvm.loop.align" validation

; RUN: split-file %s %t

; RUN: not llvm-as < %t/too-few.ll 2>&1 | FileCheck %s --check-prefix=TOO-FEW
; RUN: not llvm-as < %t/too-many.ll 2>&1 | FileCheck %s --check-prefix=TOO-MANY

; RUN: not llvm-as < %t/i16.ll 2>&1 | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not llvm-as < %t/i64.ll 2>&1 | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not llvm-as < %t/mdstring.ll 2>&1 | FileCheck %s --check-prefix=BAD-VALUE
; RUN: not llvm-as < %t/mdnode.ll 2>&1 | FileCheck %s --check-prefix=BAD-VALUE

; RUN: not llvm-as < %t/zero.ll 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
; RUN: not llvm-as < %t/not-pow2.ll 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN
; RUN: not llvm-as < %t/negative.ll 2>&1 | FileCheck %s --check-prefix=BAD-ALIGN

;--- too-few.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align"}
; TOO-FEW: Expected two operands
; TOO-FEW: !{!"llvm.loop.align"}

;--- too-many.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 64, i32 64}
; TOO-MANY: Expected two operands
; TOO-MANY: !{!"llvm.loop.align", i32 64, i32 64}

;--- i16.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i16 16}
; BAD-VALUE: Expected the alignment to be an integer constant of type i32

;--- i64.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i64 64}

;--- mdstring.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", !"64"}

;--- mdnode.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", !2}
!2 = !{i32 64}

;--- zero.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 0}
; BAD-ALIGN: Expected the alignment to be a power of two

;--- not-pow2.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 3}

;--- negative.ll
define void @test() {
  br label %body
body:
  br i1 0, label %body, label %exit, !llvm.loop !0
exit:
  ret void
}
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.align", i32 -8}
