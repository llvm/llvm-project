; RUN: llc -O2 < %s | FileCheck %s --check-prefix=CHECK-O2 --check-prefix=CHECK
; RUN: llc -O0 -fast-isel < %s | FileCheck %s --check-prefix=CHECK-O0 --check-prefix=CHECK
; RUN: llc -O0 -fast-isel=0 < %s | FileCheck %s --check-prefix=CHECK-O0 --check-prefix=CHECK
; RUN: llc -O0 -global-isel < %s | FileCheck %s --check-prefix=CHECK-O0 --check-prefix=CHECK

;; Ensure that an unfoldable is.constant gets lowered reasonably in
;; optimized codegen, in particular, that the "true" branch is
;; eliminated.
;;
;; This isn't asserting any specific output from non-optimized runs,
;; (e.g., currently the not-taken branch does not get eliminated). But
;; it does ensure that lowering succeeds in all 3 codegen paths.

target triple = "x86_64-unknown-linux-gnu"

declare i1 @llvm.is.constant.i32(i32 %a) nounwind readnone
declare i1 @llvm.is.constant.i64(i64 %a) nounwind readnone
declare i64 @llvm.objectsize.i64.p0i8(i8*, i1, i1) nounwind readnone

declare i32 @subfun_1()
declare i32 @subfun_2()

define i32 @test_branch(i32 %in) nounwind {
; CHECK-LABEL:    test_branch:
; CHECK-O2:       %bb.0:
; CHECK-O2-NEXT:  jmp subfun_2
  %v = call i1 @llvm.is.constant.i32(i32 %in)
  br i1 %v, label %True, label %False

True:
  %call1 = tail call i32 @subfun_1()
  ret i32 %call1

False:
  %call2 = tail call i32 @subfun_2()
  ret i32 %call2
}

;; llvm.objectsize is another tricky case which gets folded to -1 very
;; late in the game. We'd like to ensure that llvm.is.constant of
;; llvm.objectsize is true.
define i1 @test_objectsize(i8* %obj) nounwind {
; CHECK-LABEL:    test_objectsize:
; CHECK-O2:       %bb.0:
; CHECK-O2:       movb $1, %al
; CHECK-O2-NEXT:  retq
  %os = call i64 @llvm.objectsize.i64.p0i8(i8* %obj, i1 false, i1 false)
  %v = call i1 @llvm.is.constant.i64(i64 %os)
  ret i1 %v
}
