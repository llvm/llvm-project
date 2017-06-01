; RUN: opt -S < %s

; This file contains TBAA metadata that is okay and should pass the verifier.

declare void @callee()
declare void @llvm.va_start(i8*) nounwind

define void @f_0(i8* %ptr, ...) {
  %args = alloca i8, align 8
  call void @llvm.va_start(i8* %args)

  %old = atomicrmw add i8* %ptr, i8 0 seq_cst,          !tbaa !{!1, !1, i64 0}
  %pair = cmpxchg i8* %ptr, i8 0, i8 1 acquire acquire, !tbaa !{!1, !1, i64 0}
  %ld = load i8, i8* %ptr,                              !tbaa !{!1, !1, i64 0}
  store i8 1, i8* %ptr,                                 !tbaa !{!1, !1, i64 0}
  call void @callee(),                                  !tbaa !{!1, !1, i64 0}
  %argval = va_arg i8* %args, i8,                       !tbaa !{!1, !1, i64 0}
  ret void
}

!0 = !{!"root"}
!1 = !{!"scalar-a", !0}
