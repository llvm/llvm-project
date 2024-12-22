; RUN: llc -O2 -march=hexagon < %s | FileCheck %s
; CHECK: p{{[0-9]}} = or(p{{[0-9]}},and(p{{[0-9]}},p{{[0-9]}}))

target triple = "hexagon"

define i32 @foo(ptr nocapture %p, ptr nocapture %q) nounwind readonly {
entry:
  %incdec.ptr = getelementptr inbounds i64, ptr %p, i32 1
  %0 = load i64, ptr %p, align 8, !tbaa !0
  %incdec.ptr1 = getelementptr inbounds i64, ptr %q, i32 1
  %1 = load i64, ptr %q, align 8, !tbaa !0
  %2 = tail call i32 @llvm.hexagon.A2.vcmpwgtu(i64 %0, i64 %1)
  %incdec.ptr2 = getelementptr inbounds i64, ptr %p, i32 2
  %3 = load i64, ptr %incdec.ptr, align 8, !tbaa !0
  %incdec.ptr3 = getelementptr inbounds i64, ptr %q, i32 2
  %4 = load i64, ptr %incdec.ptr1, align 8, !tbaa !0
  %5 = tail call i32 @llvm.hexagon.A2.vcmpwgtu(i64 %3, i64 %4)
  %6 = load i64, ptr %incdec.ptr2, align 8, !tbaa !0
  %7 = load i64, ptr %incdec.ptr3, align 8, !tbaa !0
  %8 = tail call i32 @llvm.hexagon.A2.vcmpwgtu(i64 %6, i64 %7)
  %and = and i32 %5, %2
  %or = or i32 %8, %and
  ret i32 %or
}

declare i32 @llvm.hexagon.A2.vcmpwgtu(i64, i64) nounwind readnone

!0 = !{!"long long", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
