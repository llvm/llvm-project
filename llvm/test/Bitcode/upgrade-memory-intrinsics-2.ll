; RUN: opt -S < %s | FileCheck %s

; Test to ensure that calls to the memcpy/memcpy.inline/memmove
; intrinsics are auto-upgraded to convert just the isVolatile
; argument.  Ensure preservation of attributes, metadata and
; tailcallness.

; Check false->0 & alignment, attributes & bundles are preserved
define void @testfalse(i8* %p1, i8* %p2) {
; CHECK-LABEL: @testfalse
; CHECK: call void @llvm.memset.p0.i64(ptr nonnull align 4 %p1, i8 55, i64 64, i1 false) [ "deopt"() ]
; CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %p1, ptr nonnull align 8 %p2, i64 64, i8 0) [ "deopt"() ]
; CHECK: call void @llvm.memcpy.inline.p0.p0.i64(ptr nonnull align 4 %p1, ptr nonnull align 8 %p2, i64 64, i8 0) [ "deopt"() ]
; CHECK: call void @llvm.memmove.p0.p0.i64(ptr align 4 %p1, ptr align 8 %p2, i64 64, i8 0) [ "deopt"() ]
  call void @llvm.memset.p0i8.i64(i8* nonnull align 4 %p1, i8 55, i64 64, i1 false) [ "deopt"() ]
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %p1, i8* nonnull align 8 %p2, i64 64, i1 false) [ "deopt"() ]
  call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* nonnull align 4 %p1, i8* nonnull align 8 %p2, i64 64, i1 false) [ "deopt"() ]
  call void @llvm.memmove.p0i8.p0i8.i64(i8* align 4 %p1, i8* align 8 %p2, i64 64, i1 false) [ "deopt"() ]
  ret void
}

; Check true->3 &  metadata and tailcall properties are preserved
define void @testtrue(i8* %p1, i8* %p2) {
; CHECK-LABEL: @testtrue
; CHECK: tail call void @llvm.memset.p0.i64(ptr %p1, i8 55, i64 64, i1 true), !tbaa !0
; CHECK: tail call void @llvm.memcpy.p0.p0.i64(ptr %p1, ptr %p2, i64 64, i8 3), !tbaa !0
; CHECK: tail call void @llvm.memcpy.inline.p0.p0.i64(ptr %p1, ptr %p2, i64 64, i8 3), !tbaa !0
; CHECK: tail call void @llvm.memmove.p0.p0.i64(ptr %p1, ptr %p2, i64 64, i8 3), !tbaa !0
  tail call void @llvm.memset.p0i8.i64(i8* %p1, i8 55, i64 64, i1 true), !tbaa !0
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %p1, i8* %p2, i64 64, i1 true), !tbaa !0
  tail call void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* %p1, i8* %p2, i64 64, i1 true), !tbaa !0
  tail call void @llvm.memmove.p0i8.p0i8.i64(i8* %p1, i8* %p2, i64 64, i1 true), !tbaa !0
  ret void
}

; CHECK: declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
; CHECK: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i8 immarg)
; CHECK: declare void @llvm.memcpy.inline.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64 immarg, i8 immarg)
; CHECK: declare void @llvm.memmove.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i8 immarg)
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly , i8* nocapture readonly, i64, i1)
declare void @llvm.memcpy.inline.p0i8.p0i8.i64(i8* nocapture writeonly , i8* nocapture readonly, i64, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i1)

!0 = !{!1, !1, i64 0, i64 1}
!1 = !{!2, i64 1, !"type_0"}
!2 = !{!"root"}
