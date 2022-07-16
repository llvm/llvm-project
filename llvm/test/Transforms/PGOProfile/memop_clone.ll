; RUN: opt < %s -passes=pgo-memop-opt -verify-dom-info -S | FileCheck %s

define i32 @test(ptr %a, ptr %b) !prof !1 {
; CHECK-LABEL: test
; CHECK: MemOP.Case.3:
; CHECK: tail call void @llvm.memcpy.p0.p0.i32(ptr undef, ptr %a, i32 3, i1 false)
; CHECK: MemOP.Case.2:
; CHECK: tail call void @llvm.memcpy.p0.p0.i32(ptr undef, ptr %a, i32 2, i1 false)
; CHECK: MemOP.Default:
; CHECK: tail call void @llvm.memcpy.p0.p0.i32(ptr undef, ptr %a, i32 undef, i1 false)
; CHECK: MemOP.Case.33:
; CHECK: tail call void @llvm.memcpy.p0.p0.i64(ptr undef, ptr %b, i64 3, i1 false)
; CHECK: MemOP.Case.24:
; CHECK: tail call void @llvm.memcpy.p0.p0.i64(ptr undef, ptr %b, i64 2, i1 false)
; CHECK: MemOP.Default2:
; CHECK: tail call void @llvm.memcpy.p0.p0.i64(ptr undef, ptr %b, i64 undef, i1 false)
  tail call void @llvm.memcpy.p0.p0.i32(ptr undef, ptr %a, i32 undef, i1 false), !prof !2
  tail call void @llvm.memcpy.p0.p0.i64(ptr undef, ptr %b, i64 undef, i1 false), !prof !2
  unreachable
}

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1)
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1)

!1 = !{!"function_entry_count", i64 5170}
!2 = !{!"VP", i32 1, i64 2585, i64 3, i64 1802, i64 2, i64 783}

