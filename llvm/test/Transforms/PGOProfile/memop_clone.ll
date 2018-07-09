; RUN: opt < %s -pgo-memop-opt -verify-dom-info -S | FileCheck %s

define i32 @test(i8* %a, i8* %b) !prof !1 {
; CHECK_LABEL: test
; CHECK: MemOP.Case.3:
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* %a, i32 3, i1 false)
; CHECK: MemOP.Case.2:
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* %a, i32 2, i1 false)
; CHECK: MemOP.Default:
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* %a, i32 undef, i1 false)
; CHECK: MemOP.Case.33:
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* undef, i8* %b, i64 3, i1 false)
; CHECK  MemOP.Case.24:
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* undef, i8* %b, i64 2, i1 false)
; CHECK: MemOP.Default2:
; CHECK: tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* undef, i8* %b, i64 undef, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* %a, i32 undef, i1 false), !prof !2
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* undef, i8* %b, i64 undef, i1 false), !prof !2
  unreachable
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)

!1 = !{!"function_entry_count", i64 5170}
!2 = !{!"VP", i32 1, i64 2585, i64 3, i64 1802, i64 2, i64 783}

