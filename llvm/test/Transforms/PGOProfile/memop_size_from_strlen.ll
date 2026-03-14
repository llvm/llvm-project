; RUN: opt < %s -passes=pgo-instr-gen -S | FileCheck %s

declare void @llvm.memcpy.p0.p0.i32(ptr nocapture writeonly, ptr nocapture readonly, i32, i1)
declare i32 @strlen(ptr nocapture)

; CHECK-LABEL: test
; CHECK: %1 = zext i32 %c to i64
; CHECK:  call void @llvm.instrprof.value.profile(ptr @__profn_test, i64 {{[0-9]+}}, i64 %1, i32 1, i32 0)

define void @test(ptr %a, ptr %p) {
  %c = call i32 @strlen(ptr %p)
  call void @llvm.memcpy.p0.p0.i32(ptr %a, ptr %p, i32 %c, i1 false)
  ret void
}
