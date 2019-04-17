; RUN: opt < %s -pgo-instr-gen -S | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)
declare i32 @strlen(i8* nocapture)

; CHECK_LABEL: test
; CHECK: %1 = zext i32 %c to i64
; CHECK:  call void @llvm.instrprof.value.profile(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @__profn_test, i32 0, i32 0), i64 12884901887, i64 %1, i32 1, i32 0)

define void @test(i8* %a, i8* %p) {
  %c = call i32 @strlen(i8* %p)
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a, i8* %p, i32 %c, i1 false)
  ret void
}
