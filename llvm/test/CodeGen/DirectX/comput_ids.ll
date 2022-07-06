; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for all ComputeID dxil operations are generated.

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; CHECK-LABEL:test_thread_id
; Function Attrs: noinline nounwind optnone
define i32 @test_thread_id(i32 %a) #0 {
entry:
; CHECK:call i32 @dx.op.threadId.i32(i32 93, i32 %{{.*}})
  %0 = call i32 @llvm.dx.thread.id(i32 %a)
  ret i32 %0
}

; CHECK-LABEL:test_group_id
; Function Attrs: noinline nounwind optnone
define i32 @test_group_id(i32 %a) #0 {
entry:
; CHECK:call i32 @dx.op.groupId.i32(i32 94, i32 %{{.*}})
  %0 = call i32 @llvm.dx.group.id(i32 %a)
  ret i32 %0
}

; CHECK-LABEL:test_thread_id_in_group
; Function Attrs: noinline nounwind optnone
define i32 @test_thread_id_in_group(i32 %a) #0 {
entry:
; CHECK:call i32 @dx.op.threadIdInGroup.i32(i32 95, i32 %{{.*}})
  %0 = call i32 @llvm.dx.thread.id.in.group(i32 %a)
  ret i32 %0
}

; CHECK-LABEL:test_flattened_thread_id_in_group
; Function Attrs: noinline nounwind optnone
define i32 @test_flattened_thread_id_in_group() #0 {
entry:
; CHECK:call i32 @dx.op.flattenedThreadIdInGroup.i32(i32 96)
  %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
  ret i32 %0
}

; Function Attrs: nounwind readnone willreturn
declare i32 @llvm.dx.thread.id(i32) #1
declare i32 @llvm.dx.group.id(i32) #1
declare i32 @llvm.dx.flattened.thread.id.in.group() #1
declare i32 @llvm.dx.thread.id.in.group(i32) #1

attributes #0 = { noinline nounwind }
attributes #1 = { nounwind readnone willreturn }
