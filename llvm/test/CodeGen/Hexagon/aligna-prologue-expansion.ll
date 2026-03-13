; RUN: llc -mtriple=hexagon  < %s | FileCheck %s

; CHECK-LABEL: test_aligna_expansion:
; CHECK: allocframe(r29,#{{[0-9]+}}):raw
; CHECK: r[[AP:[0-9]+]] = and(r30,#-128)

declare void @external_func(ptr, ptr, i32, i32)
declare i32 @llvm.smin.i32(i32, i32) #0

define void @test_aligna_expansion(i32 %rows, i32 %depth, ptr %lhs_, ptr %blocking, i32 %param1, i32 %mul) {
entry:
  %buffer = alloca i8, i32 %rows, align 128
  br label %loop_header

loop_header:
  %i = phi i32 [ 0, %entry ], [ %mul, %loop_body ]
  %add = or i32 %i, %param1
  br label %loop_body

loop_body:
  %k = phi i32 [ 0, %loop_header ], [ %add_k, %loop_body ]
  %add_k = add i32 %k, %depth
  %min_val = call i32 @llvm.smin.i32(i32 %add_k, i32 0)
  %sub = sub i32 %min_val, %k
  store i32 %rows, ptr %lhs_, align 4
  call void @external_func(ptr %buffer, ptr %blocking, i32 %sub, i32 %add)
  %cmp = icmp slt i32 %add_k, 0
  br i1 %cmp, label %loop_body, label %loop_header
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
