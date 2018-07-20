; RUN: opt < %s -newgvn -S | FileCheck %s

; CHECK-LABEL: @test1
; CHECK: ret i32 %arg
define i32 @test1(i32 %arg) {
  %cmp = icmp sge i32 %arg, 5
  call void @llvm.assume(i1 %cmp)
  ret i32 %arg
}

; CHECK-LABEL: @test2
; CHECK: ret i32 %arg
define i32 @test2(i32 %arg, i1 %b) {
  br label %bb

bb:
  %a = phi i32 [ 1, %0 ], [ 2, %bb ]
  %cmp = icmp eq i32 %arg, %a
  call void @llvm.assume(i1 %cmp)
  br i1 %b, label %bb, label %end

end:
  ret i32 %arg
}

declare void @llvm.assume(i1 %cond)
