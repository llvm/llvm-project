; RUN: opt -mtriple=amdgcn-- -mcpu=gfx90a -passes=loop-vectorize %s -S -o - | FileCheck %s

; CHECK-LABEL: @test
; CHECK-LABEL: vector.body:
; CHECK: br i1 %{{[0-9]+}}, label %middle.block, label %vector.body, !llvm.loop !0
; CHECK-LABEL: middle.block:
; CHECK-LABEL: scalar.ph:
; CHECK-LABEL: loop.body:
; CHECK: br i1 %cond, label %exit, label %loop.body, !llvm.loop !2
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"llvm.loop.isvectorized", i32 1}
; CHECK: !2 = distinct !{!2, !3, !1}
; CHECK: !3 = !{!"llvm.loop.unroll.runtime.disable"}


define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(3) %lds, i32 %n) {

entry:
  br label %loop.body

loop.body:
  %counter = phi i32 [0, %entry], [%inc, %loop.body]
  %ptr_lds = getelementptr i32, ptr addrspace(3) %lds, i32 %counter
  %val = load i32, ptr addrspace(3) %ptr_lds
  %ptr_out = getelementptr i32, ptr addrspace(1) %out, i32 %counter
  store i32 %val, ptr addrspace(1) %ptr_out
  %inc = add i32 %counter, 1
  %cond = icmp sge i32 %counter, %n
  br i1 %cond, label  %exit, label %loop.body

exit:
  ret void
}

