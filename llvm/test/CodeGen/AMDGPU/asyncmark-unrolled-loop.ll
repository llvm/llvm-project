; RUN: opt -O2 -S %s | llc -global-isel=0 -mtriple=amdgpu12.50 -o - | FileCheck %s
; RUN: opt -O2 -S %s | llc -global-isel=1 -mtriple=amdgpu12.50 -o - | FileCheck %s

; CHECK: ; wait_asyncmark(2)
; CHECK: ; wait_asyncmark(1)

define amdgpu_kernel void @unrolled_loop() {
entry:
  br label %loop

loop:
  %i = phi i16 [ 0, %entry ], [ %inc, %loop ]
  call void @llvm.amdgcn.asyncmark()
  %n = sub i16 2, %i
  call void @llvm.amdgcn.wait.asyncmark(i16 %n)
  %inc = add i16 %i, 1
  %cmp = icmp ult i16 %inc, 2
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}

declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)
