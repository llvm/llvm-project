; RUN: opt -S -mtriple=amdgcn-amd- -amdgpu-annotate-kernel-features %s | FileCheck %s 

; Test to ensure recursive functions exhibit proper behaviour
; Test to generate fibonacci numbers

; CHECK: define i32 @fib(i32 %n) #[[FIB:[0-9]+]] {
define i32 @fib(i32 %n) #0 {
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %exit, label %cont1

cont1:
  %cmp2 = icmp eq i32 %n, 1
  br i1 %cmp2, label %exit, label %cont2

cont2:
  %nm1 = sub i32 %n, 1
  %fibm1 = call i32 @fib(i32 %nm1)
  %nm2 = sub i32 %n, 2
  %fibm2 = call i32 @fib(i32 %nm2)
  %retval = add i32 %fibm1, %fibm2

  ret i32 %retval

exit:
  ret i32 1
}

; CHECK: define amdgpu_kernel void @kernel(i32 addrspace(1)* %m) #[[FIB]] {
define amdgpu_kernel void @kernel(i32 addrspace(1)* %m) #1 {
  %r = call i32 @fib(i32 5)
  store i32 %r, i32 addrspace(1)* %m
  ret void
}

attributes #1 = { "uniform-work-group-size"="true" }

; CHECK: attributes #[[FIB]] = { "uniform-work-group-size"="true" }
