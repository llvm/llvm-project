; RUN: opt -loop-unroll -unroll-threshold=2000 -S < %s | llc -march=r600 -mcpu=cypress | FileCheck %s
; REQUIRES: asserts

; CHECK: {{^}}alu_limits:
; CHECK: CF_END

%struct.foo = type {i32, i32, i32}

define amdgpu_kernel void @alu_limits(ptr addrspace(1) %out, ptr addrspace(5) %in, i32 %offset) {
entry:
  %ptr = getelementptr inbounds %struct.foo, ptr addrspace(5) %in, i32 1, i32 2
  %x = load i32, ptr  addrspace(5) %ptr, align 4
  br label %loop
loop:
  %i = phi i32 [ 100, %entry ], [ %nexti, %loop ]
  %val = phi i32 [ 1, %entry ], [ %nextval, %loop ]

  %nexti = sub i32 %i, 1

  %y = xor i32 %x, %i
  %nextval = mul i32 %val, %y

  %cond = icmp ne i32 %nexti, 0
  br i1 %cond, label %loop, label %end
end:
  %out_val = add i32 %nextval, 4
  store i32 %out_val, ptr addrspace(1) %out, align 4
  ret void
}
