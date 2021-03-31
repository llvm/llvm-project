; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK: 'test_br_cost'
; CHECK: estimated cost of 10 for instruction: br i1
; CHECK: estimated cost of 10 for instruction: br label
; CHECK: estimated cost of 10 for instruction: ret void
define amdgpu_kernel void @test_br_cost(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
bb0:
  br i1 undef, label %bb1, label %bb2

bb1:
  %vec = load i32, i32 addrspace(1)* %vaddr
  %add = add i32 %vec, %b
  store i32 %add, i32 addrspace(1)* %out
  br label %bb2

bb2:
  ret void

}

; CHECK: 'test_switch_cost'
; CHECK: estimated cost of -1 for instruction:   switch
define amdgpu_kernel void @test_switch_cost(i32 %a) #0 {
entry:
  switch i32 %a, label %default [
    i32 0, label %case0
    i32 1, label %case1
  ]

case0:
  store volatile i32 undef, i32 addrspace(1)* undef
  ret void

case1:
  store volatile i32 undef, i32 addrspace(1)* undef
  ret void

default:
  store volatile i32 undef, i32 addrspace(1)* undef
  ret void

end:
  ret void
}
