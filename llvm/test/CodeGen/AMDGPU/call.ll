; RUN: not llc -march=amdgcn -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -march=r600 -mcpu=cypress < %s 2>&1 | FileCheck %s

; CHECK: in function test_call_external{{.*}}: unsupported call to function external_function
; CHECK: in function test_call{{.*}}: unsupported call to function defined_function
; CHECK: in function test_tail_call{{.*}}: unsupported call to function defined_function


declare i32 @external_function(i32) nounwind

define void @test_call_external(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %c = call i32 @external_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define i32 @defined_function(i32 %x) nounwind noinline {
  %y = add i32 %x, 8
  ret i32 %y
}

define void @test_call(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %c = call i32 @defined_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, i32 addrspace(1)* %out
  ret void
}

define void @test_tail_call(i32 addrspace(1)* %out, i32 addrspace(1)* %in) {
  %b_ptr = getelementptr i32, i32 addrspace(1)* %in, i32 1
  %a = load i32, i32 addrspace(1)* %in
  %b = load i32, i32 addrspace(1)* %b_ptr
  %c = tail call i32 @defined_function(i32 %b) nounwind
  %result = add i32 %a, %c
  store i32 %result, i32 addrspace(1)* %out
  ret void
}
