; XFAIL: *
; RUN: llc -O0 -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -O0 -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}test_branch(
define void @test_branch(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in, i32 %val) nounwind {
  %cmp = icmp ne i32 %val, 0
  br i1 %cmp, label %store, label %end

store:
  store i32 222, i32 addrspace(1)* %out
  ret void

end:
  ret void
}
