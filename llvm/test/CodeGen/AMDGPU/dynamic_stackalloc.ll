; RUN: not llc -march=amdgcn -mcpu=tahiti -mattr=+promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -march=amdgcn -mcpu=tahiti -mattr=-promote-alloca -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not llc -march=r600 -mcpu=cypress < %s 2>&1 | FileCheck %s

; CHECK: in function test_dynamic_stackalloc{{.*}}: unsupported dynamic alloca

define void @test_dynamic_stackalloc(i32 addrspace(1)* %out, i32 %n) {
  %alloca = alloca i32, i32 %n
  store volatile i32 0, i32* %alloca
  ret void
}
