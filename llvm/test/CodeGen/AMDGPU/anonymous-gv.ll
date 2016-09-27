; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji | FileCheck %s

; Make sure we don't crash on a global variable with no name.
@0 = external addrspace(1) global i32

; CHECK-LABEL: {{^}}test:
; CHECK: s_add_u32 s{{[0-9]+}}, s{{[0-9]+}}, __unnamed_1
; CHECK: s_endpgm
define void @test() {
  store i32 1, i32 addrspace(1)* @0
  ret void
}

; CHECK-LABEL: {{^}}__unnamed_2:
; CHECK: s_endpgm
define void @1() {
  ret void
}
