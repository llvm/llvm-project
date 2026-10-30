; A vector of pointers from integers (as the SLP vectorizer makes of two
; loads through addresses read from memory) is written as its integers: AIR
; has no vectors of pointers.
; RUN: llc -mtriple=air -filetype=obj -air-output=air %s -o %t.air
; RUN: llvm-dis %t.air -o - | FileCheck %s

; CHECK-LABEL: define void @k(
; CHECK-NOT: x ptr
; CHECK: ret void

define void @k(ptr addrspace(1) %in, ptr addrspace(1) %out) "air-kernel" {
  %v = load <2 x i64>, ptr addrspace(1) %in, align 16
  %p = inttoptr <2 x i64> %v to <2 x ptr addrspace(1)>
  %p0 = extractelement <2 x ptr addrspace(1)> %p, i64 0
  %p1 = extractelement <2 x ptr addrspace(1)> %p, i64 1
  %a = load i32, ptr addrspace(1) %p0, align 4
  %b = load i32, ptr addrspace(1) %p1, align 4
  %s = add i32 %a, %b
  store i32 %s, ptr addrspace(1) %out, align 4
  ret void
}
