; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK: 'or_i32'
; CHECK: estimated cost of 1 for {{.*}} or i32
define void @or_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = or i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'or_i64'
; CHECK: estimated cost of 2 for {{.*}} or i64
define void @or_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = or i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; CHECK: 'xor_i32'
; CHECK: estimated cost of 1 for {{.*}} xor i32
define void @xor_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = xor i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'xor_i64'
; CHECK: estimated cost of 2 for {{.*}} xor i64
define void @xor_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = xor i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}


; CHECK: 'and_i32'
; CHECK: estimated cost of 1 for {{.*}} and i32
define void @and_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = and i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; CHECK: 'and_i64'
; CHECK: estimated cost of 2 for {{.*}} and i64
define void @and_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = and i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}


attributes #0 = { nounwind }
