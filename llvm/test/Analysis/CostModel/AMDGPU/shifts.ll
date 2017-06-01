; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=+half-rate-64-ops < %s | FileCheck -check-prefix=ALL -check-prefix=FAST64 %s
; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa -mattr=-half-rate-64-ops < %s | FileCheck -check-prefix=ALL -check-prefix=SLOW64 %s

; ALL: 'shl_i32'
; ALL: estimated cost of 1 for {{.*}} shl i32
define void @shl_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = shl i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL: 'shl_i64'
; FAST64: estimated cost of 2 for {{.*}} shl i64
; SLOW64: estimated cost of 3 for {{.*}} shl i64
define void @shl_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = shl i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL: 'lshr_i32'
; ALL: estimated cost of 1 for {{.*}} lshr i32
define void @lshr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = lshr i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL: 'lshr_i64'
; FAST64: estimated cost of 2 for {{.*}} lshr i64
; SLOW64: estimated cost of 3 for {{.*}} lshr i64
define void @lshr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = lshr i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

; ALL: 'ashr_i32'
; ALL: estimated cost of 1 for {{.*}} ashr i32
define void @ashr_i32(i32 addrspace(1)* %out, i32 addrspace(1)* %vaddr, i32 %b) #0 {
  %vec = load i32, i32 addrspace(1)* %vaddr
  %or = ashr i32 %vec, %b
  store i32 %or, i32 addrspace(1)* %out
  ret void
}

; ALL: 'ashr_i64'
; FAST64: estimated cost of 2 for {{.*}} ashr i64
; SLOW64: estimated cost of 3 for {{.*}} ashr i64
define void @ashr_i64(i64 addrspace(1)* %out, i64 addrspace(1)* %vaddr, i64 %b) #0 {
  %vec = load i64, i64 addrspace(1)* %vaddr
  %or = ashr i64 %vec, %b
  store i64 %or, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
