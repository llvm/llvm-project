; RUN: opt -cost-model -analyze -mtriple=amdgcn-unknown-amdhsa < %s | FileCheck %s

; CHECK: 'insertelement_v2i32'
; CHECK: estimated cost of 0 for {{.*}} insertelement <2 x i32>
define void @insertelement_v2i32(<2 x i32> addrspace(1)* %out, <2 x i32> addrspace(1)* %vaddr) {
  %vec = load <2 x i32>, <2 x i32> addrspace(1)* %vaddr
  %insert = insertelement <2 x i32> %vec, i32 1, i32 123
  store <2 x i32> %insert, <2 x i32> addrspace(1)* %out
  ret void
}

; CHECK: 'insertelement_v2i64'
; CHECK: estimated cost of 0 for {{.*}} insertelement <2 x i64>
define void @insertelement_v2i64(<2 x i64> addrspace(1)* %out, <2 x i64> addrspace(1)* %vaddr) {
  %vec = load <2 x i64>, <2 x i64> addrspace(1)* %vaddr
  %insert = insertelement <2 x i64> %vec, i64 1, i64 123
  store <2 x i64> %insert, <2 x i64> addrspace(1)* %out
  ret void
}

; CHECK: 'insertelement_v2i16'
; CHECK: estimated cost of 0 for {{.*}} insertelement <2 x i16>
define void @insertelement_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %vaddr) {
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %vaddr
  %insert = insertelement <2 x i16> %vec, i16 1, i16 123
  store <2 x i16> %insert, <2 x i16> addrspace(1)* %out
  ret void
}

; CHECK: 'insertelement_v2i8'
; CHECK: estimated cost of 0 for {{.*}} insertelement <2 x i8>
define void @insertelement_v2i8(<2 x i8> addrspace(1)* %out, <2 x i8> addrspace(1)* %vaddr) {
  %vec = load <2 x i8>, <2 x i8> addrspace(1)* %vaddr
  %insert = insertelement <2 x i8> %vec, i8 1, i8 123
  store <2 x i8> %insert, <2 x i8> addrspace(1)* %out
  ret void
}
