; RUN: llc -O0 -march=amdgcn -mcpu=bonaire -mattr=-promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-NO-PROMOTE %s
; RUN: llc -O0 -march=amdgcn -mcpu=bonaire -mattr=+promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-PROMOTE %s
; RUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=-promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-NO-PROMOTE %s
; RUN: llc -O0 -march=amdgcn -mcpu=tonga -mattr=+promote-alloca < %s | FileCheck -check-prefix=CHECK -check-prefix=CHECK-PROMOTE %s

; Disable optimizations in case there are optimizations added that
; specialize away generic pointer accesses.


; These testcases might become useless when there are optimizations to
; remove generic pointers.

; CHECK-LABEL: {{^}}store_flat_i32:
; CHECK: v_mov_b32_e32 v[[DATA:[0-9]+]], {{s[0-9]+}}
; CHECK: v_mov_b32_e32 v[[LO_VREG:[0-9]+]], {{s[0-9]+}}
; CHECK: v_mov_b32_e32 v[[HI_VREG:[0-9]+]], {{s[0-9]+}}
; CHECK: flat_store_dword v[[DATA]], v{{\[}}[[LO_VREG]]:[[HI_VREG]]{{\]}}
define void @store_flat_i32(i32 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i32 addrspace(1)* %gptr to i32 addrspace(4)*
  store i32 %x, i32 addrspace(4)* %fptr, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_flat_i64:
; CHECK: flat_store_dwordx2
define void @store_flat_i64(i64 addrspace(1)* %gptr, i64 %x) #0 {
  %fptr = addrspacecast i64 addrspace(1)* %gptr to i64 addrspace(4)*
  store i64 %x, i64 addrspace(4)* %fptr, align 8
  ret void
}

; CHECK-LABEL: {{^}}store_flat_v4i32:
; CHECK: flat_store_dwordx4
define void @store_flat_v4i32(<4 x i32> addrspace(1)* %gptr, <4 x i32> %x) #0 {
  %fptr = addrspacecast <4 x i32> addrspace(1)* %gptr to <4 x i32> addrspace(4)*
  store <4 x i32> %x, <4 x i32> addrspace(4)* %fptr, align 16
  ret void
}

; CHECK-LABEL: {{^}}store_flat_trunc_i16:
; CHECK: flat_store_short
define void @store_flat_trunc_i16(i16 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16 addrspace(4)*
  %y = trunc i32 %x to i16
  store i16 %y, i16 addrspace(4)* %fptr, align 2
  ret void
}

; CHECK-LABEL: {{^}}store_flat_trunc_i8:
; CHECK: flat_store_byte
define void @store_flat_trunc_i8(i8 addrspace(1)* %gptr, i32 %x) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8 addrspace(4)*
  %y = trunc i32 %x to i8
  store i8 %y, i8 addrspace(4)* %fptr, align 2
  ret void
}



; CHECK-LABEL: load_flat_i32:
; CHECK: flat_load_dword
define void @load_flat_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i32 addrspace(1)* %gptr to i32 addrspace(4)*
  %fload = load i32, i32 addrspace(4)* %fptr, align 4
  store i32 %fload, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: load_flat_i64:
; CHECK: flat_load_dwordx2
define void @load_flat_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i64 addrspace(1)* %gptr to i64 addrspace(4)*
  %fload = load i64, i64 addrspace(4)* %fptr, align 4
  store i64 %fload, i64 addrspace(1)* %out, align 8
  ret void
}

; CHECK-LABEL: load_flat_v4i32:
; CHECK: flat_load_dwordx4
define void @load_flat_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast <4 x i32> addrspace(1)* %gptr to <4 x i32> addrspace(4)*
  %fload = load <4 x i32>, <4 x i32> addrspace(4)* %fptr, align 4
  store <4 x i32> %fload, <4 x i32> addrspace(1)* %out, align 8
  ret void
}

; CHECK-LABEL: sextload_flat_i8:
; CHECK: flat_load_sbyte
define void @sextload_flat_i8(i32 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8 addrspace(4)*
  %fload = load i8, i8 addrspace(4)* %fptr, align 4
  %ext = sext i8 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: zextload_flat_i8:
; CHECK: flat_load_ubyte
define void @zextload_flat_i8(i32 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i8 addrspace(1)* %gptr to i8 addrspace(4)*
  %fload = load i8, i8 addrspace(4)* %fptr, align 4
  %ext = zext i8 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: sextload_flat_i16:
; CHECK: flat_load_sshort
define void @sextload_flat_i16(i32 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16 addrspace(4)*
  %fload = load i16, i16 addrspace(4)* %fptr, align 4
  %ext = sext i16 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

; CHECK-LABEL: zextload_flat_i16:
; CHECK: flat_load_ushort
define void @zextload_flat_i16(i32 addrspace(1)* noalias %out, i16 addrspace(1)* noalias %gptr) #0 {
  %fptr = addrspacecast i16 addrspace(1)* %gptr to i16 addrspace(4)*
  %fload = load i16, i16 addrspace(4)* %fptr, align 4
  %ext = zext i16 %fload to i32
  store i32 %ext, i32 addrspace(1)* %out, align 4
  ret void
}

declare void @llvm.AMDGPU.barrier.local() #1
declare i32 @llvm.r600.read.tidig.x() #3

attributes #0 = { nounwind }
attributes #1 = { nounwind noduplicate }
attributes #3 = { nounwind readnone }
