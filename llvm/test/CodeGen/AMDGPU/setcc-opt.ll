; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}sext_bool_icmp_eq_0:
; GCN-NOT: v_cmp
; GCN: v_cmp_ne_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT:buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm

; EG: SETNE_INT * [[CMP:T[0-9]+]].[[CMPCHAN:[XYZW]]], KC0[2].Z, KC0[2].W
; EG: AND_INT T{{[0-9]+.[XYZW]}}, PS, 1
define void @sext_bool_icmp_eq_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_ne_0:
; GCN-NOT: v_cmp
; GCN: v_cmp_ne_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm

; EG: SETNE_INT * [[CMP:T[0-9]+]].[[CMPCHAN:[XYZW]]], KC0[2].Z, KC0[2].W
; EG: AND_INT T{{[0-9]+.[XYZW]}}, PS, 1
define void @sext_bool_icmp_ne_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; This really folds away to false
; FUNC-LABEL: {{^}}sext_bool_icmp_eq_1:
; GCN: v_cmp_eq_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[TMP:v[0-9]+]], 0, -1, vcc
; GCN-NEXT: v_cmp_eq_i32_e32 vcc, 1, [[TMP]]{{$}}
; GCN-NEXT: v_cndmask_b32_e64 [[TMP:v[0-9]+]], 0, 1,
; GCN-NEXT: buffer_store_byte [[TMP]]
; GCN-NEXT: s_endpgm
define void @sext_bool_icmp_eq_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; This really folds away to true
; FUNC-LABEL: {{^}}sext_bool_icmp_ne_1:
; GCN: v_cmp_ne_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[TMP:v[0-9]+]], 0, -1, vcc
; GCN-NEXT: v_cmp_ne_i32_e32 vcc, 1, [[TMP]]{{$}}
; GCN-NEXT: v_cndmask_b32_e64 [[TMP:v[0-9]+]], 0, 1,
; GCN-NEXT: buffer_store_byte [[TMP]]
; GCN-NEXT: s_endpgm
define void @sext_bool_icmp_ne_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_eq_0:
; GCN-NOT: v_cmp
; GCN: v_cmp_ne_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define void @zext_bool_icmp_eq_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_ne_0:
; GCN-NOT: v_cmp
; GCN: v_cmp_ne_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define void @zext_bool_icmp_ne_0(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 0
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_eq_1:
; GCN-NOT: v_cmp
; GCN: v_cmp_eq_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define void @zext_bool_icmp_eq_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp eq i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_ne_1:
; GCN-NOT: v_cmp
; GCN: v_cmp_eq_i32_e32 vcc,
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT: buffer_store_byte [[RESULT]]
define void @zext_bool_icmp_ne_1(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 1
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}sext_bool_icmp_ne_k:
; SI-DAG: s_load_dword [[A:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[B:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; VI-DAG: s_load_dword [[A:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; VI-DAG: s_load_dword [[B:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x30
; GCN: v_mov_b32_e32 [[VB:v[0-9]+]], [[B]]
; GCN: v_cmp_ne_i32_e32 vcc, 2, [[VB]]{{$}}
; GCN: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN: buffer_store_byte
; GCN: s_endpgm
define void @sext_bool_icmp_ne_k(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = sext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 2
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_zext_k_i8max:
; GCN: buffer_load_ubyte [[B:v[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0 offset:44
; GCN: v_mov_b32_e32 [[K255:v[0-9]+]], 0xff{{$}}
; GCN: v_cmp_ne_i32_e32 vcc, [[K255]], [[B]]
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define void @cmp_zext_k_i8max(i1 addrspace(1)* %out, i8 %b) nounwind {
  %b.ext = zext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, 255
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_sext_k_neg1:
; GCN: buffer_load_sbyte [[B:v[0-9]+]]
; GCN: v_cmp_ne_i32_e32 vcc, -1, [[B]]{{$}}
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define void @cmp_sext_k_neg1(i1 addrspace(1)* %out, i8 addrspace(1)* %b.ptr) nounwind {
  %b = load i8, i8 addrspace(1)* %b.ptr
  %b.ext = sext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_sext_k_neg1_i8_sext_arg:
; GCN: s_load_dword [[B:s[0-9]+]]
; GCN: v_cmp_ne_i32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -1, [[B]]
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, [[CMP]]
; GCN-NEXT: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define void @cmp_sext_k_neg1_i8_sext_arg(i1 addrspace(1)* %out, i8 signext %b) nounwind {
  %b.ext = sext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FIXME: This ends up doing a buffer_load_ubyte, and and compare to
; 255. Seems to be because of ordering problems when not allowing load widths to be reduced.
; Should do a buffer_load_sbyte and compare with -1

; FUNC-LABEL: {{^}}cmp_sext_k_neg1_i8_arg:
; GCN-DAG: buffer_load_ubyte [[B:v[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0xff{{$}}
; GCN: v_cmp_ne_i32_e32 vcc, [[K]], [[B]]{{$}}
; GCN-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, 1, vcc
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define void @cmp_sext_k_neg1_i8_arg(i1 addrspace(1)* %out, i8 %b) nounwind {
  %b.ext = sext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}cmp_zext_k_neg1:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[RESULT]]
; GCN: s_endpgm
define void @cmp_zext_k_neg1(i1 addrspace(1)* %out, i8 %b) nounwind {
  %b.ext = zext i8 %b to i32
  %icmp0 = icmp ne i32 %b.ext, -1
  store i1 %icmp0, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_ne_k:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 1{{$}}
; GCN: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define void @zext_bool_icmp_ne_k(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp ne i32 %ext, 2
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}zext_bool_icmp_eq_k:
; GCN: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; GCN: buffer_store_byte [[RESULT]]
; GCN-NEXT: s_endpgm
define void @zext_bool_icmp_eq_k(i1 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = zext i1 %icmp0 to i32
  %icmp1 = icmp eq i32 %ext, 2
  store i1 %icmp1, i1 addrspace(1)* %out
  ret void
}
