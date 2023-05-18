; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs -o - %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: extract_2xi16
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: buffer_load_ushort
; GCN: v_bfe_i32
; GCN: v_bfe_i32

define <2 x i16> @extract_2xi16(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <8 x i16>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <8 x i16>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <8 x i16> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <8 x i16> %m, <8 x i16> undef, <2 x i32> <i32 0, i32 1>
  %b2 = icmp sgt <2 x i16> %v2, <i16 -1, i16 -1>
  %r2 = select <2 x i1> %b2, <2 x i16> <i16 -32768, i16 -32768>, <2 x i16> <i16 -1, i16 -1>
  ret <2 x i16> %r2
}

; GCN-LABEL: extract_2xi64
; GCN-COUNT-2: v_cndmask_b32
define <2 x i64> @extract_2xi64(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <8 x i64>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <8 x i64>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <8 x i64> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <8 x i64> %m, <8 x i64> undef, <2 x i32> <i32 0, i32 1>
  %b2 = icmp sgt <2 x i64> %v2, <i64 -1, i64 -1>
  %r2 = select <2 x i1> %b2, <2 x i64> <i64 -32768, i64 -32768>, <2 x i64> <i64 -1, i64 -1>
  ret <2 x i64> %r2
}

; GCN-LABEL: extract_4xi64
; GCN-COUNT-4: v_cndmask_b32
define <4 x i64> @extract_4xi64(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <8 x i64>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <8 x i64>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <8 x i64> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <8 x i64> %m, <8 x i64> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %b2 = icmp sgt <4 x i64> %v2, <i64 -1, i64 -1, i64 -1, i64 -1>
  %r2 = select <4 x i1> %b2, <4 x i64> <i64 -32768, i64 -32768, i64 -32768, i64 -32768>, <4 x i64> <i64 -1, i64 -1, i64 -1, i64 -1>
  ret <4 x i64> %r2
}

; GCN-LABEL: extract_8xi64
; GCN-COUNT-8: v_cndmask_b32
define <8 x i64> @extract_8xi64(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <16 x i64>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <16 x i64>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <16 x i64> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <16 x i64> %m, <16 x i64> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %b2 = icmp sgt <8 x i64> %v2, <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  %r2 = select <8 x i1> %b2, <8 x i64> <i64 -32768, i64 -32768, i64 -32768, i64 -32768, i64 -32768, i64 -32768, i64 -32768, i64 -32768>, <8 x i64> <i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1, i64 -1>
  ret <8 x i64> %r2
}

; GCN-LABEL: extract_2xf64
; GCN-COUNT-2: v_cndmask_b32
define <2 x double> @extract_2xf64(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <8 x double>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <8 x double>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <8 x double> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <8 x double> %m, <8 x double> undef, <2 x i32> <i32 0, i32 1>
  %b2 = fcmp ogt <2 x double> %v2, <double -1.0, double -1.0>
  %r2 = select <2 x i1> %b2, <2 x double> <double -2.0, double -2.0>, <2 x double> <double -1.0, double -1.0>
  ret <2 x double> %r2
}

; GCN-LABEL: extract_4xf64
; GCN-COUNT-4: v_cndmask_b32
define <4 x double> @extract_4xf64(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <8 x double>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <8 x double>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <8 x double> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <8 x double> %m, <8 x double> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %b2 = fcmp ogt <4 x double> %v2, <double -1.0, double -1.0, double -1.0, double -1.0>
  %r2 = select <4 x i1> %b2, <4 x double> <double -2.0, double -2.0, double -2.0, double -2.0>, <4 x double> <double -1.0, double -1.0, double -1.0, double -1.0>
  ret <4 x double> %r2
}

; GCN-LABEL: extract_8xf64
; GCN-COUNT-8: v_cndmask_b32
define <8 x double> @extract_8xf64(ptr addrspace(1) %p0, ptr addrspace(1) %p1, i1 %c0) {
  br i1 %c0, label %T, label %F

T:
  %t = load volatile <16 x double>, ptr addrspace(1) %p0
  br label %exit

F:
  %f = load volatile <16 x double>, ptr addrspace(1) %p1
  br label %exit

exit:
  %m = phi <16 x double> [ %t, %T ], [ %f, %F ]
  %v2 = shufflevector <16 x double> %m, <16 x double> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %b2 = fcmp ogt <8 x double> %v2, <double -1.0, double -1.0, double -1.0, double -1.0, double -1.0, double -1.0, double -1.0, double -1.0>
  %r2 = select <8 x i1> %b2, <8 x double> <double -2.0, double -2.0, double -2.0, double -2.0, double -2.0, double -2.0, double -2.0, double -2.0>, <8 x double> <double -1.0, double -1.0, double -1.0, double -1.0, double -1.0, double -1.0, double -1.0, double -1.0>
  ret <8 x double> %r2
}
