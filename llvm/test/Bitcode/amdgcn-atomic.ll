; RUN: llvm-as < %s | llvm-dis | FileCheck %s


define void @atomic_inc(ptr %ptr0, ptr addrspace(1) %ptr1, ptr addrspace(3) %ptr3) {
  ; CHECK: atomicrmw uinc_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result0 = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %ptr0, i32 42, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !1
  %result1 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(3) %ptr3, i32 46 syncscope("agent") seq_cst, align 4{{$}}
  %result2 = call i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) %ptr3, i32 46, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr %ptr0, i64 48 syncscope("agent") seq_cst, align 8
  %result3 = call i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr %ptr0, i64 48, i32 0, i32 0, i1 false)

 ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i64 45 syncscope("agent") seq_cst, align 8
  %result4 = call i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) %ptr1, i64 45, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result5 = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw volatile uinc_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result6 = call i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 true)
  ret void
}

define void @atomic_dec(ptr %ptr0, ptr addrspace(1) %ptr1, ptr addrspace(3) %ptr3) {
  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result0 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !1
  %result1 = call i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(3) %ptr3, i32 46 syncscope("agent") seq_cst, align 4{{$}}
  %result2 = call i32 @llvm.amdgcn.atomic.dec.i32.p3(ptr addrspace(3) %ptr3, i32 46, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i64 48 syncscope("agent") seq_cst, align 8
  %result3 = call i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr %ptr0, i64 48, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(1) %ptr1, i64 45 syncscope("agent") seq_cst, align 8
  %result4 = call i64 @llvm.amdgcn.atomic.dec.i64.p1(ptr addrspace(1) %ptr1, i64 45, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw udec_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result5 = call i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr addrspace(3) %ptr3, i64 4345 syncscope("agent") seq_cst, align 8
  %result6 = call i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) %ptr3, i64 4345, i32 0, i32 0, i1 true)
  ret void
}

; Test some invalid ordering handling
define void @ordering(ptr %ptr0, ptr addrspace(1) %ptr1, ptr addrspace(3) %ptr3) {
  ; CHECK: atomicrmw volatile uinc_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result0 = call i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr %ptr0, i32 42, i32 -1, i32 0, i1 true)

  ; CHECK: atomicrmw volatile uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !1
  %result1 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 0, i32 0, i1 true)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !1
  %result2 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 1, i32 0, i1 false)

  ; CHECK: atomicrmw volatile uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") monotonic, align 4, !amdgpu.no.fine.grained.memory !1
  %result3 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 2, i32 0, i1 true)

  ; CHECK: atomicrmw uinc_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !1
  %result4 = call i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 3, i32 0, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result5 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 4, i1 true)

  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result6 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 5, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result7 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 6, i1 true)

  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result8 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 7, i1 false)

  ; CHECK:= atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result9 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 0, i32 8, i1 true)

  ; CHECK:= atomicrmw volatile udec_wrap ptr addrspace(1) %ptr1, i32 43 syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !1
  %result10 = call i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) %ptr1, i32 43, i32 3, i32 0, i1 true)
  ret void
}

define void @immarg_violations(ptr %ptr0, i32 %val32, i1 %val1) {
  ; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result0 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 %val32, i32 0, i1 false)

; CHECK: atomicrmw udec_wrap ptr %ptr0, i32 42 syncscope("agent") monotonic, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result1 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 2, i32 %val32, i1 false)

  ; CHECK: atomicrmw volatile udec_wrap ptr %ptr0, i32 42 syncscope("agent") monotonic, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !1{{$}}
  %result2 = call i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr %ptr0, i32 42, i32 2, i32 0, i1 %val1)
  ret void
}

declare i32 @llvm.amdgcn.atomic.inc.i32.p1(ptr addrspace(1) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.inc.i32.p3(ptr addrspace(3) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.inc.i32.p0(ptr nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.inc.i64.p1(ptr addrspace(1) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.inc.i64.p3(ptr addrspace(3) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.inc.i64.p0(ptr nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0

declare i32 @llvm.amdgcn.atomic.dec.i32.p1(ptr addrspace(1) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.dec.i32.p3(ptr addrspace(3) nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i32 @llvm.amdgcn.atomic.dec.i32.p0(ptr nocapture, i32, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.dec.i64.p1(ptr addrspace(1) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.dec.i64.p3(ptr addrspace(3) nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0
declare i64 @llvm.amdgcn.atomic.dec.i64.p0(ptr nocapture, i64, i32 immarg, i32 immarg, i1 immarg) #0

; ptr, rmw_value, ordering, scope, isVolatile)
declare float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) nocapture, float, i32 immarg, i32 immarg, i1 immarg)
declare double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) nocapture, double, i32 immarg, i32 immarg, i1 immarg)
declare <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) nocapture, <2 x half>, i32 immarg, i32 immarg, i1 immarg)
declare <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) nocapture, <2 x i16>)

define float @upgrade_amdgcn_ds_fadd_f32(ptr addrspace(3) %ptr, float %val) {
  ; CHECK: atomicrmw fadd ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result0 = call float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 false)

  ; CHECK: = atomicrmw volatile fadd ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result1 = call float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 true)

  ; CHECK: = atomicrmw fadd ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result2 = call float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %ptr, float %val, i32 43, i32 3, i1 false)

  ; CHECK: = atomicrmw fadd ptr addrspace(3) %ptr, float %val syncscope("agent") acquire, align 4
  %result3 = call float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %ptr, float %val, i32 4, i32 2, i1 false)

  ret float %result3
}

; Handle missing type suffix
declare float @llvm.amdgcn.ds.fadd(ptr addrspace(3) nocapture, float, i32 immarg, i32 immarg, i1 immarg)

define float @upgrade_amdgcn_ds_fadd_f32_no_suffix(ptr addrspace(3) %ptr, float %val) {
  ; CHECK: atomicrmw fadd ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result0 = call float @llvm.amdgcn.ds.fadd(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 false)
  ret float %result0
}

define void @immarg_violations_ds_fadd_f32(ptr addrspace(3) %ptr, float %fval, i32 %val32, i1 %val1) {
  ; CHECK: = atomicrmw volatile fadd ptr addrspace(3) %ptr, float %fval syncscope("agent") seq_cst, align 4
  %result0 = call float @llvm.amdgcn.ds.fadd.f32(ptr addrspace(3) %ptr, float %fval, i32 %val32, i32 %val32, i1 %val1)
  ret void
}

declare float @llvm.amdgcn.ds.fadd.f32broken0(i32, float, i32 immarg, i32 immarg, i1 immarg)

; This will just delete the invalid call, which isn't ideal, but these
; cases were never emitted.
; CHECK-LABEL: define void @ds_fadd_f32_invalid_not_ptr(
; CHECK-NEXT: ret void
define void @ds_fadd_f32_invalid_not_ptr(i32 %ptr, float %fval) {
  %result0 = call float @llvm.amdgcn.ds.fadd.f32broken0(i32 %ptr, float %fval, i32 0, i32 0, i1 false)
  ret void
}

declare float @llvm.amdgcn.ds.fadd.f32broken1(ptr addrspace(3), double, i32 immarg, i32 immarg, i1 immarg)

; CHECK-LABEL: define void @ds_fadd_f32_invalid_misatch(
; CHECK-NEXT: ret void
define void @ds_fadd_f32_invalid_misatch(ptr addrspace(3) %ptr, double %fval) {
  %result0 = call float @llvm.amdgcn.ds.fadd.f32broken1(ptr addrspace(3) %ptr, double %fval, i32 0, i32 0, i1 false)
  ret void
}

define double @upgrade_amdgcn_ds_fadd_f64(ptr addrspace(3) %ptr, double %val) {
  ; CHECK: atomicrmw fadd ptr addrspace(3) %ptr, double %val syncscope("agent") seq_cst, align 8
  %result0 = call double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) %ptr, double %val, i32 0, i32 0, i1 false)

  ; CHECK: = atomicrmw volatile fadd ptr addrspace(3) %ptr, double %val syncscope("agent") seq_cst, align 8
  %result1 = call double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) %ptr, double %val, i32 0, i32 0, i1 true)

  ; CHECK: = atomicrmw fadd ptr addrspace(3) %ptr, double %val syncscope("agent") seq_cst, align 8
  %result2 = call double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) %ptr, double %val, i32 43, i32 3, i1 false)

  ; CHECK: = atomicrmw fadd ptr addrspace(3) %ptr, double %val syncscope("agent") acquire, align 8
  %result3 = call double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) %ptr, double %val, i32 4, i32 2, i1 false)

  ret double %result3
}

; CHECK-LABEL: @immarg_violations_ds_fadd_f64(
define void @immarg_violations_ds_fadd_f64(ptr addrspace(3) %ptr, double %fval, i32 %val32, i1 %val1) {
  ; CHECK: = atomicrmw volatile fadd ptr addrspace(3) %ptr, double %fval syncscope("agent") seq_cst, align 8
  %result0 = call double @llvm.amdgcn.ds.fadd.f64(ptr addrspace(3) %ptr, double %fval, i32 %val32, i32 %val32, i1 %val1)
  ret void
}

define <2 x half> @upgrade_amdgcn_ds_fadd_v2f16(ptr addrspace(3) %ptr, <2 x half> %val) {
  ; CHECK: atomicrmw fadd ptr addrspace(3) %ptr, <2 x half> %val syncscope("agent") seq_cst, align 4
  %result0 = call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %ptr, <2 x half> %val, i32 0, i32 0, i1 false)

  ; CHECK: = atomicrmw volatile fadd ptr addrspace(3) %ptr, <2 x half> %val syncscope("agent") seq_cst, align 4
  %result1 = call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %ptr, <2 x half> %val, i32 0, i32 0, i1 true)

  ; CHECK: = atomicrmw fadd ptr addrspace(3) %ptr, <2 x half> %val syncscope("agent") seq_cst, align 4
  %result2 = call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %ptr, <2 x half> %val, i32 43, i32 3, i1 false)

  ; CHECK: = atomicrmw fadd ptr addrspace(3) %ptr, <2 x half> %val syncscope("agent") acquire, align 4
  %result3 = call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %ptr, <2 x half> %val, i32 4, i32 2, i1 false)

  ret <2 x half> %result3
}

define void @immarg_violations_ds_fadd_v2f16(ptr addrspace(3) %ptr, <2 x half> %fval, i32 %val32, i1 %val1) {
  ; CHECK: = atomicrmw volatile fadd ptr addrspace(3) %ptr, <2 x half> %fval syncscope("agent") seq_cst, align 4
  %result0 = call <2 x half> @llvm.amdgcn.ds.fadd.v2f16(ptr addrspace(3) %ptr, <2 x half> %fval, i32 %val32, i32 %val32, i1 %val1)
  ret void
}

define <2 x i16> @upgrade_amdgcn_ds_fadd_v2bf16__as_i16(ptr addrspace(3) %ptr, <2 x i16> %val) {
  ; CHECK: [[BC0:%[0-9]+]] = bitcast <2 x i16> %val to <2 x bfloat>
  ; CHECK-NEXT: [[RMW0:%[0-9]+]] = atomicrmw fadd ptr addrspace(3) %ptr, <2 x bfloat> [[BC0]] syncscope("agent") seq_cst, align 4
  ; CHECK-NEXT: = bitcast <2 x bfloat> [[RMW0]] to <2 x i16>
  %result0 = call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %ptr, <2 x i16> %val, i32 0, i32 0, i1 false)

  ; CHECK: [[BC1:%[0-9]+]] = bitcast <2 x i16> %val to <2 x bfloat>
  ; CHECK-NEXT: [[RMW1:%[0-9]+]] = atomicrmw volatile fadd ptr addrspace(3) %ptr, <2 x bfloat> [[BC1]] syncscope("agent") seq_cst, align 4
  ; CHECK-NEXT: = bitcast <2 x bfloat> [[RMW1]] to <2 x i16>
  %result1 = call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %ptr, <2 x i16> %val, i32 0, i32 0, i1 true)

  ; CHECK: [[BC2:%[0-9]+]] = bitcast <2 x i16> %val to <2 x bfloat>
  ; CHECK-NEXT: [[RMW2:%[0-9]+]] = atomicrmw fadd ptr addrspace(3) %ptr, <2 x bfloat> [[BC2]] syncscope("agent") seq_cst, align 4
  ; CHECK-NEXT: = bitcast <2 x bfloat> [[RMW2]] to <2 x i16>
  %result2 = call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %ptr, <2 x i16> %val, i32 43, i32 3, i1 false)

  ; CHECK: [[BC3:%[0-9]+]] = bitcast <2 x i16> %val to <2 x bfloat>
  ; CHECK-NEXT: [[RMW3:%[0-9]+]] = atomicrmw fadd ptr addrspace(3) %ptr, <2 x bfloat> [[BC3]] syncscope("agent") acquire, align 4
  ; CHECK-NEXT: = bitcast <2 x bfloat> [[RMW3]] to <2 x i16>
  %result3 = call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %ptr, <2 x i16> %val, i32 4, i32 2, i1 false)

  ret <2 x i16> %result3
}

; Somehow the bf16 version was defined as a separate intrinsic with missing arguments.
define <2 x i16> @upgrade_amdgcn_ds_fadd_v2bf16__missing_args_as_i16(ptr addrspace(3) %ptr, <2 x i16> %val) {
  ; CHECK: [[BC0:%[0-9]+]] = bitcast <2 x i16> %val to <2 x bfloat>
  ; CHECK-NEXT: [[RMW0:%[0-9]+]] = atomicrmw fadd ptr addrspace(3) %ptr, <2 x bfloat> [[BC0]] syncscope("agent") seq_cst, align 4
  ; CHECK-NEXT: [[BC1:%[0-9]+]] = bitcast <2 x bfloat> [[RMW0]] to <2 x i16>
  ; CHECK-NEXT: ret <2 x i16> [[BC1]]
  %result0 = call <2 x i16> @llvm.amdgcn.ds.fadd.v2bf16(ptr addrspace(3) %ptr, <2 x i16> %val)
  ret <2 x i16> %result0
}

declare float @llvm.amdgcn.ds.fmin.f32(ptr addrspace(3) nocapture, float, i32 immarg, i32 immarg, i1 immarg)
declare double @llvm.amdgcn.ds.fmin.f64(ptr addrspace(3) nocapture, double, i32 immarg, i32 immarg, i1 immarg)

define float @upgrade_amdgcn_ds_fmin_f32(ptr addrspace(3) %ptr, float %val) {
  ; CHECK: atomicrmw fmin ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result0 = call float @llvm.amdgcn.ds.fmin.f32(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 false)

  ; CHECK: = atomicrmw volatile fmin ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result1 = call float @llvm.amdgcn.ds.fmin.f32(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 true)

  ; CHECK: = atomicrmw fmin ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result2 = call float @llvm.amdgcn.ds.fmin.f32(ptr addrspace(3) %ptr, float %val, i32 43, i32 3, i1 false)

  ; CHECK: = atomicrmw fmin ptr addrspace(3) %ptr, float %val syncscope("agent") acquire, align 4
  %result3 = call float @llvm.amdgcn.ds.fmin.f32(ptr addrspace(3) %ptr, float %val, i32 4, i32 2, i1 false)

  ret float %result3
}

define double @upgrade_amdgcn_ds_fmin_f64(ptr addrspace(3) %ptr, double %val) {
  ; CHECK: atomicrmw fmin ptr addrspace(3) %ptr, double %val syncscope("agent") seq_cst, align 8
  %result0 = call double @llvm.amdgcn.ds.fmin.f64(ptr addrspace(3) %ptr, double %val, i32 0, i32 0, i1 false)

  ; CHECK: = atomicrmw volatile fmin ptr addrspace(3) %ptr, double %val syncscope("agent") seq_cst, align 8
  %result1 = call double @llvm.amdgcn.ds.fmin.f64(ptr addrspace(3) %ptr, double %val, i32 0, i32 0, i1 true)

  ; CHECK: = atomicrmw fmin ptr addrspace(3) %ptr, double %val syncscope("agent") seq_cst, align 8
  %result2 = call double @llvm.amdgcn.ds.fmin.f64(ptr addrspace(3) %ptr, double %val, i32 43, i32 3, i1 false)

  ; CHECK: = atomicrmw fmin ptr addrspace(3) %ptr, double %val syncscope("agent") acquire, align 8
  %result3 = call double @llvm.amdgcn.ds.fmin.f64(ptr addrspace(3) %ptr, double %val, i32 4, i32 2, i1 false)

  ret double %result3
}

declare float @llvm.amdgcn.ds.fmin(ptr addrspace(3) nocapture, float, i32 immarg, i32 immarg, i1 immarg)

define float @upgrade_amdgcn_ds_fmin_f32_no_suffix(ptr addrspace(3) %ptr, float %val) {
  ; CHECK: = atomicrmw fmin ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4

  %result0 = call float @llvm.amdgcn.ds.fmin(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 false)
  ret float %result0
}

declare float @llvm.amdgcn.ds.fmax(ptr addrspace(3) nocapture, float, i32 immarg, i32 immarg, i1 immarg)

define float @upgrade_amdgcn_ds_fmax_f32_no_suffix(ptr addrspace(3) %ptr, float %val) {
  ; CHECK: = atomicrmw fmax ptr addrspace(3) %ptr, float %val syncscope("agent") seq_cst, align 4
  %result0 = call float @llvm.amdgcn.ds.fmax(ptr addrspace(3) %ptr, float %val, i32 0, i32 0, i1 false)
  ret float %result0
}

declare <2 x i16> @llvm.amdgcn.flat.atomic.fadd.v2bf16.p0(ptr, <2 x i16>)

define <2 x i16> @upgrade_amdgcn_flat_atomic_fadd_v2bf16_p0(ptr %ptr, <2 x i16> %data) {
  ; CHECK: [[BC0:%.+]] = bitcast <2 x i16> %data to <2 x bfloat>
  ; CHECK-NEXT: [[ATOMIC:%.+]] = atomicrmw fadd ptr %ptr, <2 x bfloat> [[BC0]] syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  ; CHECK-NEXT: [[BC1:%.+]] = bitcast <2 x bfloat> [[ATOMIC]] to <2 x i16>
  ; CHECK-NEXT: ret <2 x i16> [[BC1]]
  %result = call <2 x i16> @llvm.amdgcn.flat.atomic.fadd.v2bf16.p0(ptr %ptr, <2 x i16> %data)
  ret <2 x i16> %result
}

declare <2 x i16> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1(ptr addrspace(1), <2 x i16>)

define <2 x i16> @upgrade_amdgcn_global_atomic_fadd_v2bf16_p1(ptr addrspace(1) %ptr, <2 x i16> %data) {
  ; CHECK: [[BC0:%.+]] = bitcast <2 x i16> %data to <2 x bfloat>
  ; CHECK-NEXT: [[ATOMIC:%.+]] = atomicrmw fadd ptr addrspace(1) %ptr, <2 x bfloat> [[BC0]] syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  ; CHECK-NEXT: [[BC1:%.+]] = bitcast <2 x bfloat> [[ATOMIC]] to <2 x i16>
  ; CHECK-NEXT: ret <2 x i16> [[BC1]]
  %result = call <2 x i16> @llvm.amdgcn.global.atomic.fadd.v2bf16.p1(ptr addrspace(1) %ptr, <2 x i16> %data)
  ret <2 x i16> %result
}

declare <2 x half> @llvm.amdgcn.flat.atomic.fadd.v2f16.p0.v2f16(ptr nocapture, <2 x half>) #0

define <2 x half> @upgrade_amdgcn_flat_atomic_fadd_v2f16_p0_v2f16(ptr %ptr, <2 x half> %data) {
  ; CHECK: %{{.+}} = atomicrmw fadd ptr %ptr, <2 x half> %data syncscope("agent") seq_cst, align 4, !noalias.addrspace !{{[0-9]+}}, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call <2 x half> @llvm.amdgcn.flat.atomic.fadd.v2f16.p0.v2f16(ptr %ptr, <2 x half> %data)
  ret <2 x half> %result
}

declare <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) nocapture, <2 x half>) #0

define <2 x half> @upgrade_amdgcn_global_atomic_fadd_v2f16_p1_v2f16(ptr addrspace(1) %ptr, <2 x half> %data) {
  ; CHECK: %{{.+}} = atomicrmw fadd ptr addrspace(1) %ptr, <2 x half> %data syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call <2 x half> @llvm.amdgcn.global.atomic.fadd.v2f16.p1.v2f16(ptr addrspace(1) %ptr, <2 x half> %data)
  ret <2 x half> %result
}

declare float @llvm.amdgcn.flat.atomic.fadd.f32.p0.f32(ptr nocapture, float) #0

define float @upgrade_amdgcn_flat_atomic_fadd_f32_p0_f32(ptr %ptr, float %data) {
  ; CHECK: %{{.+}} = atomicrmw fadd ptr %ptr, float %data syncscope("agent") seq_cst, align 4, !noalias.addrspace !{{[0-9]+}}, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+$}}
  %result = call float @llvm.amdgcn.flat.atomic.fadd.f32.p0.f32(ptr %ptr, float %data)
  ret float %result
}

declare float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) nocapture, float) #0

define float @upgrade_amdgcn_global_atomic_fadd_f32_p1_f32(ptr addrspace(1) %ptr, float %data) {
  ; CHECK: %{{.+}} = atomicrmw fadd ptr addrspace(1) %ptr, float %data syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+}}, !amdgpu.ignore.denormal.mode !{{[0-9]+$}}
  %result = call float @llvm.amdgcn.global.atomic.fadd.f32.p1.f32(ptr addrspace(1) %ptr, float %data)
  ret float %result
}

declare float @llvm.amdgcn.flat.atomic.fmin.f32.p0.f32(ptr nocapture, float) #0

define float @upgrade_amdgcn_flat_atomic_fmin_f32_p0_f32(ptr %ptr, float %data) {
  ; CHECK: %{{.+}} = atomicrmw fmin ptr %ptr, float %data syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call float @llvm.amdgcn.flat.atomic.fmin.f32.p0.f32(ptr %ptr, float %data)
  ret float %result
}

declare float @llvm.amdgcn.global.atomic.fmin.f32.p1.f32(ptr addrspace(1) nocapture, float) #0

define float @upgrade_amdgcn_global_atomic_fmin_f32_p1_f32(ptr addrspace(1) %ptr, float %data) {
  ; CHECK: %{{.+}} = atomicrmw fmin ptr addrspace(1) %ptr, float %data syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call float @llvm.amdgcn.global.atomic.fmin.f32.p1.f32(ptr addrspace(1) %ptr, float %data)
  ret float %result
}

declare double @llvm.amdgcn.flat.atomic.fmin.f64.p0.f64(ptr nocapture, double) #0

define double @upgrade_amdgcn_flat_atomic_fmin_f64_p0_f64(ptr %ptr, double %data) {
  ; CHECK: %{{.+}} = atomicrmw fmin ptr %ptr, double %data syncscope("agent") seq_cst, align 8, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call double @llvm.amdgcn.flat.atomic.fmin.f64.p0.f64(ptr %ptr, double %data)
  ret double %result
}

declare double @llvm.amdgcn.global.atomic.fmin.f64.p1.f64(ptr addrspace(1) nocapture, double) #0

define double @upgrade_amdgcn_global_atomic_fmin_f64_p1_f64(ptr addrspace(1) %ptr, double %data) {
  ; CHECK: %{{.+}} = atomicrmw fmin ptr addrspace(1) %ptr, double %data syncscope("agent") seq_cst, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call double @llvm.amdgcn.global.atomic.fmin.f64.p1.f64(ptr addrspace(1) %ptr, double %data)
  ret double %result
}

declare float @llvm.amdgcn.flat.atomic.fmax.f32.p0.f32(ptr nocapture, float) #0

define float @upgrade_amdgcn_flat_atomic_fmax_f32_p0_f32(ptr %ptr, float %data) {
  ; CHECK: %{{.+}} = atomicrmw fmax ptr %ptr, float %data syncscope("agent") seq_cst, align 4, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call float @llvm.amdgcn.flat.atomic.fmax.f32.p0.f32(ptr %ptr, float %data)
  ret float %result
}

declare float @llvm.amdgcn.global.atomic.fmax.f32.p1.f32(ptr addrspace(1) nocapture, float) #0

define float @upgrade_amdgcn_global_atomic_fmax_f32_p1_f32(ptr addrspace(1) %ptr, float %data) {
  ; CHECK: %{{.+}} = atomicrmw fmax ptr addrspace(1) %ptr, float %data syncscope("agent") seq_cst, align 4, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call float @llvm.amdgcn.global.atomic.fmax.f32.p1.f32(ptr addrspace(1) %ptr, float %data)
  ret float %result
}

declare double @llvm.amdgcn.flat.atomic.fmax.f64.p0.f64(ptr nocapture, double) #0

define double @upgrade_amdgcn_flat_atomic_fmax_f64_p0_f64(ptr %ptr, double %data) {
  ; CHECK: %{{.+}} = atomicrmw fmax ptr %ptr, double %data syncscope("agent") seq_cst, align 8, !noalias.addrspace !0, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call double @llvm.amdgcn.flat.atomic.fmax.f64.p0.f64(ptr %ptr, double %data)
  ret double %result
}

declare double @llvm.amdgcn.global.atomic.fmax.f64.p1.f64(ptr addrspace(1) nocapture, double) #0

define double @upgrade_amdgcn_global_atomic_fmax_f64_p1_f64(ptr addrspace(1) %ptr, double %data) {
  ; CHECK: %{{.+}} = atomicrmw fmax ptr addrspace(1) %ptr, double %data syncscope("agent") seq_cst, align 8, !amdgpu.no.fine.grained.memory !{{[0-9]+$}}
  %result = call double @llvm.amdgcn.global.atomic.fmax.f64.p1.f64(ptr addrspace(1) %ptr, double %data)
  ret double %result
}

attributes #0 = { argmemonly nounwind willreturn }

; CHECK: !0 = !{i32 5, i32 6}
; CHECK: !1 = !{}
