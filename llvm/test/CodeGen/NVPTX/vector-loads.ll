; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; Even though general vector types are not supported in PTX, we can still
; optimize loads/stores with pseudo-vector instructions of the form:
;
; ld.v2.f32 {%f0, %f1}, [%r0]
;
; which will load two floats at once into scalar registers.

; CHECK-LABEL: foo
define void @foo(ptr %a) {
; CHECK: ld.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <2 x float>, ptr %a
  %t2 = fmul <2 x float> %t1, %t1
  store <2 x float> %t2, ptr %a
  ret void
}

; CHECK-LABEL: foo2
define void @foo2(ptr %a) {
; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <4 x float>, ptr %a
  %t2 = fmul <4 x float> %t1, %t1
  store <4 x float> %t2, ptr %a
  ret void
}

; CHECK-LABEL: foo3
define void @foo3(ptr %a) {
; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
; CHECK-NEXT: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <8 x float>, ptr %a
  %t2 = fmul <8 x float> %t1, %t1
  store <8 x float> %t2, ptr %a
  ret void
}



; CHECK-LABEL: foo4
define void @foo4(ptr %a) {
; CHECK: ld.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <2 x i32>, ptr %a
  %t2 = mul <2 x i32> %t1, %t1
  store <2 x i32> %t2, ptr %a
  ret void
}

; CHECK-LABEL: foo5
define void @foo5(ptr %a) {
; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <4 x i32>, ptr %a
  %t2 = mul <4 x i32> %t1, %t1
  store <4 x i32> %t2, ptr %a
  ret void
}

; CHECK-LABEL: foo6
define void @foo6(ptr %a) {
; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
; CHECK-NEXT: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <8 x i32>, ptr %a
  %t2 = mul <8 x i32> %t1, %t1
  store <8 x i32> %t2, ptr %a
  ret void
}

; The following test wasn't passing previously as the address
; computation was still too complex when LSV was called.
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
; CHECK-LABEL: foo_complex
define void @foo_complex(ptr nocapture readonly align 16 dereferenceable(134217728) %alloc0) {
  %t0 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !1
  %t1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %t2 = lshr i32 %t1, 8
  %t3 = shl nuw nsw i32 %t1, 9
  %ttile_origin.2 = and i32 %t3, 130560
  %tstart_offset_x_mul = shl nuw nsw i32 %t0, 1
  %t4 = or i32 %ttile_origin.2, %tstart_offset_x_mul
  %t6 = or i32 %t4, 1
  %t8 = or i32 %t4, 128
  %t9 = zext i32 %t8 to i64
  %t10 = or i32 %t4, 129
  %t11 = zext i32 %t10 to i64
  %t20 = zext i32 %t2 to i64
  %t27 = getelementptr inbounds [1024 x [131072 x i8]], ptr %alloc0, i64 0, i64 %t20, i64 %t9
; CHECK: ld.v2.u8
  %t28 = load i8, ptr %t27, align 2
  %t31 = getelementptr inbounds [1024 x [131072 x i8]], ptr %alloc0, i64 0, i64 %t20, i64 %t11
  %t32 = load i8, ptr %t31, align 1
  %t33 = icmp ult i8 %t28, %t32
  %t34 = select i1 %t33, i8 %t32, i8 %t28
  store i8 %t34, ptr %t31
; CHECK: ret
  ret void
}

; CHECK-LABEL: extv8f16_global_a16(
define void @extv8f16_global_a16(ptr addrspace(1) noalias readonly align 16 %dst, ptr addrspace(1) noalias readonly align 16 %src) #0 {
; CHECK: ld.global.v4.b16 {%f
; CHECK: ld.global.v4.b16 {%f
  %v = load <8 x half>, ptr addrspace(1) %src, align 16
  %ext = fpext <8 x half> %v to <8 x float>
; CHECK: st.global.v4.f32
; CHECK: st.global.v4.f32
  store <8 x float> %ext, ptr addrspace(1) %dst, align 16
  ret void
}

; CHECK-LABEL: extv8f16_global_a4(
define void @extv8f16_global_a4(ptr addrspace(1) noalias readonly align 16 %dst, ptr addrspace(1) noalias readonly align 16 %src) #0 {
; CHECK: ld.global.v2.b16 {%f
; CHECK: ld.global.v2.b16 {%f
; CHECK: ld.global.v2.b16 {%f
; CHECK: ld.global.v2.b16 {%f
  %v = load <8 x half>, ptr addrspace(1) %src, align 4
  %ext = fpext <8 x half> %v to <8 x float>
; CHECK: st.global.v4.f32
; CHECK: st.global.v4.f32
  store <8 x float> %ext, ptr addrspace(1) %dst, align 16
  ret void
}


; CHECK-LABEL: extv8f16_generic_a16(
define void @extv8f16_generic_a16(ptr noalias readonly align 16 %dst, ptr noalias readonly align 16 %src) #0 {
; CHECK: ld.v4.b16 {%f
; CHECK: ld.v4.b16 {%f
  %v = load <8 x half>, ptr %src, align 16
  %ext = fpext <8 x half> %v to <8 x float>
; CHECK: st.v4.f32
; CHECK: st.v4.f32
  store <8 x float> %ext, ptr %dst, align 16
  ret void
}

; CHECK-LABEL: extv8f16_generic_a4(
define void @extv8f16_generic_a4(ptr noalias readonly align 16 %dst, ptr noalias readonly align 16 %src) #0 {
; CHECK: ld.v2.b16 {%f
; CHECK: ld.v2.b16 {%f
; CHECK: ld.v2.b16 {%f
; CHECK: ld.v2.b16 {%f
  %v = load <8 x half>, ptr %src, align 4
  %ext = fpext <8 x half> %v to <8 x float>
; CHECK: st.v4.f32
; CHECK: st.v4.f32
  store <8 x float> %ext, ptr %dst, align 16
  ret void
}


!1 = !{i32 0, i32 64}
