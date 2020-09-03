; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s

; Even though general vector types are not supported in PTX, we can still
; optimize loads/stores with pseudo-vector instructions of the form:
;
; ld.v2.f32 {%f0, %f1}, [%r0]
;
; which will load two floats at once into scalar registers.

; CHECK-LABEL: foo
define void @foo(<2 x float>* %a) {
; CHECK: ld.v2.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <2 x float>, <2 x float>* %a
  %t2 = fmul <2 x float> %t1, %t1
  store <2 x float> %t2, <2 x float>* %a
  ret void
}

; CHECK-LABEL: foo2
define void @foo2(<4 x float>* %a) {
; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <4 x float>, <4 x float>* %a
  %t2 = fmul <4 x float> %t1, %t1
  store <4 x float> %t2, <4 x float>* %a
  ret void
}

; CHECK-LABEL: foo3
define void @foo3(<8 x float>* %a) {
; CHECK: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
; CHECK-NEXT: ld.v4.f32 {%f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}, %f{{[0-9]+}}}
  %t1 = load <8 x float>, <8 x float>* %a
  %t2 = fmul <8 x float> %t1, %t1
  store <8 x float> %t2, <8 x float>* %a
  ret void
}



; CHECK-LABEL: foo4
define void @foo4(<2 x i32>* %a) {
; CHECK: ld.v2.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <2 x i32>, <2 x i32>* %a
  %t2 = mul <2 x i32> %t1, %t1
  store <2 x i32> %t2, <2 x i32>* %a
  ret void
}

; CHECK-LABEL: foo5
define void @foo5(<4 x i32>* %a) {
; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <4 x i32>, <4 x i32>* %a
  %t2 = mul <4 x i32> %t1, %t1
  store <4 x i32> %t2, <4 x i32>* %a
  ret void
}

; CHECK-LABEL: foo6
define void @foo6(<8 x i32>* %a) {
; CHECK: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
; CHECK-NEXT: ld.v4.u32 {%r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}, %r{{[0-9]+}}}
  %t1 = load <8 x i32>, <8 x i32>* %a
  %t2 = mul <8 x i32> %t1, %t1
  store <8 x i32> %t2, <8 x i32>* %a
  ret void
}

; The following test wasn't passing previously as the address
; computation was still too complex when LSV was called.
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
; CHECK-LABEL: foo_complex
define void @foo_complex(i8* nocapture readonly align 16 dereferenceable(134217728) %alloc0) {
  %targ0.1.typed = bitcast i8* %alloc0 to [1024 x [131072 x i8]]*
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
  %t27 = getelementptr inbounds [1024 x [131072 x i8]], [1024 x [131072 x i8]]* %targ0.1.typed, i64 0, i64 %t20, i64 %t9
; CHECK: ld.v2.u8
  %t28 = load i8, i8* %t27, align 2
  %t31 = getelementptr inbounds [1024 x [131072 x i8]], [1024 x [131072 x i8]]* %targ0.1.typed, i64 0, i64 %t20, i64 %t11
  %t32 = load i8, i8* %t31, align 1
  %t33 = icmp ult i8 %t28, %t32
  %t34 = select i1 %t33, i8 %t32, i8 %t28
  store i8 %t34, i8* %t31
; CHECK: ret
  ret void
}


!1 = !{i32 0, i32 64}
