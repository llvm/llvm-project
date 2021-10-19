; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

declare i16 @llvm.convert.to.fp16.f32(float %a)
declare i16 @llvm.convert.to.fp16.f64(double %a)

declare float @llvm.convert.from.fp16.f32(i16 %a)
declare double @llvm.convert.from.fp16.f64(i16 %a)

define float @func_i16fp32(i16* %a) {
; CHECK-LABEL: func_i16fp32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    lea %s1, __gnu_h2f_ieee@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_h2f_ieee@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %a.val = load i16, i16* %a, align 4
  %a.asd = call float @llvm.convert.from.fp16.f32(i16 %a.val)
  ret float %a.asd
}

define double @func_i16fp64(i16* %a) {
; CHECK-LABEL: func_i16fp64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    lea %s1, __gnu_h2f_ieee@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_h2f_ieee@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %a.val = load i16, i16* %a, align 4
  %a.asd = call double @llvm.convert.from.fp16.f64(i16 %a.val)
  ret double %a.asd
}

define float @func_fp16fp32(half* %a) {
; CHECK-LABEL: func_fp16fp32:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    lea %s1, __gnu_h2f_ieee@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_h2f_ieee@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9
  %a.val = load half, half* %a, align 4
  %a.asd = fpext half %a.val to float
  ret float %a.asd
}

define double @func_fp16fp64(half* %a) {
; CHECK-LABEL: func_fp16fp64:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    ld2b.zx %s0, (, %s0)
; CHECK-NEXT:    lea %s1, __gnu_h2f_ieee@lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_h2f_ieee@hi(, %s1)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    or %s11, 0, %s9
  %a.val = load half, half* %a, align 4
  %a.asd = fpext half %a.val to double
  ret double %a.asd
}

define void @func_fp32i16(i16* %fl.ptr, float %val) {
; CHECK-LABEL: func_fp32i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    or %s18, 0, %s0
; CHECK-NEXT:    lea %s0, __gnu_f2h_ieee@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_f2h_ieee@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    st2b %s0, (, %s18)
; CHECK-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %val.asf = call i16 @llvm.convert.to.fp16.f32(float %val)
  store i16 %val.asf, i16* %fl.ptr
  ret void
}

define half @func_fp32fp16(half* %fl.ptr, float %a) {
; CHECK-LABEL: func_fp32fp16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    st %s19, 296(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    or %s18, 0, %s0
; CHECK-NEXT:    lea %s0, __gnu_f2h_ieee@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_f2h_ieee@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s19, 0, %s0
; CHECK-NEXT:    lea %s0, __gnu_h2f_ieee@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __gnu_h2f_ieee@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s19
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    st2b %s19, (, %s18)
; CHECK-NEXT:    ld %s19, 296(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %a.asd = fptrunc float %a to half
  store half %a.asd, half* %fl.ptr
  ret half %a.asd
}

define double @func_fp32fp64(float* %a) {
; CHECK-LABEL: func_fp32fp64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    ldu %s0, (, %s0)
; CHECK-NEXT:    cvt.d.s %s0, %s0
; CHECK-NEXT:    b.l.t (, %s10)
  %a.val = load float, float* %a, align 4
  %a.asd = fpext float %a.val to double
  ret double %a.asd
}

define void @func_fp64i16(i16* %fl.ptr, double %val) {
; CHECK-LABEL: func_fp64i16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    or %s18, 0, %s0
; CHECK-NEXT:    lea %s0, __truncdfhf2@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __truncdfhf2@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    st2b %s0, (, %s18)
; CHECK-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %val.asf = call i16 @llvm.convert.to.fp16.f64(double %val)
  store i16 %val.asf, i16* %fl.ptr
  ret void
}

define void @func_fp64fp16(half* %fl.ptr, double %val) {
; CHECK-LABEL: func_fp64fp16:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s18, 288(, %s11) # 8-byte Folded Spill
; CHECK-NEXT:    or %s18, 0, %s0
; CHECK-NEXT:    lea %s0, __truncdfhf2@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, __truncdfhf2@hi(, %s0)
; CHECK-NEXT:    or %s0, 0, %s1
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    st2b %s0, (, %s18)
; CHECK-NEXT:    ld %s18, 288(, %s11) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  %val.asf = fptrunc double %val to half
  store half %val.asf, half* %fl.ptr
  ret void
}

define void @func_fp64fp32(float* %fl.ptr, double %val) {
; CHECK-LABEL: func_fp64fp32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    cvt.s.d %s1, %s1
; CHECK-NEXT:    stu %s1, (, %s0)
; CHECK-NEXT:    b.l.t (, %s10)
  %val.asf = fptrunc double %val to float
  store float %val.asf, float* %fl.ptr
  ret void
}
