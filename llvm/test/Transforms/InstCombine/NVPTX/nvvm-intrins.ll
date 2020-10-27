; Check that nvvm intrinsics get simplified to target-generic intrinsics where
; possible.
;
; We run this test twice; once with ftz on, and again with ftz off.  Behold the
; hackery:

; RUN: cat %s > %t.ftz
; RUN: echo 'attributes #0 = { "denormal-fp-math-f32" = "preserve-sign" }' >> %t.ftz
; RUN: opt < %t.ftz -instcombine -mtriple=nvptx64-nvidia-cuda -S | FileCheck %s --check-prefix=CHECK --check-prefix=FTZ

; RUN: cat %s > %t.noftz
; RUN: echo 'attributes #0 = { "denormal-fp-math-f32" = "ieee" }' >> %t.noftz
; RUN: opt < %t.noftz -instcombine -mtriple=nvptx64-nvidia-cuda -S | FileCheck %s --check-prefix=CHECK --check-prefix=NOFTZ

; We handle nvvm intrinsics with ftz variants as follows:
;  - If the module is in ftz mode, the ftz variant is transformed into the
;    regular llvm intrinsic, and the non-ftz variant is left alone.
;  - If the module is not in ftz mode, it's the reverse: Only the non-ftz
;    variant is transformed, and the ftz variant is left alone.

; Check NVVM intrinsics that map directly to LLVM target-generic intrinsics.

; CHECK-LABEL: @ceil_double
define double @ceil_double(double %a) #0 {
; CHECK: call double @llvm.ceil.f64
  %ret = call double @llvm.nvvm.ceil.d(double %a)
  ret double %ret
}
; CHECK-LABEL: @ceil_float
define float @ceil_float(float %a) #0 {
; NOFTZ: call float @llvm.ceil.f32
; FTZ: call float @llvm.nvvm.ceil.f
  %ret = call float @llvm.nvvm.ceil.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @ceil_float_ftz
define float @ceil_float_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.ceil.ftz.f
; FTZ: call float @llvm.ceil.f32
  %ret = call float @llvm.nvvm.ceil.ftz.f(float %a)
  ret float %ret
}

; CHECK-LABEL: @fabs_double
define double @fabs_double(double %a) #0 {
; CHECK: call double @llvm.fabs.f64
  %ret = call double @llvm.nvvm.fabs.d(double %a)
  ret double %ret
}
; CHECK-LABEL: @fabs_float
define float @fabs_float(float %a) #0 {
; NOFTZ: call float @llvm.fabs.f32
; FTZ: call float @llvm.nvvm.fabs.f
  %ret = call float @llvm.nvvm.fabs.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @fabs_float_ftz
define float @fabs_float_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.fabs.ftz.f
; FTZ: call float @llvm.fabs.f32
  %ret = call float @llvm.nvvm.fabs.ftz.f(float %a)
  ret float %ret
}

; CHECK-LABEL: @floor_double
define double @floor_double(double %a) #0 {
; CHECK: call double @llvm.floor.f64
  %ret = call double @llvm.nvvm.floor.d(double %a)
  ret double %ret
}
; CHECK-LABEL: @floor_float
define float @floor_float(float %a) #0 {
; NOFTZ: call float @llvm.floor.f32
; FTZ: call float @llvm.nvvm.floor.f
  %ret = call float @llvm.nvvm.floor.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @floor_float_ftz
define float @floor_float_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.floor.ftz.f
; FTZ: call float @llvm.floor.f32
  %ret = call float @llvm.nvvm.floor.ftz.f(float %a)
  ret float %ret
}

; CHECK-LABEL: @fma_double
define double @fma_double(double %a, double %b, double %c) #0 {
; CHECK: call double @llvm.fma.f64
  %ret = call double @llvm.nvvm.fma.rn.d(double %a, double %b, double %c)
  ret double %ret
}
; CHECK-LABEL: @fma_float
define float @fma_float(float %a, float %b, float %c) #0 {
; NOFTZ: call float @llvm.fma.f32
; FTZ: call float @llvm.nvvm.fma.rn.f
  %ret = call float @llvm.nvvm.fma.rn.f(float %a, float %b, float %c)
  ret float %ret
}
; CHECK-LABEL: @fma_float_ftz
define float @fma_float_ftz(float %a, float %b, float %c) #0 {
; NOFTZ: call float @llvm.nvvm.fma.rn.ftz.f
; FTZ: call float @llvm.fma.f32
  %ret = call float @llvm.nvvm.fma.rn.ftz.f(float %a, float %b, float %c)
  ret float %ret
}

; CHECK-LABEL: @fmax_double
define double @fmax_double(double %a, double %b) #0 {
; CHECK: call double @llvm.maxnum.f64
  %ret = call double @llvm.nvvm.fmax.d(double %a, double %b)
  ret double %ret
}
; CHECK-LABEL: @fmax_float
define float @fmax_float(float %a, float %b) #0 {
; NOFTZ: call float @llvm.maxnum.f32
; FTZ: call float @llvm.nvvm.fmax.f
  %ret = call float @llvm.nvvm.fmax.f(float %a, float %b)
  ret float %ret
}
; CHECK-LABEL: @fmax_float_ftz
define float @fmax_float_ftz(float %a, float %b) #0 {
; NOFTZ: call float @llvm.nvvm.fmax.ftz.f
; FTZ: call float @llvm.maxnum.f32
  %ret = call float @llvm.nvvm.fmax.ftz.f(float %a, float %b)
  ret float %ret
}

; CHECK-LABEL: @fmin_double
define double @fmin_double(double %a, double %b) #0 {
; CHECK: call double @llvm.minnum.f64
  %ret = call double @llvm.nvvm.fmin.d(double %a, double %b)
  ret double %ret
}
; CHECK-LABEL: @fmin_float
define float @fmin_float(float %a, float %b) #0 {
; NOFTZ: call float @llvm.minnum.f32
; FTZ: call float @llvm.nvvm.fmin.f
  %ret = call float @llvm.nvvm.fmin.f(float %a, float %b)
  ret float %ret
}
; CHECK-LABEL: @fmin_float_ftz
define float @fmin_float_ftz(float %a, float %b) #0 {
; NOFTZ: call float @llvm.nvvm.fmin.ftz.f
; FTZ: call float @llvm.minnum.f32
  %ret = call float @llvm.nvvm.fmin.ftz.f(float %a, float %b)
  ret float %ret
}

; CHECK-LABEL: @round_double
define double @round_double(double %a) #0 {
; CHECK: call double @llvm.round.f64
  %ret = call double @llvm.nvvm.round.d(double %a)
  ret double %ret
}
; CHECK-LABEL: @round_float
define float @round_float(float %a) #0 {
; NOFTZ: call float @llvm.round.f32
; FTZ: call float @llvm.nvvm.round.f
  %ret = call float @llvm.nvvm.round.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @round_float_ftz
define float @round_float_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.round.ftz.f
; FTZ: call float @llvm.round.f32
  %ret = call float @llvm.nvvm.round.ftz.f(float %a)
  ret float %ret
}

; CHECK-LABEL: @trunc_double
define double @trunc_double(double %a) #0 {
; CHECK: call double @llvm.trunc.f64
  %ret = call double @llvm.nvvm.trunc.d(double %a)
  ret double %ret
}
; CHECK-LABEL: @trunc_float
define float @trunc_float(float %a) #0 {
; NOFTZ: call float @llvm.trunc.f32
; FTZ: call float @llvm.nvvm.trunc.f
  %ret = call float @llvm.nvvm.trunc.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @trunc_float_ftz
define float @trunc_float_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.trunc.ftz.f
; FTZ: call float @llvm.trunc.f32
  %ret = call float @llvm.nvvm.trunc.ftz.f(float %a)
  ret float %ret
}

; Check NVVM intrinsics that correspond to LLVM cast operations.

; CHECK-LABEL: @test_d2i
define i32 @test_d2i(double %a) #0 {
; CHECK: fptosi double %a to i32
  %ret = call i32 @llvm.nvvm.d2i.rz(double %a)
  ret i32 %ret
}
; CHECK-LABEL: @test_f2i
define i32 @test_f2i(float %a) #0 {
; CHECK: fptosi float %a to i32
  %ret = call i32 @llvm.nvvm.f2i.rz(float %a)
  ret i32 %ret
}
; CHECK-LABEL: @test_d2ll
define i64 @test_d2ll(double %a) #0 {
; CHECK: fptosi double %a to i64
  %ret = call i64 @llvm.nvvm.d2ll.rz(double %a)
  ret i64 %ret
}
; CHECK-LABEL: @test_f2ll
define i64 @test_f2ll(float %a) #0 {
; CHECK: fptosi float %a to i64
  %ret = call i64 @llvm.nvvm.f2ll.rz(float %a)
  ret i64 %ret
}
; CHECK-LABEL: @test_d2ui
define i32 @test_d2ui(double %a) #0 {
; CHECK: fptoui double %a to i32
  %ret = call i32 @llvm.nvvm.d2ui.rz(double %a)
  ret i32 %ret
}
; CHECK-LABEL: @test_f2ui
define i32 @test_f2ui(float %a) #0 {
; CHECK: fptoui float %a to i32
  %ret = call i32 @llvm.nvvm.f2ui.rz(float %a)
  ret i32 %ret
}
; CHECK-LABEL: @test_d2ull
define i64 @test_d2ull(double %a) #0 {
; CHECK: fptoui double %a to i64
  %ret = call i64 @llvm.nvvm.d2ull.rz(double %a)
  ret i64 %ret
}
; CHECK-LABEL: @test_f2ull
define i64 @test_f2ull(float %a) #0 {
; CHECK: fptoui float %a to i64
  %ret = call i64 @llvm.nvvm.f2ull.rz(float %a)
  ret i64 %ret
}

; CHECK-LABEL: @test_i2d
define double @test_i2d(i32 %a) #0 {
; CHECK: sitofp i32 %a to double
  %ret = call double @llvm.nvvm.i2d.rz(i32 %a)
  ret double %ret
}
; CHECK-LABEL: @test_i2f
define float @test_i2f(i32 %a) #0 {
; CHECK: sitofp i32 %a to float
  %ret = call float @llvm.nvvm.i2f.rz(i32 %a)
  ret float %ret
}
; CHECK-LABEL: @test_ll2d
define double @test_ll2d(i64 %a) #0 {
; CHECK: sitofp i64 %a to double
  %ret = call double @llvm.nvvm.ll2d.rz(i64 %a)
  ret double %ret
}
; CHECK-LABEL: @test_ll2f
define float @test_ll2f(i64 %a) #0 {
; CHECK: sitofp i64 %a to float
  %ret = call float @llvm.nvvm.ll2f.rz(i64 %a)
  ret float %ret
}
; CHECK-LABEL: @test_ui2d
define double @test_ui2d(i32 %a) #0 {
; CHECK: uitofp i32 %a to double
  %ret = call double @llvm.nvvm.ui2d.rz(i32 %a)
  ret double %ret
}
; CHECK-LABEL: @test_ui2f
define float @test_ui2f(i32 %a) #0 {
; CHECK: uitofp i32 %a to float
  %ret = call float @llvm.nvvm.ui2f.rz(i32 %a)
  ret float %ret
}
; CHECK-LABEL: @test_ull2d
define double @test_ull2d(i64 %a) #0 {
; CHECK: uitofp i64 %a to double
  %ret = call double @llvm.nvvm.ull2d.rz(i64 %a)
  ret double %ret
}
; CHECK-LABEL: @test_ull2f
define float @test_ull2f(i64 %a) #0 {
; CHECK: uitofp i64 %a to float
  %ret = call float @llvm.nvvm.ull2f.rz(i64 %a)
  ret float %ret
}

; Check NVVM intrinsics that map to LLVM binary operations.

; CHECK-LABEL: @test_add_rn_d
define double @test_add_rn_d(double %a, double %b) #0 {
; CHECK: fadd
  %ret = call double @llvm.nvvm.add.rn.d(double %a, double %b)
  ret double %ret
}
; CHECK-LABEL: @test_add_rn_f
define float @test_add_rn_f(float %a, float %b) #0 {
; NOFTZ: fadd
; FTZ: call float @llvm.nvvm.add.rn.f
  %ret = call float @llvm.nvvm.add.rn.f(float %a, float %b)
  ret float %ret
}
; CHECK-LABEL: @test_add_rn_f_ftz
define float @test_add_rn_f_ftz(float %a, float %b) #0 {
; NOFTZ: call float @llvm.nvvm.add.rn.f
; FTZ: fadd
  %ret = call float @llvm.nvvm.add.rn.ftz.f(float %a, float %b)
  ret float %ret
}

; CHECK-LABEL: @test_mul_rn_d
define double @test_mul_rn_d(double %a, double %b) #0 {
; CHECK: fmul
  %ret = call double @llvm.nvvm.mul.rn.d(double %a, double %b)
  ret double %ret
}
; CHECK-LABEL: @test_mul_rn_f
define float @test_mul_rn_f(float %a, float %b) #0 {
; NOFTZ: fmul
; FTZ: call float @llvm.nvvm.mul.rn.f
  %ret = call float @llvm.nvvm.mul.rn.f(float %a, float %b)
  ret float %ret
}
; CHECK-LABEL: @test_mul_rn_f_ftz
define float @test_mul_rn_f_ftz(float %a, float %b) #0 {
; NOFTZ: call float @llvm.nvvm.mul.rn.f
; FTZ: fmul
  %ret = call float @llvm.nvvm.mul.rn.ftz.f(float %a, float %b)
  ret float %ret
}

; CHECK-LABEL: @test_div_rn_d
define double @test_div_rn_d(double %a, double %b) #0 {
; CHECK: fdiv
  %ret = call double @llvm.nvvm.div.rn.d(double %a, double %b)
  ret double %ret
}
; CHECK-LABEL: @test_div_rn_f
define float @test_div_rn_f(float %a, float %b) #0 {
; NOFTZ: fdiv
; FTZ: call float @llvm.nvvm.div.rn.f
  %ret = call float @llvm.nvvm.div.rn.f(float %a, float %b)
  ret float %ret
}
; CHECK-LABEL: @test_div_rn_f_ftz
define float @test_div_rn_f_ftz(float %a, float %b) #0 {
; NOFTZ: call float @llvm.nvvm.div.rn.f
; FTZ: fdiv
  %ret = call float @llvm.nvvm.div.rn.ftz.f(float %a, float %b)
  ret float %ret
}

; Check NVVM intrinsics that require us to emit custom IR.

; CHECK-LABEL: @test_rcp_rn_f
define float @test_rcp_rn_f(float %a) #0 {
; NOFTZ: fdiv float 1.0{{.*}} %a
; FTZ: call float @llvm.nvvm.rcp.rn.f
  %ret = call float @llvm.nvvm.rcp.rn.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @test_rcp_rn_f_ftz
define float @test_rcp_rn_f_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.rcp.rn.f
; FTZ: fdiv float 1.0{{.*}} %a
  %ret = call float @llvm.nvvm.rcp.rn.ftz.f(float %a)
  ret float %ret
}

; CHECK-LABEL: @test_sqrt_rn_d
define double @test_sqrt_rn_d(double %a) #0 {
; CHECK: call double @llvm.sqrt.f64(double %a)
  %ret = call double @llvm.nvvm.sqrt.rn.d(double %a)
  ret double %ret
}
; nvvm.sqrt.f is a special case: It goes to a llvm.sqrt.f
; CHECK-LABEL: @test_sqrt_f
define float @test_sqrt_f(float %a) #0 {
; CHECK: call float @llvm.sqrt.f32(float %a)
  %ret = call float @llvm.nvvm.sqrt.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @test_sqrt_rn_f
define float @test_sqrt_rn_f(float %a) #0 {
; NOFTZ: call float @llvm.sqrt.f32(float %a)
; FTZ: call float @llvm.nvvm.sqrt.rn.f
  %ret = call float @llvm.nvvm.sqrt.rn.f(float %a)
  ret float %ret
}
; CHECK-LABEL: @test_sqrt_rn_f_ftz
define float @test_sqrt_rn_f_ftz(float %a) #0 {
; NOFTZ: call float @llvm.nvvm.sqrt.rn.f
; FTZ: call float @llvm.sqrt.f32(float %a)
  %ret = call float @llvm.nvvm.sqrt.rn.ftz.f(float %a)
  ret float %ret
}

declare double @llvm.nvvm.add.rn.d(double, double)
declare float @llvm.nvvm.add.rn.f(float, float)
declare float @llvm.nvvm.add.rn.ftz.f(float, float)
declare double @llvm.nvvm.ceil.d(double)
declare float @llvm.nvvm.ceil.f(float)
declare float @llvm.nvvm.ceil.ftz.f(float)
declare float @llvm.nvvm.d2f.rm(double)
declare float @llvm.nvvm.d2f.rm.ftz(double)
declare float @llvm.nvvm.d2f.rp(double)
declare float @llvm.nvvm.d2f.rp.ftz(double)
declare float @llvm.nvvm.d2f.rz(double)
declare float @llvm.nvvm.d2f.rz.ftz(double)
declare i32 @llvm.nvvm.d2i.rz(double)
declare i64 @llvm.nvvm.d2ll.rz(double)
declare i32 @llvm.nvvm.d2ui.rz(double)
declare i64 @llvm.nvvm.d2ull.rz(double)
declare double @llvm.nvvm.div.rn.d(double, double)
declare float @llvm.nvvm.div.rn.f(float, float)
declare float @llvm.nvvm.div.rn.ftz.f(float, float)
declare i16 @llvm.nvvm.f2h.rz(float)
declare i16 @llvm.nvvm.f2h.rz.ftz(float)
declare i32 @llvm.nvvm.f2i.rz(float)
declare i32 @llvm.nvvm.f2i.rz.ftz(float)
declare i64 @llvm.nvvm.f2ll.rz(float)
declare i64 @llvm.nvvm.f2ll.rz.ftz(float)
declare i32 @llvm.nvvm.f2ui.rz(float)
declare i32 @llvm.nvvm.f2ui.rz.ftz(float)
declare i64 @llvm.nvvm.f2ull.rz(float)
declare i64 @llvm.nvvm.f2ull.rz.ftz(float)
declare double @llvm.nvvm.fabs.d(double)
declare float @llvm.nvvm.fabs.f(float)
declare float @llvm.nvvm.fabs.ftz.f(float)
declare double @llvm.nvvm.floor.d(double)
declare float @llvm.nvvm.floor.f(float)
declare float @llvm.nvvm.floor.ftz.f(float)
declare double @llvm.nvvm.fma.rn.d(double, double, double)
declare float @llvm.nvvm.fma.rn.f(float, float, float)
declare float @llvm.nvvm.fma.rn.ftz.f(float, float, float)
declare double @llvm.nvvm.fmax.d(double, double)
declare float @llvm.nvvm.fmax.f(float, float)
declare float @llvm.nvvm.fmax.ftz.f(float, float)
declare double @llvm.nvvm.fmin.d(double, double)
declare float @llvm.nvvm.fmin.f(float, float)
declare float @llvm.nvvm.fmin.ftz.f(float, float)
declare double @llvm.nvvm.i2d.rz(i32)
declare float @llvm.nvvm.i2f.rz(i32)
declare double @llvm.nvvm.ll2d.rz(i64)
declare float @llvm.nvvm.ll2f.rz(i64)
declare double @llvm.nvvm.lohi.i2d(i32, i32)
declare double @llvm.nvvm.mul.rn.d(double, double)
declare float @llvm.nvvm.mul.rn.f(float, float)
declare float @llvm.nvvm.mul.rn.ftz.f(float, float)
declare double @llvm.nvvm.rcp.rm.d(double)
declare double @llvm.nvvm.rcp.rn.d(double)
declare float @llvm.nvvm.rcp.rn.f(float)
declare float @llvm.nvvm.rcp.rn.ftz.f(float)
declare double @llvm.nvvm.round.d(double)
declare float @llvm.nvvm.round.f(float)
declare float @llvm.nvvm.round.ftz.f(float)
declare float @llvm.nvvm.sqrt.f(float)
declare double @llvm.nvvm.sqrt.rn.d(double)
declare float @llvm.nvvm.sqrt.rn.f(float)
declare float @llvm.nvvm.sqrt.rn.ftz.f(float)
declare double @llvm.nvvm.trunc.d(double)
declare float @llvm.nvvm.trunc.f(float)
declare float @llvm.nvvm.trunc.ftz.f(float)
declare double @llvm.nvvm.ui2d.rz(i32)
declare float @llvm.nvvm.ui2f.rn(i32)
declare float @llvm.nvvm.ui2f.rz(i32)
declare double @llvm.nvvm.ull2d.rz(i64)
declare float @llvm.nvvm.ull2f.rz(i64)
