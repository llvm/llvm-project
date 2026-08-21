; RUN: opt -mtriple=wasm32 -mattr=+simd128 -passes=slp-vectorizer %s | llc -mtriple=wasm32 -mattr=+simd128 -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s

%struct.TwoBytes = type { i8, i8 }
%struct.FourBytes = type { i8, i8, i8, i8 }
%struct.TwoFloats = type { float, float }
%struct.FourFloats = type { float, float, float, float }

; CHECK-LABEL: mac_2d_f32_i8_fmuladd:
; CHECK-NOT: v128.load
define hidden void @mac_2d_f32_i8_fmuladd(ptr dead_on_unwind noalias writable sret(%struct.TwoFloats) align 4 captures(none) %agg.result, ptr noundef readonly captures(none) %x, ptr noundef readonly captures(none) %y, i32 noundef %n) {
entry:
  %agg.result.promoted = load float, ptr %agg.result, align 4
  %cmp18.not = icmp eq i32 %n, 0
  br i1 %cmp18.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:
  %b10 = getelementptr inbounds nuw i8, ptr %agg.result, i32 4
  %b10.promoted = load float, ptr %b10, align 4
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  store float %7, ptr %b10, align 4
  br label %for.cond.cleanup

for.cond.cleanup:
  %.lcssa = phi float [ %4, %for.cond.for.cond.cleanup_crit_edge ], [ %agg.result.promoted, %entry ]
  store float %.lcssa, ptr %agg.result, align 4
  ret void

for.body:
  %0 = phi float [ %b10.promoted, %for.body.lr.ph ], [ %7, %for.body ]
  %i.019 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %1 = phi float [ %agg.result.promoted, %for.body.lr.ph ], [ %4, %for.body ]
  %arrayidx = getelementptr inbounds nuw %struct.TwoBytes, ptr %x, i32 %i.019
  %2 = load i8, ptr %arrayidx, align 1
  %conv = sitofp i8 %2 to float
  %arrayidx1 = getelementptr inbounds nuw %struct.TwoBytes, ptr %y, i32 %i.019
  %3 = load i8, ptr %arrayidx1, align 1
  %conv3 = sitofp i8 %3 to float
  %4 = tail call float @llvm.fmuladd.f32(float %conv, float %conv3, float %1)
  %b = getelementptr inbounds nuw i8, ptr %arrayidx, i32 1
  %5 = load i8, ptr %b, align 1
  %conv6 = sitofp i8 %5 to float
  %b8 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 1
  %6 = load i8, ptr %b8, align 1
  %conv9 = sitofp i8 %6 to float
  %7 = tail call float @llvm.fmuladd.f32(float %conv6, float %conv9, float %0)
  %inc = add nuw i32 %i.019, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}

; CHECK-LABEL: mac_2d_f32_i8:
; CHECK-NOT: v128.load
define hidden void @mac_2d_f32_i8(ptr dead_on_unwind noalias writable sret(%struct.TwoFloats) align 4 captures(none) %agg.result, ptr noundef readonly captures(none) %x, ptr noundef readonly captures(none) %y, i32 noundef %n) {
entry:
  %agg.result.promoted = load float, ptr %agg.result, align 4
  %cmp18.not = icmp eq i32 %n, 0
  br i1 %cmp18.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:
  %b10 = getelementptr inbounds nuw i8, ptr %agg.result, i32 4
  %b10.promoted = load float, ptr %b10, align 4
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  store float %7, ptr %b10, align 4
  br label %for.cond.cleanup

for.cond.cleanup:
  %.lcssa = phi float [ %4, %for.cond.for.cond.cleanup_crit_edge ], [ %agg.result.promoted, %entry ]
  store float %.lcssa, ptr %agg.result, align 4
  ret void

for.body:
  %0 = phi float [ %b10.promoted, %for.body.lr.ph ], [ %7, %for.body ]
  %i.019 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %1 = phi float [ %agg.result.promoted, %for.body.lr.ph ], [ %4, %for.body ]
  %arrayidx = getelementptr inbounds nuw %struct.TwoBytes, ptr %x, i32 %i.019
  %2 = load i8, ptr %arrayidx, align 1
  %conv = sitofp i8 %2 to float
  %arrayidx1 = getelementptr inbounds nuw %struct.TwoBytes, ptr %y, i32 %i.019
  %3 = load i8, ptr %arrayidx1, align 1
  %conv3 = sitofp i8 %3 to float
  %fmul = fmul float %conv, %conv3
  %4 = fadd float %fmul, %1
  %b = getelementptr inbounds nuw i8, ptr %arrayidx, i32 1
  %5 = load i8, ptr %b, align 1
  %conv6 = sitofp i8 %5 to float
  %b8 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 1
  %6 = load i8, ptr %b8, align 1
  %conv9 = sitofp i8 %6 to float
  %fmul.1 = fmul float %conv6, %conv9
  %7 = fadd float %fmul.1, %0
  %inc = add nuw i32 %i.019, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}

declare float @llvm.fmuladd.f32(float, float, float)

; CHECK-LABEL: mac_4d_f32_i8_fmuladd:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: f32x4.convert_i32x4_s
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: f32x4.convert_i32x4_s
; CHECK: f32x4.mul
; CHECK: f32x4.add
define hidden void @mac_4d_f32_i8_fmuladd(ptr dead_on_unwind noalias writable sret(%struct.FourFloats) align 4 captures(none) %agg.result, ptr noundef readonly captures(none) %x, ptr noundef readonly captures(none) %y, i32 noundef %n) {
entry:
  %agg.result.promoted = load float, ptr %agg.result, align 4
  %cmp38.not = icmp eq i32 %n, 0
  br i1 %cmp38.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:
  %b10 = getelementptr inbounds nuw i8, ptr %agg.result, i32 4
  %c16 = getelementptr inbounds nuw i8, ptr %agg.result, i32 8
  %d22 = getelementptr inbounds nuw i8, ptr %agg.result, i32 12
  %b10.promoted = load float, ptr %b10, align 4
  %c16.promoted = load float, ptr %c16, align 4
  %d22.promoted = load float, ptr %d22, align 4
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  store float %9, ptr %b10, align 4
  store float %12, ptr %c16, align 4
  store float %15, ptr %d22, align 4
  br label %for.cond.cleanup

for.cond.cleanup:
  %.lcssa = phi float [ %6, %for.cond.for.cond.cleanup_crit_edge ], [ %agg.result.promoted, %entry ]
  store float %.lcssa, ptr %agg.result, align 4
  ret void

for.body:
  %0 = phi float [ %d22.promoted, %for.body.lr.ph ], [ %15, %for.body ]
  %1 = phi float [ %c16.promoted, %for.body.lr.ph ], [ %12, %for.body ]
  %2 = phi float [ %b10.promoted, %for.body.lr.ph ], [ %9, %for.body ]
  %i.039 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %3 = phi float [ %agg.result.promoted, %for.body.lr.ph ], [ %6, %for.body ]
  %arrayidx = getelementptr inbounds nuw %struct.FourBytes, ptr %x, i32 %i.039
  %4 = load i8, ptr %arrayidx, align 1
  %conv = sitofp i8 %4 to float
  %arrayidx1 = getelementptr inbounds nuw %struct.FourBytes, ptr %y, i32 %i.039
  %5 = load i8, ptr %arrayidx1, align 1
  %conv3 = sitofp i8 %5 to float
  %6 = tail call float @llvm.fmuladd.f32(float %conv, float %conv3, float %3)
  %b = getelementptr inbounds nuw i8, ptr %arrayidx, i32 1
  %7 = load i8, ptr %b, align 1
  %conv6 = sitofp i8 %7 to float
  %b8 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 1
  %8 = load i8, ptr %b8, align 1
  %conv9 = sitofp i8 %8 to float
  %9 = tail call float @llvm.fmuladd.f32(float %conv6, float %conv9, float %2)
  %c = getelementptr inbounds nuw i8, ptr %arrayidx, i32 2
  %10 = load i8, ptr %c, align 1
  %conv12 = sitofp i8 %10 to float
  %c14 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 2
  %11 = load i8, ptr %c14, align 1
  %conv15 = sitofp i8 %11 to float
  %12 = tail call float @llvm.fmuladd.f32(float %conv12, float %conv15, float %1)
  %d = getelementptr inbounds nuw i8, ptr %arrayidx, i32 3
  %13 = load i8, ptr %d, align 1
  %conv18 = sitofp i8 %13 to float
  %d20 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 3
  %14 = load i8, ptr %d20, align 1
  %conv21 = sitofp i8 %14 to float
  %15 = tail call float @llvm.fmuladd.f32(float %conv18, float %conv21, float %0)
  %inc = add nuw i32 %i.039, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}

; CHECK-LABEL: mac_4d_f32_i8:
; CHECK: loop
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: f32x4.convert_i32x4_s
; CHECK: v128.load32_zero
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extend_low_i16x8_s
; CHECK: f32x4.convert_i32x4_s
; CHECK: f32x4.mul
; CHECK: f32x4.add
define hidden void @mac_4d_f32_i8(ptr dead_on_unwind noalias writable sret(%struct.FourFloats) align 4 captures(none) %agg.result, ptr noundef readonly captures(none) %x, ptr noundef readonly captures(none) %y, i32 noundef %n) {
entry:
  %agg.result.promoted = load float, ptr %agg.result, align 4
  %cmp38.not = icmp eq i32 %n, 0
  br i1 %cmp38.not, label %for.cond.cleanup, label %for.body.lr.ph

for.body.lr.ph:
  %b10 = getelementptr inbounds nuw i8, ptr %agg.result, i32 4
  %c16 = getelementptr inbounds nuw i8, ptr %agg.result, i32 8
  %d22 = getelementptr inbounds nuw i8, ptr %agg.result, i32 12
  %b10.promoted = load float, ptr %b10, align 4
  %c16.promoted = load float, ptr %c16, align 4
  %d22.promoted = load float, ptr %d22, align 4
  br label %for.body

for.cond.for.cond.cleanup_crit_edge:
  store float %9, ptr %b10, align 4
  store float %12, ptr %c16, align 4
  store float %15, ptr %d22, align 4
  br label %for.cond.cleanup

for.cond.cleanup:
  %.lcssa = phi float [ %6, %for.cond.for.cond.cleanup_crit_edge ], [ %agg.result.promoted, %entry ]
  store float %.lcssa, ptr %agg.result, align 4
  ret void

for.body:
  %0 = phi float [ %d22.promoted, %for.body.lr.ph ], [ %15, %for.body ]
  %1 = phi float [ %c16.promoted, %for.body.lr.ph ], [ %12, %for.body ]
  %2 = phi float [ %b10.promoted, %for.body.lr.ph ], [ %9, %for.body ]
  %i.039 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %3 = phi float [ %agg.result.promoted, %for.body.lr.ph ], [ %6, %for.body ]
  %arrayidx = getelementptr inbounds nuw %struct.FourBytes, ptr %x, i32 %i.039
  %4 = load i8, ptr %arrayidx, align 1
  %conv = sitofp i8 %4 to float
  %arrayidx1 = getelementptr inbounds nuw %struct.FourBytes, ptr %y, i32 %i.039
  %5 = load i8, ptr %arrayidx1, align 1
  %conv3 = sitofp i8 %5 to float
  %fmul = fmul float %conv, %conv3
  %6 = fadd float %fmul, %3
  %b = getelementptr inbounds nuw i8, ptr %arrayidx, i32 1
  %7 = load i8, ptr %b, align 1
  %conv6 = sitofp i8 %7 to float
  %b8 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 1
  %8 = load i8, ptr %b8, align 1
  %conv9 = sitofp i8 %8 to float
  %fmul.1 = fmul float %conv6, %conv9
  %9 = fadd float %fmul.1, %2
  %c = getelementptr inbounds nuw i8, ptr %arrayidx, i32 2
  %10 = load i8, ptr %c, align 1
  %conv12 = sitofp i8 %10 to float
  %c14 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 2
  %11 = load i8, ptr %c14, align 1
  %conv15 = sitofp i8 %11 to float
  %fmul.2 = fmul float %conv12, %conv15
  %12 = fadd float %fmul.2, %1
  %d = getelementptr inbounds nuw i8, ptr %arrayidx, i32 3
  %13 = load i8, ptr %d, align 1
  %conv18 = sitofp i8 %13 to float
  %d20 = getelementptr inbounds nuw i8, ptr %arrayidx1, i32 3
  %14 = load i8, ptr %d20, align 1
  %conv21 = sitofp i8 %14 to float
  %fmul.3 = fmul float %conv18, %conv21
  %15 = fadd float %fmul.3, %0
  %inc = add nuw i32 %i.039, 1
  %exitcond.not = icmp eq i32 %inc, %n
  br i1 %exitcond.not, label %for.cond.for.cond.cleanup_crit_edge, label %for.body
}
