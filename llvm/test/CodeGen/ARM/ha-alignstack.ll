; RUN: llc --mtriple armv7-eabihf %s -o - | FileCheck %s

%struct.S0 = type { [4 x float] }
%struct.S1 = type { [2 x float] }
%struct.S2 = type { [4 x float] }
%struct.D0 = type { [2 x double] }
%struct.D1 = type { [2 x double] }
%struct.D2 = type { [4 x double] }

; pass in registers
define dso_local float @f0_0(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, %struct.S0 %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S0 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f0_0:
; CHECK:       vmov.f32 s0, s12
; CHECK-NEXT:  bx       lr

; pass in memory, no memory/regs split
define dso_local float @f0_1(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, float %x, %struct.S0 %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S0 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f0_1:
; CHECK:       vldr s0, [sp]
; CHECK-NEXT:  bx   lr

; pass in memory, alignment 4
define dso_local float @f0_2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %x, %struct.S0 %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S0 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f0_2:
; CHECK:       vldr s0, [sp, #4]
; CHECK-NEXT:  bx   lr

; pass in registers
define dso_local float @f1_0(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, %struct.S1 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S1 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f1_0:
; CHECK:       vmov.f32 s0, s14
; CHECK-NEXT:  bx       lr

; pass in memory, no memory/regs split
define dso_local float @f1_1(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, float %x, %struct.S1 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S1 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f1_1:
; CHECK:       vldr s0, [sp]
; CHECK-NEXT:  bx   lr

; pass in memory, alignment 8
define dso_local float @f1_2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %x, %struct.S1 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S1 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f1_2:
; CHECK:       vldr s0, [sp, #8]
; CHECK-NEXT:  bx   lr

; pass in registers
define dso_local float @f2_0(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, %struct.S2 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S2 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f2_0:
; CHECK:       vmov.f32 s0, s12
; CHECK-NEXT:  bx       lr

; pass in memory, no memory/regs split
define dso_local float @f2_1(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, float %x, %struct.S2 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S2 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f2_1:
; CHECK:       vldr s0, [sp]
; CHECK-NEXT:  bx   lr

; pass in memory, alignment 8
define dso_local float @f2_2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %x, %struct.S2 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.S2 %s.coerce, 0, 0
  ret float %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: f2_2:
; CHECK:       vldr s0, [sp, #8]
; CHECK-NEXT:  bx   lr

; pass in registers
define dso_local double @g0_0(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, %struct.D0 %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D0 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g0_0:
; CHECK:       vmov.f64 d0, d6
; CHECK-NEXT:  bx       lr

; pass in memory, no memory/regs split
define dso_local double @g0_1(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, %struct.D0 %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D0 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g0_1:
; CHECK:       vldr d0, [sp]
; CHECK-NEXT:  bx   lr

; pass in memory, alignment 8
define dso_local double @g0_2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %x, %struct.D0 %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D0 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g0_2:
; CHECK:       vldr d0, [sp, #8]
; CHECK-NEXT:  bx   lr

; pass in registers
define dso_local double @g1_0(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, %struct.D1 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D1 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g1_0:
; CHECK:       vmov.f64 d0, d6
; CHECK-NEXT:  bx       lr

; pass in memory, no memory/regs split
define dso_local double @g1_1(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, %struct.D1 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D1 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g1_1:
; CHECK:       vldr d0, [sp]
; CHECK-NEXT:  bx   lr

; pass in memory, alignment 8
define dso_local double @g1_2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %x, %struct.D1 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D1 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g1_2:
; CHECK:       vldr d0, [sp, #8]
; CHECK-NEXT:  bx   lr

; pass in registers
define dso_local double @g2_0(double %d0, double %d1, double %d2, double %d3, %struct.D2 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D2 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g2_0:
; CHECK:       vmov.f64 d0, d4
; CHECK-NEXT:  bx       lr

; pass in memory, no memory/regs split
define dso_local double @g2_1(double %d0, double %d1, double %d2, double %d3, double %d4, %struct.D2 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D2 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g2_1:
; CHECK:       vldr d0, [sp]
; CHECK-NEXT:  bx   lr

; pass in memory, alignment 8
define dso_local double @g2_2(double %d0, double %d1, double %d2, double %d3, double %d4, double %d5, double %d6, double %d7, float %x, %struct.D2 alignstack(8) %s.coerce) local_unnamed_addr #0 {
entry:
  %s.coerce.fca.0.0.extract = extractvalue %struct.D2 %s.coerce, 0, 0
  ret double %s.coerce.fca.0.0.extract
}
; CHECK-LABEL: g2_2:
; CHECK:       vldr d0, [sp, #8]
; CHECK-NEXT:  bx   lr

attributes #0 = { "target-cpu"="generic" "target-features"="+armv7-a,+d32,+dsp,+fp64,+neon,+strict-align,+vfp2,+vfp2sp,+vfp3,+vfp3d16,+vfp3d16sp,+vfp3sp,-thumb-mode" }
