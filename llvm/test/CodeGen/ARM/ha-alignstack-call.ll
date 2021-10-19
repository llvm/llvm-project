; RUN: llc --mtriple=armv7-eabihf %s -o - | FileCheck %s --enable-var-scope 

%struct.S0 = type { [4 x float] }
%struct.S1 = type { [2 x float] }
%struct.S2 = type { [4 x float] }
%struct.D0 = type { [2 x double] }
%struct.D1 = type { [2 x double] }
%struct.D2 = type { [4 x double] }

; pass in regs
declare dso_local float @f0_0(double, double, double, double, double, double, %struct.S0) local_unnamed_addr #0
define dso_local float @f0_0_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f0_0(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, %struct.S0 { [4 x float] [float 0x3FE3333340000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f0_0_call:
; CHECK:       vldr s12, .L[[L:.*]]
; CHECK:       b    f0_0
; CHECK:      .L[[L]]:
; CHECK-NEXT: .long 0x3f19999a

; pass in memory, no split
declare dso_local float @f0_1(double, double, double, double, double, double, float, %struct.S0) local_unnamed_addr #0
define dso_local float @f0_1_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f0_1(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, float 0x3FE3333340000000, %struct.S0 { [4 x float] [float 0x3FE6666660000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f0_1_call:
; CHECK:       movw r1, #13107
; CHECK:       mov  r0, #0
; CHECK:       movt r1, #16179
; CHECK-DAG:   str  r1, [sp]
; CHECK-DAG:   str  r0, [sp, #4]
; CHECK-DAG:   str  r0, [sp, #8]
; CHECK-DAG:   str  r0, [sp, #12]
; CHECK:       bl   f0_1

; pass memory, alignment 4
declare dso_local float @f0_2(double, double, double, double, double, double, double, double, float, %struct.S0) local_unnamed_addr #0
define dso_local float @f0_2_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f0_2(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 0x3FE6666666666666, float 0x3FE99999A0000000, %struct.S0 { [4 x float] [float 0x3FECCCCCC0000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f0_2_call:
; CHECK:       movw r1, #26214
; CHECK:       movw r2, #52429
; CHECK:       mov  r0, #0
; CHECK:       movt r1, #16230
; CHECK:       movt r2, #16204
; CHECK-DAG:   str  r2, [sp]
; CHECK-DAG:   str  r1, [sp, #4]
; CHECK-DAG:   str  r0, [sp, #8]
; CHECK-DAG:   str  r0, [sp, #12]
; CHECK-DAG:   str  r0, [sp, #16]
; CHECK:       bl       f0_2

; pass in regs
declare dso_local float @f1_0(double, double, double, double, double, double, double, %struct.S1 alignstack(8)) local_unnamed_addr #0
define dso_local float @f1_0_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f1_0(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, %struct.S1 alignstack(8) { [2 x float] [float 0x3FE6666660000000, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f1_0_call:
; CHECK-DAG:   vldr s14, .L[[L0:.*]]
; CHECK-DAG:   vldr s15, .L[[L1:.*]]
; CHECK:       b    f1_0
; CHECK:       .L[[L0]]:
; CHECK-NEXT:  .long 0x3f333333
; CHECK:       .L[[L1:.*]]:
; CHECK-NEXT:  .long 0x00000000

; pass in memory, no split
declare dso_local float @f1_1(double, double, double, double, double, double, double, float, %struct.S1 alignstack(8)) local_unnamed_addr #0
define dso_local float @f1_1_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f1_1(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, float 0x3FE6666660000000, %struct.S1 alignstack(8) { [2 x float] [float 0x3FE99999A0000000, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f1_1_call:
; CHECK:       movw r1, #52429
; CHECK:       mov  r0, #0
; CHECK:       movt r1, #16204
; CHECK-DAG:   str  r1, [sp]
; CHECK-DAG:   str  r0, [sp, #4]
; CHECK:       bl   f1_1

; pass in memory, alignment 8
declare dso_local float @f1_2(double, double, double, double, double, double, double, double, float, %struct.S1 alignstack(8)) local_unnamed_addr #0
define dso_local float @f1_2_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f1_2(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 0x3FE6666666666666, float 0x3FE99999A0000000, %struct.S1 alignstack(8) { [2 x float] [float 0x3FECCCCCC0000000, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f1_2_call:
; CHECK-DAG:   mov   r0, #0
; CHECK-DAG:   movw  r1, #26214
; CHECK:       str   r0, [sp, #12]
; CHECK:       movw  r0, #52429
; CHECK:       movt  r1, #16230
; CHECK:       movt  r0, #16204
; CHECK-DAG:   str   r1, [sp, #8]
; CHECK-DAG:   str   r0, [sp]
; CHECK:       bl    f1_2


; pass in registers
declare dso_local float @f2_0(double, double, double, double, double, double, %struct.S2 alignstack(8)) local_unnamed_addr #0
define dso_local float @f2_0_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f2_0(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, %struct.S2 alignstack(8) { [4 x float] [float 0x3FE3333340000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f2_0_call:
; CHECK-DAG:   vldr     s12, .L[[L0:.*]]
; CHECK-DAG:   vldr     s13, .L[[L1:.*]]
; CHECK-DAG:   vmov.f32 s14, s13
; CHECK-DAG:   vmov.f32 s15, s13
; CHECK:       b        f2_0
; CHECK:       .L[[L0]]:
; CHECK-NEXT:  .long 0x3f19999a
; CHECK:       .L[[L1]]:
; CHECK-NEXT:  .long 0x00000000

; pass in memory, no split
declare dso_local float @f2_1(double, double, double, double, double, double, float, %struct.S2 alignstack(8)) local_unnamed_addr #0
define dso_local float @f2_1_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f2_1(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, float 0x3FE3333340000000, %struct.S2 alignstack(8) { [4 x float] [float 0x3FE6666660000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f2_1_call:
; CHECK:       movw  r1, #13107
; CHECK:       mov   r0, #0
; CHECK:       movt  r1, #16179
; CHECK:       str   r1, [sp]
; CHECK:       str   r0, [sp, #4]
; CHECK:       vldr  s12, .L[[L:.*]]
; CHECK:       str   r0, [sp, #8]
; CHECK:       str   r0, [sp, #12]
; CHECK:       bl    f2_1
; CHECK:       .L[[L]]:
; CHECK-NEXT:  .long    0x3f19999a

; pass in memory, alignment 8
declare dso_local float @f2_2(double, double, double, double, double, double, double, double, float, %struct.S2 alignstack(8)) local_unnamed_addr #0
define dso_local float @f2_2_call() local_unnamed_addr #0 {
entry:
  %call = tail call float @f2_2(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 0x3FE6666666666666, float 0x3FE99999A0000000, %struct.S2 alignstack(8) { [4 x float] [float 0x3FECCCCCC0000000, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00] }) #0
  ret float %call
}
; CHECK-LABEL: f2_2_call:
; CHECK:       mov  r0, #0
; CHECK:       movw r1, #26214
; CHECK:       str  r0, [sp, #12]
; CHECK:       str  r0, [sp, #16]
; CHECK:       movt r1, #16230
; CHECK:       str  r0, [sp, #20]
; CHECK:       movw r0, #52429
; CHECK:       movt r0, #16204
; CHECK:       str  r1, [sp, #8]
; CHECK:       str  r0, [sp]
; CHECK:       bl   f2_2

; pass in registers
declare dso_local double @g0_0(double, double, double, double, double, double, %struct.D0) local_unnamed_addr #0
define dso_local double @g0_0_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g0_0(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, %struct.D0 { [2 x double] [double 6.000000e-01, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g0_0_call:
; CHECK:       vldr    d6, .L[[L:.*]]
; CHECK:       b    g0_0
; CHECK:       .L[[L]]
; CHECK-NEXT:  long 858993459
; CHECK-NEXT:  long 1071854387

; pass in memory, no split
declare dso_local double @g0_1(double, double, double, double, double, double, double, %struct.D0) local_unnamed_addr #0
define dso_local double @g0_1_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g0_1(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, %struct.D0 { [2 x double] [double 0x3FE6666666666666, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g0_1_call:
; CHECK:       movw  r0, #26214
; CHECK:       movw  r1, #26214
; CHECK:       mov   r2, #0
; CHECK:       movt  r0, #16358
; CHECK:       movt  r1, #26214
; CHECK:       str   r1, [sp]
; CHECK:       stmib sp, {r0, r2}
; CHECK:       str   r2, [sp, #12]
; CHECK:       bl    g0_1

; pass in memory, alignment 8
declare dso_local double @g0_2(double, double, double, double, double, double, double, double, float, %struct.D0) local_unnamed_addr #0
define dso_local double @g0_2_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g0_2(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 0x3FE6666666666666, float 0x3FE99999A0000000, %struct.D0 { [2 x double] [double 9.000000e-01, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g0_2_call:
; CHECK:       movw r0, #52428
; CHECK:       movt r0, #16364
; CHECK:       movw r1, #52429
; CHECK:       str  r0, [sp, #12]
; CHECK:       movw r0, #52429
; CHECK:       mov  r2, #0
; CHECK:       movt r1, #52428
; CHECK:       movt r0, #16204
; CHECK:       str  r1, [sp, #8]
; CHECK:       str  r2, [sp, #16]
; CHECK:       str  r2, [sp, #20]
; CHECK:       str  r0, [sp]
; CHECK:       bl   g0_2

; pass in registers
declare dso_local double @g1_0(double, double, double, double, double, double, %struct.D1 alignstack(8)) local_unnamed_addr #0
define dso_local double @g1_0_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g1_0(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, %struct.D1 alignstack(8) { [2 x double] [double 6.000000e-01, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g1_0_call:
; CHECK-DAG:   vmov.i32 d7, #0x0
; CHECK-DAG:   vldr     d6, .L[[L:.*]]
; CHECK:       b        g1_0
; CHECK:      .L[[L]]:
; CHECK-NEXT: .long    858993459
; CHECK-NEXT: .long    107185438

; pass in memory, no split
declare dso_local double @g1_1(double, double, double, double, double, double, double, %struct.D1 alignstack(8)) local_unnamed_addr #0
define dso_local double @g1_1_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g1_1(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, %struct.D1 alignstack(8) { [2 x double] [double 0x3FE6666666666666, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g1_1_call:
; CHECK:       movw  r0, #26214
; CHECK:       movw  r1, #26214
; CHECK:       mov   r2, #0
; CHECK:       movt  r0, #16358
; CHECK:       movt  r1, #26214
; CHECK:       str   r1, [sp]
; CHECK:       stmib sp, {r0, r2}
; CHECK:       str   r2, [sp, #12]
; CHECK:       bl    g1_1

; pass in memory, alignment 8
declare dso_local double @g1_2(double, double, double, double, double, double, double, double, float, %struct.D1 alignstack(8)) local_unnamed_addr #0
define dso_local double @g1_2_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g1_2(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 0x3FE6666666666666, float 0x3FE99999A0000000, %struct.D1 alignstack(8) { [2 x double] [double 9.000000e-01, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g1_2_call:
; CHECK:       movw r0, #52428
; CHECK:       movt r0, #16364
; CHECK:       movw r1, #52429
; CHECK:       str  r0, [sp, #12]
; CHECK:       movw r0, #52429
; CHECK:       mov  r2, #0
; CHECK:       movt r1, #52428
; CHECK:       movt r0, #16204
; CHECK:       str  r1, [sp, #8]
; CHECK:       str  r2, [sp, #16]
; CHECK:       str  r2, [sp, #20]
; CHECK:       str  r0, [sp]
; CHECK:       bl   g1_2

; pass in registers
declare dso_local double @g2_0(double, double, double, double, %struct.D2 alignstack(8)) local_unnamed_addr #0
define dso_local double @g2_0_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g2_0(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, %struct.D2 alignstack(8) { [4 x double] [double 4.000000e-01, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g2_0_call:
; CHECK-DAG:   vldr     d4, .L[[L:.*]]
; CHECK-DAG:   vmov.i32 d5, #0x0
; CHECK-DAG:   vmov.i32 d6, #0x0
; CHECK-DAG:   vmov.i32 d7, #0x0
; CHECK:       b        g2_0
; CHECK:       .L[[L]]:
; CHECK-NEXT:  .long    2576980378
; CHECK-NEXT:  .long    1071225241

; pass in memory, no split
; [sp] [sp + 4] =  0x00000000 0x3fe00000 = .5
; [sp + 8] [sp + 12] = 0 0 = .0
; ...
declare dso_local double @g2_1(double, double, double, double, double, %struct.D2 alignstack(8)) local_unnamed_addr #0
define dso_local double @g2_1_call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g2_1(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, %struct.D2 alignstack(8) { [4 x double] [double 5.000000e-01, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g2_1_call:
; CHECK:       movw   r0, #0
; CHECK:       mov    r1, #0
; CHECK:       movt   r0, #16352
; CHECK:       str    r1, [sp]
; CHECK:       stmib  sp, {r0, r1}
; CHECK:       str    r1, [sp, #12]
; CHECK:       str    r1, [sp, #16]
; CHECK:       str    r1, [sp, #20]
; CHECK:       str    r1, [sp, #24]
; CHECK:       str    r1, [sp, #28]
; CHECK:       bl    g2_1

; pass in memory, alignment 8
declare dso_local double @g2_2(double, double, double, double, double, double, double, double, float, %struct.D2 alignstack(8)) local_unnamed_addr #0
define dso_local double @g2_2call() local_unnamed_addr #0 {
entry:
  %call = tail call double @g2_2(double 0.000000e+00, double 1.000000e-01, double 2.000000e-01, double 3.000000e-01, double 4.000000e-01, double 5.000000e-01, double 6.000000e-01, double 0x3FE6666666666666, float 0x3FE99999A0000000, %struct.D2 alignstack(8) { [4 x double] [double 9.000000e-01, double 0.000000e+00, double 0.000000e+00, double 0.000000e+00] }) #0
  ret double %call
}
; CHECK-LABEL: g2_2call:
; CHECK:       movw r0, #52428
; CHECK:       movt r0, #16364
; CHECK:       movw r1, #52429
; CHECK:       str  r0, [sp, #12]
; CHECK:       movw r0, #52429
; CHECK:       mov  r2, #0
; CHECK:       movt r1, #52428
; CHECK:       movt r0, #16204
; CHECK:       str  r1, [sp, #8]
; CHECK:       str  r2, [sp, #16]
; CHECK:       str  r2, [sp, #20]
; CHECK:       str  r2, [sp, #24]
; CHECK:       str  r2, [sp, #28]
; CHECK:       str  r2, [sp, #32]
; CHECK:       str  r2, [sp, #36]
; CHECK:       str  r0, [sp]
; CHECK:       bl   g2_2

attributes #0 = { nounwind "target-cpu"="generic" "target-features"="+armv7-a,+d32,+dsp,+fp64,+neon,+strict-align,+vfp2,+vfp2sp,+vfp3,+vfp3d16,+vfp3d16sp,+vfp3sp,-thumb-mode" }
