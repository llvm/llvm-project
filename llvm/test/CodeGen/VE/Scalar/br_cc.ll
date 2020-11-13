; RUN: llc < %s -mtriple=ve | FileCheck %s

; Function Attrs: nounwind
define void @br_cc_i1_var(i1 zeroext %0, i1 zeroext %1) {
; CHECK-LABEL: br_cc_i1_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    xor %s0, %s0, %s1
; CHECK-NEXT:    brne.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = xor i1 %0, %1
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i8_var(i8 signext %0, i8 signext %1) {
; CHECK-LABEL: br_cc_i8_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i8 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u8_var(i8 zeroext %0, i8 zeroext %1) {
; CHECK-LABEL: br_cc_u8_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i8 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i16_var(i16 signext %0, i16 signext %1) {
; CHECK-LABEL: br_cc_i16_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i16 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u16_var(i16 zeroext %0, i16 zeroext %1) {
; CHECK-LABEL: br_cc_u16_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i16 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i32_var(i32 signext %0, i32 signext %1) {
; CHECK-LABEL: br_cc_i32_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i32 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u32_var(i32 zeroext %0, i32 zeroext %1) {
; CHECK-LABEL: br_cc_u32_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i32 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i64_var(i64 %0, i64 %1) {
; CHECK-LABEL: br_cc_i64_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.l %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i64 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u64_var(i64 %0, i64 %1) {
; CHECK-LABEL: br_cc_u64_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.l %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i64 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i128_var(i128 %0, i128 %1) {
; CHECK-LABEL: br_cc_i128_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    brne.l 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i128 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u128_var(i128 %0, i128 %1) {
; CHECK-LABEL: br_cc_u128_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    xor %s1, %s1, %s3
; CHECK-NEXT:    xor %s0, %s0, %s2
; CHECK-NEXT:    or %s0, %s0, %s1
; CHECK-NEXT:    brne.l 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = icmp eq i128 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_float_var(float %0, float %1) {
; CHECK-LABEL: br_cc_float_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.s %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp fast oeq float %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_double_var(double %0, double %1) {
; CHECK-LABEL: br_cc_double_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.d %s0, %s1, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp fast oeq double %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_quad_var(fp128 %0, fp128 %1) {
; CHECK-LABEL: br_cc_quad_var:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    fcmp.q %s0, %s2, %s0
; CHECK-NEXT:    brne.d 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %3 = fcmp fast oeq fp128 %0, %1
  br i1 %3, label %4, label %5

4:                                                ; preds = %2
  tail call void asm sideeffect "nop", ""()
  br label %5

5:                                                ; preds = %4, %2
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i1_imm(i1 zeroext %0) {
; CHECK-LABEL: br_cc_i1_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brne.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  br i1 %0, label %3, label %2

2:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %3

3:                                                ; preds = %2, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i8_imm(i8 signext %0) {
; CHECK-LABEL: br_cc_i8_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brlt.w -10, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp slt i8 %0, -9
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u8_imm(i8 zeroext %0) {
; CHECK-LABEL: br_cc_u8_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.w %s0, 8, %s0
; CHECK-NEXT:    brgt.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ult i8 %0, 9
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i16_imm(i16 signext %0) {
; CHECK-LABEL: br_cc_i16_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brlt.w 62, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp slt i16 %0, 63
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u16_imm(i16 zeroext %0) {
; CHECK-LABEL: br_cc_u16_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.w %s0, 63, %s0
; CHECK-NEXT:    brgt.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ult i16 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i32_imm(i32 signext %0) {
; CHECK-LABEL: br_cc_i32_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brlt.w 63, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp slt i32 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u32_imm(i32 zeroext %0) {
; CHECK-LABEL: br_cc_u32_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.w %s0, 63, %s0
; CHECK-NEXT:    brgt.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ult i32 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i64_imm(i64 %0) {
; CHECK-LABEL: br_cc_i64_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brlt.l 63, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp slt i64 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u64_imm(i64 %0) {
; CHECK-LABEL: br_cc_u64_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.l %s0, 63, %s0
; CHECK-NEXT:    brgt.l 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ult i64 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_i128_imm(i128 %0) {
; CHECK-LABEL: br_cc_i128_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmps.l %s1, %s1, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.gt %s3, (63)0, %s1
; CHECK-NEXT:    cmpu.l %s0, %s0, (58)0
; CHECK-NEXT:    cmov.l.gt %s2, (63)0, %s0
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s1
; CHECK-NEXT:    brne.w 0, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp slt i128 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_u128_imm(i128 %0) {
; CHECK-LABEL: br_cc_u128_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    cmps.l %s1, %s1, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.ne %s3, (63)0, %s1
; CHECK-NEXT:    cmpu.l %s0, %s0, (58)0
; CHECK-NEXT:    cmov.l.gt %s2, (63)0, %s0
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s1
; CHECK-NEXT:    brne.w 0, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ult i128 %0, 64
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_float_imm(float %0) {
; CHECK-LABEL: br_cc_float_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brle.s 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fcmp fast olt float %0, 0.000000e+00
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_double_imm(double %0) {
; CHECK-LABEL: br_cc_double_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brle.d 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fcmp fast olt double %0, 0.000000e+00
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_quad_imm(fp128 %0) {
; CHECK-LABEL: br_cc_quad_imm:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fcmp.q %s0, %s4, %s0
; CHECK-NEXT:    brge.d 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fcmp fast olt fp128 %0, 0xL00000000000000000000000000000000
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_i1(i1 zeroext %0) {
; CHECK-LABEL: br_cc_imm_i1:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    breq.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  br i1 %0, label %2, label %3

2:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %3

3:                                                ; preds = %2, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_i8(i8 signext %0) {
; CHECK-LABEL: br_cc_imm_i8:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brgt.w -9, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp sgt i8 %0, -10
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_u8(i8 zeroext %0) {
; CHECK-LABEL: br_cc_imm_u8:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.w %s0, 9, %s0
; CHECK-NEXT:    brlt.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ugt i8 %0, 8
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_i16(i16 signext %0) {
; CHECK-LABEL: br_cc_imm_i16:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brgt.w 63, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp sgt i16 %0, 62
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_u16(i16 zeroext %0) {
; CHECK-LABEL: br_cc_imm_u16:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    lea %s1, 64
; CHECK-NEXT:    cmpu.w %s0, %s1, %s0
; CHECK-NEXT:    brlt.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ugt i16 %0, 63
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_i32(i32 signext %0) {
; CHECK-LABEL: br_cc_imm_i32:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brgt.w -64, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp sgt i32 %0, -65
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_u32(i32 zeroext %0) {
; CHECK-LABEL: br_cc_imm_u32:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.w %s0, -64, %s0
; CHECK-NEXT:    brlt.w 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ugt i32 %0, -65
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_i64(i64 %0) {
; CHECK-LABEL: br_cc_imm_i64:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brgt.l -64, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp sgt i64 %0, -65
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_u64(i64 %0) {
; CHECK-LABEL: br_cc_imm_u64:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmpu.l %s0, -64, %s0
; CHECK-NEXT:    brlt.l 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ugt i64 %0, -65
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_i128(i128 %0) {
; CHECK-LABEL: br_cc_imm_i128:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmps.l %s1, %s1, (0)0
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.lt %s3, (63)0, %s1
; CHECK-NEXT:    cmpu.l %s0, %s0, (58)1
; CHECK-NEXT:    cmov.l.lt %s2, (63)0, %s0
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s1
; CHECK-NEXT:    brne.w 0, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp sgt i128 %0, -65
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_u128(i128 %0) {
; CHECK-LABEL: br_cc_imm_u128:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    cmps.l %s1, %s1, (0)0
; CHECK-NEXT:    or %s2, 0, (0)1
; CHECK-NEXT:    or %s3, 0, (0)1
; CHECK-NEXT:    cmov.l.ne %s3, (63)0, %s1
; CHECK-NEXT:    cmpu.l %s0, %s0, (58)1
; CHECK-NEXT:    cmov.l.lt %s2, (63)0, %s0
; CHECK-NEXT:    cmov.l.eq %s3, %s2, %s1
; CHECK-NEXT:    brne.w 0, %s3, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = icmp ugt i128 %0, -65
  br i1 %2, label %3, label %4

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_float(float %0) {
; CHECK-LABEL: br_cc_imm_float:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brgt.s 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fcmp fast ult float %0, 0.000000e+00
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_double(double %0) {
; CHECK-LABEL: br_cc_imm_double:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    brgt.d 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fcmp fast ult double %0, 0.000000e+00
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}

; Function Attrs: nounwind
define void @br_cc_imm_quad(fp128 %0) {
; CHECK-LABEL: br_cc_imm_quad:
; CHECK:       .LBB{{[0-9]+}}_4:
; CHECK-NEXT:    lea %s2, .LCPI{{[0-9]+}}_0@lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, .LCPI{{[0-9]+}}_0@hi(, %s2)
; CHECK-NEXT:    ld %s4, 8(, %s2)
; CHECK-NEXT:    ld %s5, (, %s2)
; CHECK-NEXT:    fcmp.q %s0, %s4, %s0
; CHECK-NEXT:    brlt.d 0, %s0, .LBB{{[0-9]+}}_2
; CHECK-NEXT:  # %bb.1:
; CHECK-NEXT:    #APP
; CHECK-NEXT:    nop
; CHECK-NEXT:    #NO_APP
; CHECK-NEXT:  .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  %2 = fcmp fast ult fp128 %0, 0xL00000000000000000000000000000000
  br i1 %2, label %4, label %3

3:                                                ; preds = %1
  tail call void asm sideeffect "nop", ""()
  br label %4

4:                                                ; preds = %3, %1
  ret void
}
