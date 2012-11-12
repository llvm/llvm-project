; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s
; RUN: llc < %s -march=arm -mattr=+vfp2 | FileCheck %s --check-prefix=CHECK-VFP
; RUN: llc < %s -mattr=+neon,+thumb2 -mtriple=thumbv7-apple-darwin | FileCheck %s --check-prefix=CHECK-NEON

define i32 @f1(i32 %a.s) {
;CHECK: f1:
;CHECK: moveq
entry:
    %tmp = icmp eq i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f2(i32 %a.s) {
;CHECK: f2:
;CHECK: movgt
entry:
    %tmp = icmp sgt i32 %a.s, 4
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f3(i32 %a.s, i32 %b.s) {
;CHECK: f3:
;CHECK: movlt
entry:
    %tmp = icmp slt i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f4(i32 %a.s, i32 %b.s) {
;CHECK: f4:
;CHECK: movle
entry:
    %tmp = icmp sle i32 %a.s, %b.s
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f5(i32 %a.u, i32 %b.u) {
;CHECK: f5:
;CHECK: movls
entry:
    %tmp = icmp ule i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define i32 @f6(i32 %a.u, i32 %b.u) {
;CHECK: f6:
;CHECK: movhi
entry:
    %tmp = icmp ugt i32 %a.u, %b.u
    %tmp1.s = select i1 %tmp, i32 2, i32 3
    ret i32 %tmp1.s
}

define double @f7(double %a, double %b) {
;CHECK: f7:
;CHECK: movlt
;CHECK: movlt
;CHECK-VFP: f7:
;CHECK-VFP: vmovmi
    %tmp = fcmp olt double %a, 1.234e+00
    %tmp1 = select i1 %tmp, double -1.000e+00, double %b
    ret double %tmp1
}

; <rdar://problem/7260094>
;
; We used to generate really horrible code for this function. The main cause was
; a lack of a custom lowering routine for an ISD::SELECT. This would result in
; two "it" blocks in the code: one for the "icmp" and another to move the index
; into the constant pool based on the value of the "icmp". If we have one "it"
; block generated, odds are good that we have close to the ideal code for this:
;
; CHECK-NEON:      _f8:
; CHECK-NEON:      movw    [[R3:r[0-9]+]], #1123
; CHECK-NEON:      adr     [[R2:r[0-9]+]], LCPI7_0
; CHECK-NEON-NEXT: cmp     r0, [[R3]]
; CHECK-NEON-NEXT: it      eq
; CHECK-NEON-NEXT: addeq{{.*}} [[R2]], #4
; CHECK-NEON-NEXT: ldr
; CHECK-NEON:      bx

define arm_apcscc float @f8(i32 %a) nounwind {
  %tmp = icmp eq i32 %a, 1123
  %tmp1 = select i1 %tmp, float 0x3FF3BE76C0000000, float 0x40030E9A20000000
  ret float %tmp1
}

; <rdar://problem/9049552>
; Glue values can only have a single use, but the following test exposed a
; case where a SELECT was lowered with 2 uses of a comparison, causing the
; scheduler to assert.
; CHECK-VFP: f9:

declare i8* @objc_msgSend(i8*, i8*, ...)
define void @f9() optsize {
entry:
  %cmp = icmp eq i8* undef, inttoptr (i32 4 to i8*)
  %conv191 = select i1 %cmp, float -3.000000e+00, float 0.000000e+00
  %conv195 = select i1 %cmp, double -1.000000e+00, double 0.000000e+00
  %add = fadd double %conv195, 1.100000e+01
  %conv196 = fptrunc double %add to float
  %add201 = fadd float undef, %conv191
  %tmp484 = bitcast float %conv196 to i32
  %tmp478 = bitcast float %add201 to i32
  %tmp490 = insertvalue [2 x i32] undef, i32 %tmp484, 0
  %tmp493 = insertvalue [2 x i32] %tmp490, i32 %tmp478, 1
  call void bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to void (i8*, i8*, [2 x i32], i32, float)*)(i8* undef, i8* undef, [2 x i32] %tmp493, i32 0, float 1.000000e+00) optsize
  ret void
}

; CHECK: f10
define float @f10(i32 %a, i32 %b) nounwind uwtable readnone ssp {
; CHECK-NOT: floatsisf
  %1 = icmp eq i32 %a, %b
  %2 = zext i1 %1 to i32
  %3 = sitofp i32 %2 to float
  ret float %3
}

; CHECK: f11
define float @f11(i32 %a, i32 %b) nounwind uwtable readnone ssp {
; CHECK-NOT: floatsisf
  %1 = icmp eq i32 %a, %b
  %2 = sitofp i1 %1 to float
  ret float %2
}

; CHECK: f12
define float @f12(i32 %a, i32 %b) nounwind uwtable readnone ssp {
; CHECK-NOT: floatunsisf
  %1 = icmp eq i32 %a, %b
  %2 = uitofp i1 %1 to float
  ret float %2
}

