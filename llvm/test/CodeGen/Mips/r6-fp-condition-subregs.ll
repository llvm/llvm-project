; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips -mcpu=mips32r6 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips -mcpu=mips32r6 -mattr=+micromips -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -stop-after=finalize-isel -verify-machineinstrs < %s | FileCheck %s --check-prefix=ISEL
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips -mcpu=mips32r6 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -mattr=+micromips -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips -mcpu=mips32r6 -mattr=+micromips -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs -filetype=obj < %s -o /dev/null
; RUN: llc -mtriple=mips64 -mcpu=mips64r6 -target-abi=n32 -verify-machineinstrs -filetype=obj < %s -o /dev/null

; Coalesce the low-word condition with the full FPR used by SEL.D, avoiding an
; mfc1/mtc1 round trip (llvm/llvm-project#172459).
define double @compare_select(double %a, double %b) {
; CHECK-LABEL: compare_select:
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: cmp.lt.d $f[[COND:[0-9]+]],
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: sel.d $f[[COND]],
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: .end compare_select
;
; ISEL-LABEL: name: compare_select
; ISEL: %[[CMP:[0-9]+]]:fgr64 = CMP_LT_D
; ISEL-NEXT: %[[CC:[0-9]+]]:gpr32 = COPY %[[CMP]].sub_lo
; ISEL: %[[UNDEF:[0-9]+]]:fgr64 = IMPLICIT_DEF
; ISEL: %[[EXT:[0-9]+]]:fgr64 = INSERT_SUBREG %[[UNDEF]], killed %[[CC]], %subreg.sub_lo
; ISEL: %{{[0-9]+}}:fgr64 = SEL_D %[[EXT]],
  %c = fcmp olt double %a, %b
  %r = select i1 %c, double %a, double %b
  ret double %r
}

; Sharing a predicate across SEL.D instructions requires copies that preserve
; the FPR width. This previously crashed in copyPhysReg (#223905).
define void @shared_condition(ptr %out, <2 x double> %a, <2 x double> %b, i1 %c) {
; CHECK-LABEL: shared_condition:
; CHECK: mtc1
; CHECK-COUNT-4: sel.d
  %selected = select i1 %c, <2 x double> %b, <2 x double> %a
  %cmp = fcmp olt <2 x double> %a, %selected
  %result = select <2 x i1> %cmp, <2 x double> %a, <2 x double> %selected
  store <2 x double> %result, ptr %out, align 16
  ret void
}

; A double compare can feed both a single-precision select and an integer use.
define float @double_condition_float_result(double %a, double %b,
                                             float %t, float %f, ptr %out) {
; CHECK-LABEL: double_condition_float_result:
; CHECK: cmp.lt.d
; CHECK-DAG: mfc1
; CHECK-DAG: sel.s
  %c = fcmp olt double %a, %b
  %tv = fadd float %t, 1.0
  %fv = fmul float %f, 2.0
  %r = select i1 %c, float %tv, float %fv
  store i1 %c, ptr %out
  ret float %r
}

; SEL.D can use bit zero of a single-precision comparison result.
define double @float_condition_double_result(float %a, float %b,
                                             double %t, double %f) {
; CHECK-LABEL: float_condition_double_result:
; CHECK: cmp.lt.s $f[[COND:[0-9]+]],
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: sel.d $f[[COND]],
  %c = fcmp olt float %a, %b
  %r = select i1 %c, double %t, double %f
  ret double %r
}

; Coalesce the copies between CMP.S and SEL.S without emitting register moves.
define float @single_compare_select(float %a, float %b) {
; CHECK-LABEL: single_compare_select:
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: cmp.lt.s $f[[COND:[0-9]+]],
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: sel.s $f[[COND]],
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: .end single_compare_select
;
; ISEL-LABEL: name: single_compare_select
; ISEL: %[[CMP:[0-9]+]]:fgr32 = CMP_LT_S
; ISEL-NEXT: %[[CC:[0-9]+]]:fgr32 = COPY killed %[[CMP]]
; ISEL-NEXT: %[[COND:[0-9]+]]:fgr32 = COPY killed %[[CC]]
; ISEL: %{{[0-9]+}}:fgr32 = SEL_S %[[COND]],
  %c = fcmp olt float %a, %b
  %r = select i1 %c, float %a, float %b
  ret float %r
}

; Both selects need the same FP predicate. Preserve it across the first tied
; select with an FPR copy, including when the second select has a different width.
define void @shared_fp_condition(ptr %outd, ptr %outf, double %a, double %b,
                                 float %x, float %y) {
; CHECK-LABEL: shared_fp_condition:
; CHECK: cmp.lt.d $f[[COND:[0-9]+]],
; CHECK-NOT: mfc1
; CHECK: mov.d $f[[COPY:[0-9]+]], $f[[COND]]
; CHECK-NOT: mfc1
; CHECK: sel.d $f[[COPY]],
; CHECK-NOT: mfc1
; CHECK: sel.s $f[[COND]],
; CHECK-NOT: mfc1
; CHECK: .end shared_fp_condition
  %c = fcmp olt double %a, %b
  %t = fadd float %x, 1.0
  %f = fmul float %y, 2.0
  %d = select i1 %c, double %a, double %b
  %s = select i1 %c, float %t, float %f
  store double %d, ptr %outd
  store float %s, ptr %outf
  ret void
}

; Conditions from different-width compares can also merge through an i32 PHI.
define double @phi_mixed_conditions(i1 %choose, double %a, double %b,
                                    float %x, float %y) {
; CHECK-LABEL: phi_mixed_conditions:
; CHECK: cmp.ult.s
; CHECK: cmp.le.d
; CHECK: sel.d
;
; ISEL-LABEL: name: phi_mixed_conditions
; ISEL: CMP_ULT_S
; ISEL: CMP_LE_D
; ISEL: %[[CC:[0-9]+]]:gpr32 = PHI
; ISEL: INSERT_SUBREG {{.*}}%[[CC]], %subreg.sub_lo
; ISEL: SEL_D
entry:
  br i1 %choose, label %single, label %double
single:
  %cs = fcmp ult float %x, %y
  br label %join
double:
  %cd = fcmp ole double %a, %b
  br label %join
join:
  %c = phi i1 [ %cs, %single ], [ %cd, %double ]
  %r = select i1 %c, double %a, double %b
  ret double %r
}

; An unordered comparison must select the true arm when either operand is NaN.
define float @unordered_select(float %a, float %b) {
; CHECK-LABEL: unordered_select:
; CHECK: cmp.un.s $f[[COND:[0-9]+]], $f[[A:[0-9]+]], $f[[B:[0-9]+]]
; CHECK-NOT: mfc1
; CHECK-NOT: mtc1
; CHECK: sel.s $f[[COND]], $f[[B]], $f[[A]]
  %c = fcmp uno float %a, %b
  %r = select i1 %c, float %a, float %b
  ret float %r
}

; An arbitrary integer's low bit cannot be replaced by a zero/nonzero test.
declare void @branch_true()
declare void @branch_false()

define void @branch_bit0(i32 %x) {
; CHECK-LABEL: branch_bit0:
; CHECK: andi{{(16)?}} $[[CC:[0-9]+]], ${{[0-9]+}}, 1
; CHECK: b{{eqz|nez}}{{c?}} $[[CC]],
  %c = trunc i32 %x to i1
  br i1 %c, label %t, label %f
t:
  call void @branch_true()
  ret void
f:
  call void @branch_false()
  ret void
}

; Share one compare between an FPR branch and an integer use. Only the latter
; needs to copy and normalize the FP mask to 0/1.
define void @branch_and_store_boolean(double %a, double %b, ptr %out) {
; CHECK-LABEL: branch_and_store_boolean:
; CHECK: cmp.eq.d $f[[COND:[0-9]+]],
; CHECK: mfc1 {{.*}}, $f[[COND]]
; CHECK: andi{{(16)?}} {{.*}}, 1
; CHECK-DAG: bc1eqz{{c?}} $f[[COND]],
; CHECK-DAG: sw
;
; ISEL-LABEL: name: branch_and_store_boolean
; ISEL: %[[CMP:[0-9]+]]:fgr64 = CMP_EQ_D
; ISEL-NEXT: %[[CC:[0-9]+]]:gpr32 = COPY %[[CMP]].sub_lo
; ISEL-NEXT: %[[BOOL:[0-9]+]]:gpr32 = ANDi %[[CC]], 1
; ISEL: SW killed %[[BOOL]],
; ISEL: %[[FPCOND:[0-9]+]]:fgr32 = COPY %[[CC]]
; ISEL-NEXT: BC1EQZ killed %[[FPCOND]],
  %c = fcmp oeq double %a, %b
  %b32 = zext i1 %c to i32
  store i32 %b32, ptr %out
  br i1 %c, label %t, label %f
t:
  call void @branch_true()
  ret void
f:
  call void @branch_false()
  ret void
}
