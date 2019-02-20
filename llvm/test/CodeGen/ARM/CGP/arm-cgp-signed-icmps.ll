; RUN: llc -mtriple=thumbv8m.main -mcpu=cortex-m33 -arm-disable-cgp=false -mattr=-use-misched %s -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
; RUN: llc -mtriple=thumbv7em %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP
; RUN: llc -mtriple=thumbv8 %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -arm-enable-scalar-dsp-imms=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP-IMM

; CHECK-COMMON-LABEL: eq_sgt
; CHECK-NODSP: add
; CHECK-NODSP: uxtb
; CHECK-NODSP: sxtb
; CHECK-NODSP: cmp
; CHECK-NODSP: sub
; CHECK-NODSP: sxtb
; CHECK-NODSP: cmp

; CHECK-DSP: uadd8
; CHECK-DSP: sub
; CHECK-DSP: cmp
; CHECK-DSP: sxtb
; CHECK-DSP: sxtb
; CHECK-DSP: cmp

; CHECK-DSP-IMM: uadd8 [[ADD:r[0-9]+]],
; CHECK-DSP-IMM: cmp [[ADD]],
; CHECK-DSP-IMM: subs [[SUB:r[0-9]+]],
; CHECK-DSP-IMM: sxtb [[SEXT0:r[0-9]+]], [[ADD]]
; CHECK-DSP-IMM: sxtb [[SEXT1:r[0-9]+]], [[SUB]]
; CHECK-DSP-IMM: cmp [[SEXT1]], [[SEXT0]]
define i8 @eq_sgt(i8* %x, i8 *%y, i8 zeroext %z) {
entry:
  %load0 = load i8, i8* %x, align 1
  %load1 = load i8, i8* %y, align 1
  %add = add i8 %load0, %z
  %sub = sub i8 %load1, 1
  %cmp = icmp eq i8 %add, 200
  %cmp1 = icmp sgt i8 %sub, %add
  %res0 = select i1 %cmp, i8 35, i8 47
  %res1 = select i1 %cmp1, i8 %res0, i8 %sub
  ret i8 %res1
}

; CHECK-COMMON-LABEL: ugt_slt
; CHECK-NODSP: sub
; CHECK-NODSP: sxth
; CHECK-NODSP: uxth
; CHECK-NODSP: add
; CHECK-NODSP: sxth
; CHECK-NODSP: cmp
; CHECK-NODSP: cmp

; CHECK-DSP: sub
; CHECK-DSP: sxth
; CHECK-DSP: add
; CHECK-DSP: uxth
; CHECK-DSP: sxth
; CHECK-DSP: cmp
; CHECK-DSP: cmp

; CHECK-DSP-IMM: sxth [[ARG:r[0-9]+]], r2
; CHECK-DSP-IMM: uadd16 [[ADD:r[0-9]+]],
; CHECK-DSP-IMM: sxth.w [[SEXT:r[0-9]+]], [[ADD]]
; CHECK-DSP-IMM: cmp [[SEXT]], [[ARG]]
; CHECK-DSP-IMM-NOT: uxt
; CHECK-DSP-IMM: movs [[ONE:r[0-9]+]], #1
; CHECK-DSP-IMM: usub16 [[SUB:r[0-9]+]], r1, [[ONE]]
; CHECK-DSP-IMM: cmp [[SUB]], r2
define i16 @ugt_slt(i16 *%x, i16 zeroext %y, i16 zeroext %z) {
entry:
  %load0 = load i16, i16* %x, align 1
  %add = add i16 %load0, %z
  %sub = sub i16 %y, 1
  %cmp = icmp slt i16 %add, %z
  %cmp1 = icmp ugt i16 %sub, %z
  %res0 = select i1 %cmp, i16 35, i16 -1
  %res1 = select i1 %cmp1, i16 %res0, i16 0
  ret i16 %res1
}

; CHECK-COMMON-LABEL: urem_trunc_icmps
; CHECK-COMMON-NOT: uxt
; CHECK-COMMON: sxtb [[SEXT:r[0-9]+]],
; CHECK-COMMON: cmp [[SEXT]], #7
define void @urem_trunc_icmps(i16** %in, i32* %g, i32* %k) {
entry:
  %ptr = load i16*, i16** %in, align 4
  %ld = load i16, i16* %ptr, align 2
  %cmp.i = icmp eq i16 %ld, 0
  br i1 %cmp.i, label %exit, label %cond.false.i

cond.false.i:
  %rem = urem i16 5, %ld
  %extract.t = trunc i16 %rem to i8
  br label %body

body:
  %cond.in.i.off0 = phi i8 [ %extract.t, %cond.false.i ], [ %add, %for.inc ]
  %cmp = icmp sgt i8 %cond.in.i.off0, 7
  %conv5 = zext i1 %cmp to i32
  store i32 %conv5, i32* %g, align 4
  %.pr = load i32, i32* %k, align 4
  %tobool13150 = icmp eq i32 %.pr, 0
  br i1 %tobool13150, label %for.inc, label %exit

for.inc:
  %add = add nuw i8 %cond.in.i.off0, 1
  br label %body

exit:
  ret void
}
