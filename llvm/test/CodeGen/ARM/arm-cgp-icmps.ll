; RUN: llc -mtriple=thumbv8.main -mcpu=cortex-m33 %s -arm-disable-cgp=false -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
; RUN: llc -mtriple=thumbv7em %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP
; RUN: llc -mtriple=thumbv8 %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -arm-enable-scalar-dsp-imms=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP-IMM

; CHECK-COMMON-LABEL: test_ult_254_inc_imm:
; CHECK-DSP:        adds    r0, #1
; CHECK-DSP-NEXT:   uxtb    r1, r0
; CHECK-DSP-NEXT:   movs    r0, #47
; CHECK-DSP-NEXT:   cmp     r1, #254
; CHECK-DSP-NEXT:   it      lo
; CHECK-DSP-NEXT:   movlo   r0, #35

; CHECK-DSP-IMM:      movs r1, #1
; CHECK-DSP-IMM-NEXT: uadd8 r1, r0, r1
; CHECK-DSP-IMM-NEXT: movs  r0, #47
; CHECK-DSP-IMM-NEXT: cmp r1, #254
; CHECK-DSP-IMM-NEXT: it  lo
; CHECK-DSP-IMM-NEXT: movlo r0, #35
define i32 @test_ult_254_inc_imm(i8 zeroext %x) {
entry:
  %add = add i8 %x, 1
  %cmp = icmp ult i8 %add, 254
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_slt_254_inc_imm
; CHECK-COMMON: adds
; CHECK-COMMON: sxtb
define i32 @test_slt_254_inc_imm(i8 signext %x) {
entry:
  %add = add i8 %x, 1
  %cmp = icmp slt i8 %add, 254
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_ult_254_inc_var:
; CHECK-NODSP:      add     r0, r1
; CHECK-NODSP-NEXT: uxtb    r1, r0
; CHECK-NODSP-NEXT: movs    r0, #47
; CHECK-NODSP-NEXT: cmp     r1, #254
; CHECK-NODSP-NEXT: it      lo
; CHECK-NODSP-NEXT: movlo   r0, #35

; CHECK-DSP:        uadd8   r1, r0, r1
; CHECK-DSP-NEXT:   movs    r0, #47
; CHECK-DSP-NEXT:   cmp     r1, #254
; CHECK-DSP-NEXT:   it      lo
; CHECK-DSP-NEXT:   movlo   r0, #35
define i32 @test_ult_254_inc_var(i8 zeroext %x, i8 zeroext %y) {
entry:
  %add = add i8 %x, %y
  %cmp = icmp ult i8 %add, 254
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_sle_254_inc_var
; CHECK-COMMON: add
; CHECK-COMMON: sxtb
; CHECK-COMMON: cmp
define i32 @test_sle_254_inc_var(i8 %x, i8 %y) {
entry:
  %add = add i8 %x, %y
  %cmp = icmp sle i8 %add, 254
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_ugt_1_dec_imm:
; CHECK-COMMON:      subs    r1, r0, #1
; CHECK-COMMON-NEXT: movs    r0, #47
; CHECK-COMMON-NEXT: cmp     r1, #1
; CHECK-COMMON-NEXT: it      hi
; CHECK-COMMON-NEXT: movhi   r0, #35
define i32 @test_ugt_1_dec_imm(i8 zeroext %x) {
entry:
  %add = add i8 %x, -1
  %cmp = icmp ugt i8 %add, 1
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_sgt_1_dec_imm
; CHECK-COMMON: subs
; CHECK-COMMON: sxtb
; CHECK-COMMON: cmp
define i32 @test_sgt_1_dec_imm(i8 %x) {
entry:
  %add = add i8 %x, -1
  %cmp = icmp sgt i8 %add, 1
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_ugt_1_dec_var:
; CHECK-NODSP:      subs    r0, r0, r1
; CHECK-NODSP-NEXT: uxtb    r1, r0
; CHECK-NODSP-NEXT: movs    r0, #47
; CHECK-NODSP-NEXT: cmp     r1, #1
; CHECK-NODSP-NEXT: it      hi
; CHECK-NODSP-NEXT: movhi   r0, #35

; CHECK-DSP:      usub8   r1, r0, r1
; CHECK-DSP-NEXT: movs    r0, #47
; CHECK-DSP-NEXT: cmp     r1, #1
; CHECK-DSP-NEXT: it      hi
; CHECK-DSP-NEXT: movhi   r0, #35
define i32 @test_ugt_1_dec_var(i8 zeroext %x, i8 zeroext %y) {
entry:
  %sub = sub i8 %x, %y
  %cmp = icmp ugt i8 %sub, 1
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: test_sge_1_dec_var
; CHECK-COMMON: sub
; CHECK-COMMON: sxtb
; CHECK-COMMON: cmp
define i32 @test_sge_1_dec_var(i8 %x, i8 %y) {
entry:
  %sub = sub i8 %x, %y
  %cmp = icmp sge i8 %sub, 1
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: dsp_imm1:
; CHECK-DSP:      eors    r1, r0
; CHECK-DSP-NEXT: and     r0, r0, #7
; CHECK-DSP-NEXT: subs    r0, r0, r1
; CHECK-DSP-NEXT: adds    r0, #1
; CHECK-DSP-NEXT: uxtb    r1, r0
; CHECK-DSP-NEXT: movs    r0, #47
; CHECK-DSP-NEXT: cmp     r1, #254
; CHECK-DSP-NEXT: it      lo
; CHECK-DSP-NEXT: movlo   r0, #35

; CHECK-DSP-IMM:      eors    r1, r0
; CHECK-DSP-IMM-NEXT: and     r0, r0, #7
; CHECK-DSP-IMM-NEXT: usub8   r0, r0, r1
; CHECK-DSP-IMM-NEXT: movs    r1, #1
; CHECK-DSP-IMM-NEXT: uadd8   r1, r0, r1
; CHECK-DSP-IMM-NEXT: movs    r0, #47
; CHECK-DSP-IMM-NEXT: cmp     r1, #254
; CHECK-DSP-IMM-NEXT: it      lo
; CHECK-DSP-IMM-NEXT: movlo   r0, #35
define i32 @dsp_imm1(i8 zeroext %x, i8 zeroext %y) {
entry:
  %xor = xor i8 %x, %y
  %and = and i8 %x, 7
  %sub = sub i8 %and, %xor
  %add = add i8 %sub, 1
  %cmp = icmp ult i8 %add, 254
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: dsp_imm2
; CHECK-COMMON:   add   r0, r1
; CHECK-DSP-NEXT: ldrh  r1, [r3]
; CHECK-DSP-NEXT: ldrh  r2, [r2]
; CHECK-DSP-NEXT: subs  r1, r1, r0
; CHECK-DSP-NEXT: add   r0, r2
; CHECK-DSP-NEXT: uxth  r3, r1
; CHECK-DSP-NEXT: uxth  r2, r0
; CHECK-DSP-NEXT: cmp   r2, r3

; CHECK-DSP-IMM:      movs  r1, #0
; CHECK-DSP-IMM-NEXT: uxth  r0, r0
; CHECK-DSP-IMM-NEXT: usub16  r1, r1, r0
; CHECK-DSP-IMM-NEXT: ldrh  r0, [r2]
; CHECK-DSP-IMM-NEXT: ldrh  r3, [r3]
; CHECK-DSP-IMM-NEXT: usub16  r0, r0, r1
; CHECK-DSP-IMM-NEXT: uadd16  r1, r3, r1
; CHECK-DSP-IMM-NEXT: cmp r0, r1

define i16 @dsp_imm2(i32 %arg0, i32 %arg1, i16* %gep0, i16* %gep1) {
entry:
  %add0 = add i32 %arg0, %arg1
  %conv0 = trunc i32 %add0 to i16
  %sub0 = sub i16 0, %conv0
  %load0 = load i16, i16* %gep0, align 2
  %load1 = load i16, i16* %gep1, align 2
  %sub1 = sub i16 %load0, %sub0
  %add1 = add i16 %load1, %sub0
  %cmp = icmp ult i16 %sub1, %add1
  %res = select i1 %cmp, i16 %add1, i16 %sub1
  ret i16 %res
}

; CHECK-COMMON-LABEL: dsp_var:
; CHECK-COMMON:   eors    r1, r0
; CHECK-COMMON:   and     r2, r0, #7
; CHECK-NODSP:    subs    r1, r2, r1
; CHECK-NODSP:    add.w   r0, r1, r0, lsl #1
; CHECK-NODSP:    uxtb    r1, r0
; CHECK-DSP:      usub8   r1, r2, r1
; CHECK-DSP:      lsls    r0, r0, #1
; CHECK-DSP:      uadd8   r1, r1, r0
; CHECK-DSP-NOT:  uxt
; CHECK-COMMON:   movs    r0, #47
; CHECK-COMMON:   cmp     r1, #254
; CHECK-COMMON:   it      lo
; CHECK-COMMON:   movlo   r0, #35
define i32 @dsp_var(i8 zeroext %x, i8 zeroext %y) {
  %xor = xor i8 %x, %y
  %and = and i8 %x, 7
  %sub = sub i8 %and, %xor
  %mul = shl nuw i8 %x, 1
  %add = add i8 %sub, %mul
  %cmp = icmp ult i8 %add, 254
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: store_dsp_res
; CHECK-DSP: usub8 
; CHECK-DSP: strb
define void @store_dsp_res(i8* %in, i8* %out, i8 %compare) {
  %first = getelementptr inbounds i8, i8* %in, i32 0
  %second = getelementptr inbounds i8, i8* %in, i32 1
  %ld0 = load i8, i8* %first
  %ld1 = load i8, i8* %second
  %xor = xor i8 %ld0, -1
  %cmp = icmp ult i8 %compare, %ld1
  %select = select i1 %cmp, i8 %compare, i8 %xor
  %sub = sub i8 %ld0, %select
  store i8 %sub, i8* %out, align 1
  ret void
}

; CHECK-COMMON-LABEL: ugt_1_dec_imm:
; CHECK-COMMON:      subs    r1, r0, #1
; CHECK-COMMON-NEXT: movs    r0, #47
; CHECK-COMMON-NEXT: cmp     r1, #1
; CHECK-COMMON-NEXT: it      hi
; CHECK-COMMON-NEXT: movhi   r0, #35
define i32 @ugt_1_dec_imm(i8 zeroext %x) {
entry:
  %add = add i8 %x, -1
  %cmp = icmp ugt i8 %add, 1
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: ugt_1_dec_var:
; CHECK-NODSP:      subs    r0, r0, r1
; CHECK-NODSP-NEXT: uxtb    r1, r0
; CHECK-NODSP-NEXT: movs    r0, #47
; CHECK-NODSP-NEXT: cmp     r1, #1
; CHECK-NODSP-NEXT: it      hi
; CHECK-NODSP-NEXT: movhi   r0, #35

; CHECK-DSP:      usub8   r1, r0, r1
; CHECK-DSP-NEXT: movs    r0, #47
; CHECK-DSP-NEXT: cmp     r1, #1
; CHECK-DSP-NEXT: it      hi
; CHECK-DSP-NEXT: movhi   r0, #35
define i32 @ugt_1_dec_var(i8 zeroext %x, i8 zeroext %y) {
entry:
  %sub = sub i8 %x, %y
  %cmp = icmp ugt i8 %sub, 1
  %res = select i1 %cmp, i32 35, i32 47
  ret i32 %res
}

; CHECK-COMMON-LABEL: icmp_i32_zext:
; CHECK-COMMON:     ldrb [[LD:r[^ ]+]], [r0]
; CHECK-COMMON:     subs [[SUB:r[^ ]+]], [[LD]], #1
; CHECK-COMMON-NOT: uxt
; CHECK-COMMON:     cmp [[LD]], [[SUB]]
; CHECK-COMMON-NOT: uxt
define i8 @icmp_i32_zext(i8* %ptr) {
entry:
  %gep = getelementptr inbounds i8, i8* %ptr, i32 0
  %0 = load i8, i8* %gep, align 1
  %1 = sub nuw nsw i8 %0, 1
  %conv44 = zext i8 %0 to i32
  br label %preheader

preheader:
  br label %body

body:
  %2 = phi i8 [ %1, %preheader ], [ %3, %if.end ]
  %si.0274 = phi i32 [ %conv44, %preheader ], [ %inc, %if.end ]
  %conv51266 = zext i8 %2 to i32
  %cmp52267 = icmp eq i32 %si.0274, %conv51266
  br i1 %cmp52267, label %if.end, label %exit

if.end:
  %inc = add i32 %si.0274, 1
  %gep1 = getelementptr inbounds i8, i8* %ptr, i32 %inc
  %3 = load i8, i8* %gep1, align 1
  br label %body

exit:
  ret i8 %2
}

@d_uch = hidden local_unnamed_addr global [16 x i8] zeroinitializer, align 1
@sh1 = hidden local_unnamed_addr global i16 0, align 2
@d_sh = hidden local_unnamed_addr global [16 x i16] zeroinitializer, align 2

; CHECK-COMMON-LABEL: icmp_sext_zext_store_i8_i16
; CHECK-NODSP: ldrb [[BYTE:r[^ ]+]],
; CHECK-NODSP: strh [[BYTE]],
; CHECK-NODSP: ldrsh.w
define i32 @icmp_sext_zext_store_i8_i16() {
entry:
  %0 = load i8, i8* getelementptr inbounds ([16 x i8], [16 x i8]* @d_uch, i32 0, i32 2), align 1
  %conv = zext i8 %0 to i16
  store i16 %conv, i16* @sh1, align 2
  %conv1 = zext i8 %0 to i32
  %1 = load i16, i16* getelementptr inbounds ([16 x i16], [16 x i16]* @d_sh, i32 0, i32 2), align 2
  %conv2 = sext i16 %1 to i32
  %cmp = icmp eq i32 %conv1, %conv2
  %conv3 = zext i1 %cmp to i32
  ret i32 %conv3
}

; CHECK-COMMON-LABEL: or_icmp_ugt:
; CHECK-COMMON:     ldrb [[LD:r[^ ]+]], [r1]
; CHECK-COMMON:     subs [[SUB:r[^ ]+]], #1
; CHECK-COMMON-NOT: uxtb
; CHECK-COMMON:     cmp [[SUB]], #3
define i1 @or_icmp_ugt(i32 %arg, i8* %ptr) {
entry:
  %0 = load i8, i8* %ptr
  %1 = zext i8 %0 to i32
  %mul = shl nuw nsw i32 %1, 1
  %add0 = add nuw nsw i32 %mul, 6
  %cmp0 = icmp ne i32 %arg, %add0
  %add1 = add i8 %0, -1
  %cmp1 = icmp ugt i8 %add1, 3
  %or = or i1 %cmp0, %cmp1
  ret i1 %or
}

; CHECK-COMMON-LABEL: icmp_switch_trunc:
; CHECK-COMMON-NOT: uxt
define i16 @icmp_switch_trunc(i16 zeroext %arg) {
entry:
  %conv = add nuw i16 %arg, 15
  %mul = mul nuw nsw i16 %conv, 3
  %trunc = trunc i16 %arg to i3
  switch i3 %trunc, label %default [
    i3 0, label %sw.bb
    i3 1, label %sw.bb.i
  ]

sw.bb:
  %cmp0 = icmp ult i16 %mul, 127
  %select = select i1 %cmp0, i16 %mul, i16 127
  br label %exit

sw.bb.i:
  %cmp1 = icmp ugt i16 %mul, 34
  %select.i = select i1 %cmp1, i16 %mul, i16 34
  br label %exit

default:
  br label %exit

exit:
  %res = phi i16 [ %select, %sw.bb ], [ %select.i, %sw.bb.i ], [ %mul, %default ]
  ret i16 %res
}

; CHECK-COMMON-LABEL: icmp_eq_minus_one
; CHECK-COMMON: cmp r0, #255
define i32 @icmp_eq_minus_one(i8* %ptr) {
  %load = load i8, i8* %ptr, align 1
  %conv = zext i8 %load to i32
  %cmp = icmp eq i8 %load, -1
  %ret = select i1 %cmp, i32 %conv, i32 -1
  ret i32 %ret
}

; CHECK-COMMON-LABEL: icmp_not
; CHECK-COMMON: movw r2, #65535
; CHECK-COMMON: eors r2, r0
; CHECK-COMMON: movs r0, #32
; CHECK-COMMON: cmp r2, r1
define i32 @icmp_not(i16 zeroext %arg0, i16 zeroext %arg1) {
  %not = xor i16 %arg0, -1
  %cmp = icmp eq i16 %not, %arg1
  %res = select i1 %cmp, i32 16, i32 32
  ret i32 %res
}

; CHECK-COMMON-LABEL: mul_wrap
; CHECK-COMMON: mul
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
define i16 @mul_wrap(i16 %arg0, i16 %arg1) {
  %mul = mul i16 %arg0, %arg1
  %cmp = icmp eq i16 %mul, 1
  %res = select i1 %cmp, i16 %arg0, i16 47
  ret i16 %res
}

; CHECK-COMMON-LABEL: shl_wrap
; CHECK-COMMON: lsl
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
define i16 @shl_wrap(i16 %arg0) {
  %mul = shl i16 %arg0, 4
  %cmp = icmp eq i16 %mul, 1
  %res = select i1 %cmp, i16 %arg0, i16 47
  ret i16 %res
}

; CHECK-COMMON-LABEL: add_wrap
; CHECK-COMMON: add
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
define i16 @add_wrap(i16 %arg0, i16 %arg1) {
  %add = add i16 %arg0, 128
  %cmp = icmp eq i16 %add, %arg1
  %res = select i1 %cmp, i16 %arg0, i16 1
  ret i16 %res
}

; CHECK-COMMON-LABEL: sub_wrap
; CHECK-COMMON: sub
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
define i16 @sub_wrap(i16 %arg0, i16 %arg1, i16 %arg2) {
  %sub = sub i16 %arg0, %arg2
  %cmp = icmp eq i16 %sub, %arg1
  %res = select i1 %cmp, i16 %arg0, i16 1
  ret i16 %res
}

; CHECK-COMMON-LABEL: urem_trunc_icmps
; CHECK-COMMON-NOT: uxt
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
  %cmp = icmp ugt i8 %cond.in.i.off0, 7
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
