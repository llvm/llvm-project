; RUN: llc -mtriple=thumbv8m.main -mcpu=cortex-m33 %s -arm-disable-cgp=false -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
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

; CHECK-COMMON-LABEL: icmp_i1
; CHECK-NOT: uxt
define i32 @icmp_i1(i1* %arg0, i1 zeroext %arg1, i32 %a, i32 %b) {
entry:
  %load = load i1, i1* %arg0
  %not = xor i1 %load, 1
  %cmp = icmp eq i1 %arg1, %not
  %res = select i1 %cmp, i32 %a, i32 %b
  ret i32 %res
}

; CHECK-COMMON-LABEL: icmp_i7
; CHECK-COMMON: ldrb
; CHECK-COMMON: cmp
define i32 @icmp_i7(i7* %arg0, i7 zeroext %arg1, i32 %a, i32 %b) {
entry:
  %load = load i7, i7* %arg0
  %add = add nuw i7 %load, 1
  %cmp = icmp ult i7 %arg1, %add
  %res = select i1 %cmp, i32 %a, i32 %b
  ret i32 %res
}

; CHECK-COMMON-LABEL: icmp_i15
; CHECK-COMMON: movw [[MINUS_ONE:r[0-9]+]], #32767
define i32 @icmp_i15(i15 zeroext %arg0, i15 zeroext %arg1) {
  %xor = xor i15 %arg0, -1
  %cmp = icmp eq i15 %xor, %arg1
  %res = select i1 %cmp, i32 21, i32 42
  ret i32 %res
}

; CHECK-COMMON-LABEL: icmp_minus_imm
; CHECK-NODSP: subs [[SUB:r[0-9]+]],
; CHECK-NODSP: uxtb [[UXT:r[0-9]+]],
; CHECK-NODSP: cmp [[UXT]], #251

; CHECK-DSP: subs [[SUB:r[0-9]+]],
; CHECK-DSP: uxtb [[UXT:r[0-9]+]],
; CHECK-DSP: cmp [[UXT]], #251

; CHECK-DSP-IMM: ldrb [[A:r[0-9]+]],
; CHECK-DSP-IMM: movs  [[MINUS_7:r[0-9]+]], #249
; CHECK-DSP-IMM: uadd8 [[RES:r[0-9]+]], [[A]], [[MINUS_7]]
; CHECK-DSP-IMM: cmp [[RES]], #251
define i32 @icmp_minus_imm(i8* %a) {
entry:
  %0 = load i8, i8* %a, align 1
  %add.i = add i8 %0, -7
  %cmp = icmp ugt i8 %add.i, -5
  %conv1 = zext i1 %cmp to i32
  ret i32 %conv1
}

; CHECK-COMMON-LABEL: mul_with_neg_imm
; CHECK-COMMON-NOT: uxtb
; CHECK-COMMON:     and [[BIT0:r[0-9]+]], r0, #1
; CHECK-COMMON:     add.w [[MUL32:r[0-9]+]], [[BIT0]], [[BIT0]], lsl #5
; CHECK-COMMON:     cmp.w r0, [[MUL32]], lsl #2
define void @mul_with_neg_imm(i32, i32* %b) {
entry:
  %1 = trunc i32 %0 to i8
  %2 = and i8 %1, 1
  %conv.i = mul nuw i8 %2, -124
  %tobool = icmp eq i8 %conv.i, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  store i32 0, i32* %b, align 4
  br label %if.end

if.end:
  ret void
}
