; RUN: llc -mtriple=thumbv8.main -mcpu=cortex-m33 %s -arm-disable-cgp=false -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
; RUN: llc -mtriple=thumbv7-linux-android %s -arm-disable-cgp=false -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
; RUN: llc -mtriple=thumbv7em %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP
; RUN: llc -mtriple=thumbv8 %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -arm-enable-scalar-dsp-imms=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP-IMM

; Transform will fail because the trunc is not a sink.
; CHECK-COMMON-LABEL: dsp_trunc
; CHECK-COMMON:   add   [[ADD:[^ ]+]],
; CHECK-DSP-NEXT: ldrh  r1, [r3]
; CHECK-DSP-NEXT: ldrh  r2, [r2]
; CHECK-DSP-NEXT: subs  r1, r1, [[ADD]]
; CHECK-DSP-NEXT: add   r0, r2
; CHECK-DSP-NEXT: uxth  r3, r1
; CHECK-DSP-NEXT: uxth  r2, r0
; CHECK-DSP-NEXT: cmp   r2, r3

; With DSP-IMM, we could have:
; movs  r1, #0
; uxth  r0, r0
; usub16  r1, r1, r0
; ldrh  r0, [r2]
; ldrh  r3, [r3]
; usub16  r0, r0, r1
; uadd16  r1, r3, r1
; cmp r0, r1
define i16 @dsp_trunc(i32 %arg0, i32 %arg1, i16* %gep0, i16* %gep1) {
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

; CHECK-COMMON-LABEL: trunc_i16_i8
; CHECK-COMMON: ldrh
; CHECK-COMMON: uxtb
; CHECK-COMMON: cmp
define i8 @trunc_i16_i8(i16* %ptr, i16 zeroext %arg0, i8 zeroext %arg1) {
entry:
  %0 = load i16, i16* %ptr
  %1 = add i16 %0, %arg0
  %2 = trunc i16 %1 to i8
  %3 = icmp ugt i8 %2, %arg1
  %4 = select i1 %3, i8 %2, i8 %arg1
  ret i8 %4
}

; The pass perform the transform, but a uxtb will still be inserted to handle
; the zext to the icmp.
; CHECK-COMMON-LABEL: icmp_i32_zext:
; CHECK-COMMON: sub
; CHECK-COMMON: uxtb
; CHECK-COMMON: cmp
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

; Won't don't handle sext
; CHECK-COMMON-LABEL: icmp_sext_zext_store_i8_i16
; CHECK-COMMON: ldrb
; CHECK-COMMON: ldrsh
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
; CHECK-COMMON:     ldrb
; CHECK-COMMON:     subs.w
; CHECK-COMMON-NOT: uxt
; CHECK-COMMON:     cmp
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

; We currently only handle truncs as sinks, so a uxt will still be needed for
; the icmp ugt instruction.
; CHECK-COMMON-LABEL: urem_trunc_icmps
; CHECK-COMMON: cmp
; CHECK-COMMON: uxt
; CHECK-COMMON: cmp
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

; Check that %exp requires uxth in all cases, and will also be required to
; promote %1 for the call - unless we can generate a uadd16.
; CHECK-COMMON-LABEL: zext_load_sink_call:
; CHECK-COMMON: uxt
; CHECK-DSP-IMM: uadd16
; CHECK-COMMON: cmp
; CHECK-NODSP: uxt
; CHECK-DSP-IMM-NOT: uxt
define i32 @zext_load_sink_call(i16* %ptr, i16 %exp) {
entry:
  %0 = load i16, i16* %ptr, align 4
  %1 = add i16 %exp, 3
  %cmp = icmp eq i16 %0, %exp
  br i1 %cmp, label %exit, label %if.then

if.then:
  %conv0 = zext i16 %0 to i32
  %conv1 = zext i16 %1 to i32
  %call = tail call arm_aapcs_vfpcc i32 @dummy(i32 %conv0, i32 %conv1)
  br label %exit

exit:
  %exitval = phi i32 [ %call, %if.then ], [ 0, %entry  ]
  ret i32 %exitval
}

; CHECK-COMMON-LABEL: bitcast_i16
; CHECK-COMMON-NOT: uxt
define i16 @bitcast_i16(i16 zeroext %arg0, i16 zeroext %arg1) {
entry:
  %cast = bitcast i16 12345 to i16
  %add = add nuw i16 %arg0, 1
  %cmp = icmp ule i16 %add, %cast
  %res = select i1 %cmp, i16 %arg1, i16 32657
  ret i16 %res
}

; CHECK-COMMON-LABEL: bitcast_i8
; CHECK-COMMON-NOT: uxt
define i8 @bitcast_i8(i8 zeroext %arg0, i8 zeroext %arg1) {
entry:
  %cast = bitcast i8 127 to i8
  %mul = shl nuw i8 %arg0, 1
  %cmp = icmp uge i8 %mul, %arg1
  %res = select i1 %cmp, i8 %cast, i8 128
  ret i8 %res
}

; CHECK-COMMON-LABEL: bitcast_i16_minus
; CHECK-COMMON-NOT: uxt
define i16 @bitcast_i16_minus(i16 zeroext %arg0, i16 zeroext %arg1) {
entry:
  %cast = bitcast i16 -12345 to i16
  %xor = xor i16 %arg0, 7
  %cmp = icmp eq i16 %xor, %arg1
  %res = select i1 %cmp, i16 %cast, i16 32657
  ret i16 %res
}

; CHECK-COMMON-LABEL: bitcast_i8_minus
; CHECK-COMMON-NOT: uxt
define i8 @bitcast_i8_minus(i8 zeroext %arg0, i8 zeroext %arg1) {
entry:
  %cast = bitcast i8 -127 to i8
  %and = and i8 %arg0, 3
  %cmp = icmp ne i8 %and, %arg1
  %res = select i1 %cmp, i8 %cast, i8 128
  ret i8 %res
}

declare i32 @dummy(i32, i32)

@d_uch = hidden local_unnamed_addr global [16 x i8] zeroinitializer, align 1
@sh1 = hidden local_unnamed_addr global i16 0, align 2
@d_sh = hidden local_unnamed_addr global [16 x i16] zeroinitializer, align 2

; CHECK-COMMON-LABEL: two_stage_zext_trunc_mix
; CHECK-COMMON-NOT: uxt
define i8* @two_stage_zext_trunc_mix(i32* %this, i32 %__pos1, i32 %__n1, i32** %__str, i32 %__pos2, i32 %__n2) {
entry:
  %__size_.i.i.i.i = bitcast i32** %__str to i8*
  %0 = load i8, i8* %__size_.i.i.i.i, align 4
  %1 = and i8 %0, 1
  %tobool.i.i.i.i = icmp eq i8 %1, 0
  %__size_.i5.i.i = getelementptr inbounds i32*, i32** %__str, i32 %__n1
  %cast = bitcast i32** %__size_.i5.i.i to i32*
  %2 = load i32, i32* %cast, align 4
  %3 = lshr i8 %0, 1
  %4 = zext i8 %3 to i32
  %cond.i.i = select i1 %tobool.i.i.i.i, i32 %4, i32 %2
  %__size_.i.i.i.i.i = bitcast i32* %this to i8*
  %5 = load i8, i8* %__size_.i.i.i.i.i, align 4
  %6 = and i8 %5, 1
  %tobool.i.i.i.i.i = icmp eq i8 %6, 0
  %7 = getelementptr inbounds i8, i8* %__size_.i.i.i.i, i32 %__pos1
  %8 = getelementptr inbounds i8, i8* %__size_.i.i.i.i, i32 %__pos2
  %res = select i1 %tobool.i.i.i.i.i,  i8* %7, i8* %8
  ret i8* %res
}

; CHECK-COMMON-LABEL: search_through_zext_1
; CHECK-COMMON-NOT: uxt
define i8 @search_through_zext_1(i8 zeroext %a, i8 zeroext %b, i16 zeroext %c) {
entry:
  %add = add nuw i8 %a, %b
  %conv = zext i8 %add to i16
  %cmp = icmp ult i16 %conv, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %sub = sub nuw i8 %b, %a
  %conv2 = zext i8 %sub to i16
  %cmp2 = icmp ugt i16 %conv2, %c
  %res = select i1 %cmp2, i8 %a, i8 %b
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %res, %if.then ]
  ret i8 %retval
}

; TODO: We should be able to remove the uxtb here. The transform fails because
; the icmp ugt uses an i32, which is too large... but this doesn't matter
; because it won't be writing a large value to a register as a result.
; CHECK-COMMON-LABEL: search_through_zext_2
; CHECK-COMMON: uxtb
; CHECK-COMMON: uxtb
define i8 @search_through_zext_2(i8 zeroext %a, i8 zeroext %b, i16 zeroext %c, i32 %d) {
entry:
  %add = add nuw i8 %a, %b
  %conv = zext i8 %add to i16
  %cmp = icmp ult i16 %conv, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %sub = sub nuw i8 %b, %a
  %conv2 = zext i8 %sub to i32
  %cmp2 = icmp ugt i32 %conv2, %d
  %res = select i1 %cmp2, i8 %a, i8 %b
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %res, %if.then ]
  ret i8 %retval
}

; TODO: We should be able to remove the uxtb here as all the calculations are
; performed on i8s. The promotion of i8 to i16 and then the later truncation
; results in the uxtb.
; CHECK-COMMON-LABEL: search_through_zext_3
; CHECK-COMMON: uxtb
; CHECK-COMMON: uxtb
define i8 @search_through_zext_3(i8 zeroext %a, i8 zeroext %b, i16 zeroext %c, i32 %d) {
entry:
  %add = add nuw i8 %a, %b
  %conv = zext i8 %add to i16
  %cmp = icmp ult i16 %conv, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %trunc = trunc i16 %conv to i8
  %sub = sub nuw i8 %b, %trunc
  %conv2 = zext i8 %sub to i32
  %cmp2 = icmp ugt i32 %conv2, %d
  %res = select i1 %cmp2, i8 %a, i8 %b
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %res, %if.then ]
  ret i8 %retval
}

; TODO: We should be able to remove the uxt that gets introduced for %conv2
; CHECK-COMMON-LABEL: search_through_zext_cmp
; CHECK-COMMON: uxt
define i8 @search_through_zext_cmp(i8 zeroext %a, i8 zeroext %b, i16 zeroext %c) {
entry:
  %cmp = icmp ne i8 %a, %b
  %conv = zext i1 %cmp to i16
  %cmp1 = icmp ult i16 %conv, %c
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %sub = sub nuw i8 %b, %a
  %conv2 = zext i8 %sub to i16
  %cmp3 = icmp ugt i16 %conv2, %c
  %res = select i1 %cmp3, i8 %a, i8 %b
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %res, %if.then ]
  ret i8 %retval
}

; CHECK-COMMON-LABEL: search_through_zext_load
; CHECK-COMMON-NOT: uxt
define i8 @search_through_zext_load(i8* %a, i8 zeroext %b, i16 zeroext %c) {
entry:
  %load = load i8, i8* %a
  %conv = zext i8 %load to i16
  %cmp1 = icmp ult i16 %conv, %c
  br i1 %cmp1, label %if.then, label %if.end

if.then:
  %sub = sub nuw i8 %b, %load
  %conv2 = zext i8 %sub to i16
  %cmp3 = icmp ugt i16 %conv2, %c
  %res = select i1 %cmp3, i8 %load, i8 %b
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %res, %if.then ]
  ret i8 %retval
}

; CHECK-COMMON-LABEL: trunc_sink_less_than
; CHECK-COMMON-NOT: uxth
; CHECK-COMMON: cmp
; CHECK-COMMON: uxtb
define i16 @trunc_sink_less_than_cmp(i16 zeroext %a, i16 zeroext %b, i16 zeroext %c, i8 zeroext %d) {
entry:
  %sub = sub nuw i16 %b, %a
  %cmp = icmp ult i16 %sub, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %trunc = trunc i16 %sub to i8
  %add = add nuw i8 %d, 1
  %cmp2 = icmp ugt i8 %trunc, %add
  %res = select i1 %cmp2, i16 %a, i16 %b
  br label %if.end

if.end:
  %retval = phi i16 [ 0, %entry ], [ %res, %if.then ]
  ret i16 %retval
}

; TODO: We should be able to remove the uxth introduced to handle %sub
; CHECK-COMMON-LABEL: trunc_sink_less_than_arith
; CHECK-COMMON: uxth
; CHECK-COMMON: cmp
; CHECK-COMMON: uxtb
define i16 @trunc_sink_less_than_arith(i16 zeroext %a, i16 zeroext %b, i16 zeroext %c, i8 zeroext %d, i8 zeroext %e) {
entry:
  %sub = sub nuw i16 %b, %a
  %cmp = icmp ult i16 %sub, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %trunc = trunc i16 %sub to i8
  %add = add nuw i8 %d, %trunc
  %cmp2 = icmp ugt i8 %e, %add
  %res = select i1 %cmp2, i16 %a, i16 %b
  br label %if.end

if.end:
  %retval = phi i16 [ 0, %entry ], [ %res, %if.then ]
  ret i16 %retval
}

; CHECK-COMMON-LABEL: trunc_sink_less_than_store
; CHECK-COMMON-NOT: uxt
; CHECK-COMMON: cmp
; CHECK-COMMON-NOT: uxt
define i16 @trunc_sink_less_than_store(i16 zeroext %a, i16 zeroext %b, i16 zeroext %c, i8 zeroext %d, i8* %e) {
entry:
  %sub = sub nuw i16 %b, %a
  %cmp = icmp ult i16 %sub, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %trunc = trunc i16 %sub to i8
  %add = add nuw i8 %d, %trunc
  store i8 %add, i8* %e
  br label %if.end

if.end:
  %retval = phi i16 [ 0, %entry ], [ %sub, %if.then ]
  ret i16 %retval
}

; CHECK-COMMON-LABEL: trunc_sink_less_than_ret
; CHECK-COMMON-NOT: uxt
define i8 @trunc_sink_less_than_ret(i16 zeroext %a, i16 zeroext %b, i16 zeroext %c, i8 zeroext %d, i8 zeroext %e) {
entry:
  %sub = sub nuw i16 %b, %a
  %cmp = icmp ult i16 %sub, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %trunc = trunc i16 %sub to i8
  %add = add nuw i8 %d, %trunc
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %add, %if.then ]
  ret i8 %retval
}

; CHECK-COMMON-LABEL: trunc_sink_less_than_zext_ret
; CHECK-COMMON-NOT: uxth
; CHECK-COMMON: sub
; CHECK-COMMON: uxtb
define zeroext i8 @trunc_sink_less_than_zext_ret(i16 zeroext %a, i16 zeroext %b, i16 zeroext %c, i8 zeroext %d, i8 zeroext %e) {
entry:
  %sub = sub nuw i16 %b, %a
  %cmp = icmp ult i16 %sub, %c
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %trunc = trunc i16 %sub to i8
  %add = add nuw i8 %d, %trunc
  br label %if.end

if.end:
  %retval = phi i8 [ 0, %entry ], [ %add, %if.then ]
  ret i8 %retval
}

; CHECK-COMMON-LABEL: bitcast_i1
; CHECK-COMMON-NOT: uxt
define i32 @bitcast_i1(i16 zeroext %a, i32 %b, i32 %c) {
entry:
  %0 = bitcast i1 1 to i1
  %1 = trunc i16 %a to i1
  %cmp = icmp eq i1 %1, %0
  br i1 %cmp, label %if.then, label %exit

if.then:
  %conv = zext i1 %0 to i16
  %conv1 = zext i1 %1 to i16
  %cmp1 = icmp uge i16 %conv, %conv1
  %select = select i1 %cmp1, i32 %b, i32 %c
  br label %exit

exit:
  %retval = phi i32 [ %select, %if.then ], [ 0, %entry ]
  ret i32 %retval
}

; CHECK-COMMON-LABEL: search_back_through_trunc
; CHECK-COMMON-NOT: uxt
; CHECK-COMMON: cmp
; CHECK-COMMON: strb
; CHECK-COMMON: strb
define void @search_back_through_trunc(i8* %a, i8* %b, i8* %c, i8* %d, i16* %e) {
entry:
  %0 = load i8, i8* %a, align 1
  %conv106 = zext i8 %0 to i16
  %shl = shl nuw i16 %conv106, 8
  %1 = load i8, i8* %b, align 1
  %conv108 = zext i8 %1 to i16
  %or109 = or i16 %shl, %conv108
  %2 = load i8, i8* %c, align 1
  %conv119 = zext i8 %2 to i16
  %shl120 = shl nuw i16 %conv119, 8
  %3 = load i8, i8* %d, align 1
  %conv122 = zext i8 %3 to i16
  %or123 = or i16 %shl120, %conv122
  %cmp133 = icmp eq i16 %or109, %or123
  br i1 %cmp133, label %if.end183, label %if.else136

if.else136:
  %4 = load i16, i16* %e, align 2
  %extract.t854 = trunc i16 %4 to i8
  %extract856 = lshr i16 %4, 8
  %extract.t857 = trunc i16 %extract856 to i8
  br label %if.end183

if.end183:
  %w.0.off0 = phi i8 [ %extract.t854, %if.else136 ], [ %1, %entry ]
  %w.0.off8 = phi i8 [ %extract.t857, %if.else136 ], [ %2, %entry ]
  store i8 %w.0.off8, i8* %c, align 1
  store i8 %w.0.off0, i8* %d, align 1
  ret void
}

@c = common dso_local local_unnamed_addr global i16 0, align 2
@b = common dso_local local_unnamed_addr global i16 0, align 2
@f = common dso_local local_unnamed_addr global i32 0, align 4
@e = common dso_local local_unnamed_addr global i8 0, align 1
@a = common dso_local local_unnamed_addr global i8 0, align 1
@d = common dso_local local_unnamed_addr global i32 0, align 4

; CHECK-LABEL: and_trunc
; CHECK: ldrh
; CHECK: sxth
; CHECK: uxtb
define void @and_trunc_two_zext() {
entry:
  %0 = load i16, i16* @c, align 2
  %1 = load i16, i16* @b, align 2
  %conv = sext i16 %1 to i32
  store i32 %conv, i32* @f, align 4
  %2 = trunc i16 %1 to i8
  %conv1 = and i8 %2, 1
  store i8 %conv1, i8* @e, align 1
  %3 = load i8, i8* @a, align 1
  %narrow = mul nuw i8 %3, %conv1
  %mul = zext i8 %narrow to i32
  store i32 %mul, i32* @d, align 4
  %4 = zext i8 %narrow to i16
  %conv5 = or i16 %0, %4
  %tobool = icmp eq i16 %conv5, 0
  br i1 %tobool, label %if.end, label %for.cond

for.cond:
  br label %for.cond

if.end:
  ret void
}

; CHECK-LABEL: zext_urem_trunc
; CHECK-NOT: uxt
define void @zext_urem_trunc() {
entry:
  %0 = load i16, i16* @c, align 2
  %cmp = icmp eq i16 %0, 0
  %1 = load i8, i8* @e, align 1
  br i1 %cmp, label %cond.end, label %cond.false

cond.false:
  %rem.lhs.trunc = zext i8 %1 to i16
  %rem7 = urem i16 %rem.lhs.trunc, %0
  %rem.zext = trunc i16 %rem7 to i8
  br label %cond.end

cond.end:
  %cond = phi i8 [ %rem.zext, %cond.false ], [ %1, %entry ]
  store i8 %cond, i8* @a, align 1
  ret void
}

; CHECK-LABEL: dont_replace_trunc_1
; CHECK: cmp
; CHECK: uxtb
define void @dont_replace_trunc_1(i8* %a, i16* %b, i16* %c, i32* %d, i8* %e, i32* %f) {
entry:
  %0 = load i16, i16* %c, align 2
  %1 = load i16, i16* %b, align 2
  %conv = sext i16 %1 to i32
  store i32 %conv, i32* %f, align 4
  %2 = trunc i16 %1 to i8
  %conv1 = and i8 %2, 1
  store i8 %conv1, i8* %e, align 1
  %3 = load i8, i8* %a, align 1
  %narrow = mul nuw i8 %3, %conv1
  %mul = zext i8 %narrow to i32
  store i32 %mul, i32* %d, align 4
  %4 = zext i8 %narrow to i16
  %conv5 = or i16 %0, %4
  %tobool = icmp eq i16 %conv5, 0
  br i1 %tobool, label %if.end, label %for.cond

for.cond:                                         ; preds = %entry, %for.cond
  br label %for.cond

if.end:                                           ; preds = %entry
  ret void
}

; CHECK-LABEL: dont_replace_trunc_2
; CHECK: cmp
; CHECK: uxtb
define i32 @dont_replace_trunc_2(i16* %a, i8* %b) {
entry:
  %0 = load i16, i16* %a, align 2
  %cmp = icmp ugt i16 %0, 8
  %narrow = select i1 %cmp, i16 %0, i16 0
  %cond = trunc i16 %narrow to i8
  %1 = load i8, i8* %b, align 1
  %or = or i8 %1, %cond
  store i8 %or, i8* %b, align 1
  %conv5 = zext i8 %or to i32
  ret i32 %conv5
}
