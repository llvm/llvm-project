; RUN: llc -mtriple=thumbv7m -arm-disable-cgp=false %s -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
; RUN: llc -mtriple=thumbv8m.main -arm-disable-cgp=false %s -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-NODSP
; RUN: llc -mtriple=thumbv8m.main -arm-disable-cgp=false -arm-enable-scalar-dsp=true -mcpu=cortex-m33 %s -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP
; RUN: llc -mtriple=thumbv7em %s -arm-disable-cgp=false -arm-enable-scalar-dsp=true -arm-enable-scalar-dsp-imms=true -o - | FileCheck %s --check-prefix=CHECK-COMMON --check-prefix=CHECK-DSP-IMM

; Test that ARMCodeGenPrepare can handle:
; - loops
; - call operands
; - call return values
; - ret instructions
; We use nuw on the arithmetic instructions to avoid complications.

; Check that the arguments are extended but then nothing else is.
; This also ensures that the pass can handle loops.
; CHECK-COMMON-LABEL: phi_feeding_phi_args
; CHECK-COMMON: uxtb
; CHECK-COMMON: uxtb
; CHECK-NOT: uxtb
define void @phi_feeding_phi_args(i8 %a, i8 %b) {
entry:
  %0 = icmp ugt i8 %a, %b
  br i1 %0, label %preheader, label %empty

empty:
  br label %preheader

preheader:
  %1 = phi i8 [ %a, %entry ], [ %b, %empty ]
  br label %loop

loop:
  %val = phi i8 [ %1, %preheader ], [ %inc2, %if.end ]
  %cmp = icmp ult i8 %val, 254
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = sub nuw i8 %val, 2
  br label %if.end

if.else:
  %inc1 = shl nuw i8 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i8 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp eq i8 %inc2, 255
  br i1 %cmp1, label %exit, label %loop

exit:
  ret void
}

; Same as above, but as the args are zeroext, we shouldn't see any uxts.
; CHECK-COMMON-LABEL: phi_feeding_phi_zeroext_args
; CHECK-COMMON-NOT: uxt
define void @phi_feeding_phi_zeroext_args(i8 zeroext %a, i8 zeroext %b) {
entry:
  %0 = icmp ugt i8 %a, %b
  br i1 %0, label %preheader, label %empty

empty:
  br label %preheader

preheader:
  %1 = phi i8 [ %a, %entry ], [ %b, %empty ]
  br label %loop

loop:
  %val = phi i8 [ %1, %preheader ], [ %inc2, %if.end ]
  %cmp = icmp ult i8 %val, 254
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = sub nuw i8 %val, 2
  br label %if.end

if.else:
  %inc1 = shl nuw i8 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i8 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp eq i8 %inc2, 255
  br i1 %cmp1, label %exit, label %loop

exit:
  ret void
}

; Just check that phis also work with i16s.
; CHECK-COMMON-LABEL: phi_i16:
; CHECK-COMMON-NOT:   uxt
define void @phi_i16() {
entry:
  br label %loop

loop:
  %val = phi i16 [ 0, %entry ], [ %inc2, %if.end ]
  %cmp = icmp ult i16 %val, 128
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = add nuw i16 %val, 2
  br label %if.end

if.else:
  %inc1 = add nuw i16 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i16 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp ult i16 %inc2, 253
  br i1 %cmp1, label %loop, label %exit

exit:
  ret void
}

; CHECK-COMMON-LABEL: phi_feeding_switch
; CHECK-COMMON: ldrb
; CHECK-COMMON: uxtb
; CHECK-COMMON-NOT: uxt
define void @phi_feeding_switch(i8* %memblock, i8* %store, i16 %arg) {
entry:
  %pre = load i8, i8* %memblock, align 1
  %conv = trunc i16 %arg to i8
  br label %header

header:
  %phi.0 = phi i8 [ %pre, %entry ], [ %count, %latch ]
  %phi.1 = phi i8 [ %conv, %entry ], [ %phi.3, %latch ]
  %phi.2 = phi i8 [ 0, %entry], [ %count, %latch ]
  switch i8 %phi.0, label %default [
    i8 43, label %for.inc.i
    i8 45, label %for.inc.i.i
  ]

for.inc.i:
  %xor = xor i8 %phi.1, 1
  br label %latch

for.inc.i.i:
  %and = and i8 %phi.1, 3
  br label %latch

default:
  %sub = sub i8 %phi.0, 1
  %cmp2 = icmp ugt i8 %sub, 4
  br i1 %cmp2, label %latch, label %exit

latch:
  %phi.3 = phi i8 [ %xor, %for.inc.i ], [ %and, %for.inc.i.i ], [ %phi.2, %default ]
  %count = add nuw i8 %phi.2, 1
  store i8 %count, i8* %store, align 1
  br label %header

exit:
  ret void
}

; CHECK-COMMON-LABEL: ret_i8
; CHECK-COMMON-NOT:   uxt
define i8 @ret_i8() {
entry:
  br label %loop

loop:
  %val = phi i8 [ 0, %entry ], [ %inc2, %if.end ]
  %cmp = icmp ult i8 %val, 128
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %inc = add nuw i8 %val, 2
  br label %if.end

if.else:
  %inc1 = add nuw i8 %val, 1
  br label %if.end

if.end:
  %inc2 = phi i8 [ %inc, %if.then], [ %inc1, %if.else ]
  %cmp1 = icmp ult i8 %inc2, 253
  br i1 %cmp1, label %exit, label %loop

exit:
  ret i8 %inc2
}

; Check that %exp requires uxth in all cases, and will also be required to
; promote %1 for the call - unless we can generate a uadd16.
; CHECK-COMMON-LABEL: zext_load_sink_call:
; CHECK-COMMON:       uxt
; CHECK-DSP-IMM:      uadd16
; CHECK-COMMON:       cmp
; CHECK-DSP:          uxt
; CHECK-DSP-IMM-NOT:  uxt
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


; Check that the pass doesn't try to promote the immediate parameters.
; CHECK-COMMON-LABEL: call_with_imms
; CHECK-COMMON-NOT:   uxt
define i8 @call_with_imms(i8* %arg) {
  %call = tail call arm_aapcs_vfpcc zeroext i8 @dummy2(i8* nonnull %arg, i8 zeroext 0, i8 zeroext 0)
  %cmp = icmp eq i8 %call, 0
  %res = select i1 %cmp, i8 %call, i8 1
  ret i8 %res
}

; Test that the call result is still extended.
; CHECK-COMMON-LABEL: test_call:
; CHECK-COMMON: bl
; CHECK-COMMONNEXT: sxtb r1, r0
define i16 @test_call(i8 zeroext %arg) {
  %call = call i8 @dummy_i8(i8 %arg)
  %cmp = icmp ult i8 %call, 128
  %conv = zext i1 %cmp to i16
  ret i16 %conv 
}

; Test that the transformation bails when it finds that i16 is larger than i8.
; TODO: We should be able to remove the uxtb in these cases.
; CHECK-LABEL: promote_i8_sink_i16_1
; CHECK-COMMON: bl dummy_i8
; CHECK-COMMON: adds r0, #1
; CHECK-COMMON: uxtb r0, r0
; CHECK-COMMON: cmp r0
define i16 @promote_i8_sink_i16_1(i8 zeroext %arg0, i16 zeroext %arg1, i16 zeroext %arg2) {
  %call = tail call zeroext i8 @dummy_i8(i8 %arg0)
  %add = add nuw i8 %call, 1
  %conv = zext i8 %add to i16
  %cmp = icmp ne i16 %conv, %arg1
  %sel = select i1 %cmp, i16 %arg1, i16 %arg2
  %res = tail call zeroext i16 @dummy3(i16 %sel)
  ret i16 %res
}

; CHECK-COMMON-LABEL: promote_i8_sink_i16_2
; CHECK-COMMON: bl dummy_i8
; CHECK-COMMON: adds r0, #1
; CHECK-COMMON: uxtb r0, r0
; CHECK-COMMON: cmp r0
define i16 @promote_i8_sink_i16_2(i8 zeroext %arg0, i8 zeroext %arg1, i16 zeroext %arg2) {
  %call = tail call zeroext i8 @dummy_i8(i8 %arg0)
  %add = add nuw i8 %call, 1
  %cmp = icmp ne i8 %add, %arg1
  %conv = zext i8 %arg1 to i16
  %sel = select i1 %cmp, i16 %conv, i16 %arg2
  %res = tail call zeroext i16 @dummy3(i16 %sel)
  ret i16 %res
}

@uc = global i8 42, align 1
@LL = global i64 0, align 8

; CHECK-COMMON-LABEL: zext_i64
; CHECK-COMMON: ldrb
; CHECK-COMMON: strd
define void @zext_i64() {
entry:
  %0 = load i8, i8* @uc, align 1
  %conv = zext i8 %0 to i64
  store i64 %conv, i64* @LL, align 8
  %cmp = icmp eq i8 %0, 42
  %conv1 = zext i1 %cmp to i32
  %call = tail call i32 bitcast (i32 (...)* @assert to i32 (i32)*)(i32 %conv1)
  ret void
}

@a = global i16* null, align 4
@b = global i32 0, align 4

; CHECK-COMMON-LABEL: constexpr
; CHECK-COMMON: uxth
define i32 @constexpr() {
entry:
  store i32 ptrtoint (i32* @b to i32), i32* @b, align 4
  %0 = load i16*, i16** @a, align 4
  %1 = load i16, i16* %0, align 2
  %or = or i16 %1, ptrtoint (i32* @b to i16)
  store i16 %or, i16* %0, align 2
  %cmp = icmp ne i16 %or, 4
  %conv3 = zext i1 %cmp to i32
  %call = tail call i32 bitcast (i32 (...)* @e to i32 (i32)*)(i32 %conv3) #2
  ret i32 undef
}

; Check that d.sroa.0.0.be is promoted passed directly into the tail call.
; CHECK-COMMON-LABEL: check_zext_phi_call_arg
; CHECK-COMMON-NOT: uxt
define i32 @check_zext_phi_call_arg() {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.cond.backedge, %entry
  %d.sroa.0.0 = phi i16 [ 30, %entry ], [ %d.sroa.0.0.be, %for.cond.backedge ]
  %tobool = icmp eq i16 %d.sroa.0.0, 0
  br i1 %tobool, label %for.cond.backedge, label %if.then

for.cond.backedge:                                ; preds = %for.cond, %if.then
  %d.sroa.0.0.be = phi i16 [ %call, %if.then ], [ 0, %for.cond ]
  br label %for.cond

if.then:                                          ; preds = %for.cond
  %d.sroa.0.0.insert.ext = zext i16 %d.sroa.0.0 to i32
  %call = tail call zeroext i16 bitcast (i16 (...)* @f to i16 (i32)*)(i32 %d.sroa.0.0.insert.ext) #2
  br label %for.cond.backedge
}


; The call to safe_lshift_func takes two parameters, but they're the same value just one is zext.
; CHECK-COMMON-LABEL: call_zext_i8_i32
define fastcc i32 @call_zext_i8_i32(i32 %p_45, i8 zeroext %p_46) {
for.cond8.preheader:
  %call217 = call fastcc zeroext i8 @safe_mul_func_uint8_t_u_u(i8 zeroext undef)
  %tobool219 = icmp eq i8 %call217, 0
  br i1 %tobool219, label %for.end411, label %for.cond273.preheader

for.cond273.preheader:                            ; preds = %for.cond8.preheader
  %call217.lcssa = phi i8 [ %call217, %for.cond8.preheader ]
  %conv218.le = zext i8 %call217.lcssa to i32
  %call346 = call fastcc zeroext i8 @safe_lshift_func(i8 zeroext %call217.lcssa, i32 %conv218.le)
  unreachable

for.end411:                                       ; preds = %for.cond8.preheader
  %call452 = call fastcc i64 @safe_sub_func_int64_t_s_s(i64 undef, i64 4)
  unreachable
}

%struct.anon = type { i32 }

@g_57 = hidden local_unnamed_addr global %struct.anon zeroinitializer, align 4
@g_893 = hidden local_unnamed_addr global %struct.anon zeroinitializer, align 4
@g_82 = hidden local_unnamed_addr global i32 0, align 4

; Test that the transform bails on finding a call which returns a i16**
; CHECK-COMMON-LABEL: call_return_pointer
; CHECK-COMMON: sxth
; CHECK-COMMON-NOT: uxt
define hidden i32 @call_return_pointer(i8 zeroext %p_13) local_unnamed_addr #0 {
entry:
  %conv1 = zext i8 %p_13 to i16
  %call = tail call i16** @func_62(i8 zeroext undef, i32 undef, i16 signext %conv1, i32* undef)
  %0 = load i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @g_893, i32 0, i32 0), align 4
  %conv2 = trunc i32 %0 to i16
  br label %for.cond

for.cond:                                         ; preds = %for.cond.backedge, %entry
  %p_13.addr.0 = phi i8 [ %p_13, %entry ], [ %p_13.addr.0.be, %for.cond.backedge ]
  %tobool = icmp eq i8 %p_13.addr.0, 0
  br i1 %tobool, label %for.cond.backedge, label %if.then

for.cond.backedge:                                ; preds = %for.cond, %if.then
  %p_13.addr.0.be = phi i8 [ %conv4, %if.then ], [ 0, %for.cond ]
  br label %for.cond

if.then:                                          ; preds = %for.cond
  %call3 = tail call fastcc signext i16 @safe_sub_func_int16_t_s_s(i16 signext %conv2)
  %conv4 = trunc i16 %call3 to i8
  br label %for.cond.backedge
}

declare noalias i16** @func_62(i8 zeroext %p_63, i32 %p_64, i16 signext %p_65, i32* nocapture readnone %p_66)
declare fastcc signext i16 @safe_sub_func_int16_t_s_s(i16 signext %si2)
declare dso_local fastcc i64 @safe_sub_func_int64_t_s_s(i64, i64)
declare dso_local fastcc zeroext i8 @safe_lshift_func(i8 zeroext, i32)
declare dso_local fastcc zeroext i8 @safe_mul_func_uint8_t_u_u(i8 returned zeroext)

declare dso_local i32 @e(...) local_unnamed_addr #1
declare dso_local zeroext i16 @f(...) local_unnamed_addr #1

declare i32 @dummy(i32, i32)
declare i8 @dummy_i8(i8)
declare i8 @dummy2(i8*, i8, i8)
declare i16 @dummy3(i16)
declare i32 @assert(...)
