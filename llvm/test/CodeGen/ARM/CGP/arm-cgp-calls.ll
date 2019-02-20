; RUN: llc -mtriple=thumbv8 -arm-disable-cgp=false %s -o - | FileCheck %s
; RUN: llc -mtriple=armv8 -arm-disable-cgp=false %s -o - | FileCheck %s

; Check that the pass doesn't try to promote the immediate parameters.
; CHECK-LABEL: call_with_imms
; CHECK-NOT:   uxt
define i8 @call_with_imms(i8* %arg) {
  %call = tail call arm_aapcs_vfpcc zeroext i8 @dummy2(i8* nonnull %arg, i8 zeroext 0, i8 zeroext 0)
  %cmp = icmp eq i8 %call, 0
  %res = select i1 %cmp, i8 %call, i8 1
  ret i8 %res
}

; Test that the call result is still extended.
; CHECK-LABEL: test_call:
; CHECK: bl
; CHECK-NEXT: sxtb r1, r0
define i16 @test_call(i8 zeroext %arg) {
  %call = call i8 @dummy_i8(i8 %arg)
  %cmp = icmp ult i8 %call, 128
  %conv = zext i1 %cmp to i16
  ret i16 %conv
}

; CHECK-LABEL: promote_i8_sink_i16_1
; CHECK: bl dummy_i8
; CHECK: add{{.*}} r0, #1
; CHECK-NOT: uxt
; CHECK: cmp r0
define i16 @promote_i8_sink_i16_1(i8 zeroext %arg0, i16 zeroext %arg1, i16 zeroext %arg2) {
  %call = tail call zeroext i8 @dummy_i8(i8 %arg0)
  %add = add nuw i8 %call, 1
  %conv = zext i8 %add to i16
  %cmp = icmp ne i16 %conv, %arg1
  %sel = select i1 %cmp, i16 %arg1, i16 %arg2
  %res = tail call zeroext i16 @dummy3(i16 %sel)
  ret i16 %res
}

; CHECK-LABEL: promote_i8_sink_i16_2
; CHECK: bl dummy_i8
; CHECK: add{{.*}} r0, #1
; CHECK-NOT: uxt
; CHECK: cmp r0
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

; CHECK-LABEL: zext_i64
; CHECK: ldrb
; CHECK: strd
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

; CHECK-LABEL: constexpr
; CHECK: uxth
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

; The call to safe_lshift_func takes two parameters, but they're the same value
; just one is zext. We do support zext now, so the transformation should
; trigger and we don't want see uxtb here.
; CHECK-LABEL: call_zext_i8_i32
; CHECK-NOT: uxt
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

; Test that the transform bails on finding %conv4, a trunc
; CHECK-LABEL: call_return_pointer
; CHECK: sxth
; CHECK: uxt
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

; Check that d.sroa.0.0.be is promoted passed directly into the tail call.
; CHECK-LABEL: check_zext_phi_call_arg
; CHECK-NOT: uxt
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

%struct.atomic_flag = type { i8 }

; CHECK-LABEL: atomic_flag_test_and_set
; CHECK-NOT: uxt
define zeroext i1 @atomic_flag_test_and_set(%struct.atomic_flag* %object) {
entry:
  %_Value = getelementptr inbounds %struct.atomic_flag, %struct.atomic_flag* %object, i32 0, i32 0
  %call = tail call arm_aapcscc zeroext i8 @__atomic_exchange_1(i8* %_Value, i8 zeroext 1, i32 5) #1
  %0 = and i8 %call, 1
  %tobool = icmp ne i8 %0, 0
  ret i1 %tobool
}

; CHECK-LABEL: i1_zeroext_call
; CHECK: uxt
define i1 @i1_zeroext_call(i16* %ts, i32 %a, i16* %b, i8* %c) {
entry:
  %0 = load i16, i16* %ts, align 2
  %conv.i860 = trunc i32 %a to i16
  store i16 %conv.i860, i16* %b, align 2
  %call.i848 = call zeroext i1 @i1_zeroext(i8* %c, i32 64, i16 zeroext %conv.i860)
  br i1 %call.i848, label %if.then223, label %if.else227

if.then223:
  %cmp235 = icmp eq i16 %0, %conv.i860
  br label %exit

if.else227:
  %cmp236 = icmp ult i16 %0, %conv.i860
  br label %exit

exit:
  %retval = phi i1 [ %cmp235, %if.then223 ], [ %cmp236, %if.else227 ]
  ret i1 %retval
}

; CHECK-LABEL: promote_arg_pass_to_call
; CHECK-NOT: uxt
define i16 @promote_arg_pass_to_call(i16 zeroext %arg1, i16 zeroext %arg2) {
  %conv = add nuw i16 %arg1, 15
  %mul = mul nuw nsw i16 %conv, 3
  %cmp = icmp ult i16 %mul, %arg2
  %trunc = trunc i16 %arg1 to i8
  %res = call zeroext i16 @dummy4(i1 %cmp, i8 %trunc, i16 %arg1)
  ret i16 %res
}


declare i32 @assert(...)
declare i8 @dummy_i8(i8)
declare i8 @dummy2(i8*, i8, i8)
declare i16 @dummy3(i16)
declare i16 @dummy4(i1, i8, i16)

declare dso_local i32 @e(...) local_unnamed_addr #1
declare dso_local zeroext i16 @f(...) local_unnamed_addr #1
declare dso_local arm_aapcscc i8 @__atomic_exchange_1(i8*, i8, i32) local_unnamed_addr

declare noalias i16** @func_62(i8 zeroext %p_63, i32 %p_64, i16 signext %p_65, i32* nocapture readnone %p_66)
declare fastcc signext i16 @safe_sub_func_int16_t_s_s(i16 signext %si2)
declare dso_local fastcc i64 @safe_sub_func_int64_t_s_s(i64, i64)
declare dso_local fastcc zeroext i8 @safe_lshift_func(i8 zeroext, i32)
declare dso_local fastcc zeroext i8 @safe_mul_func_uint8_t_u_u(i8 returned zeroext)
declare i1 @i1_zeroext(i8*, i32, i16 zeroext)
