; RUN: llc < %s -mtriple=avr -mcpu=atmega328 | FileCheck %s

; This test case is designed to trigger a bug caused by the register
; allocator not handling the case where a target generates a load/store with
; a frame index that happens to be an earlyclobber instruction.
;
; It is taken from the translated LLVM IR of the Rust core library.
;
; The error message looked like
; Assertion failed: (MI && "No instruction defining live value"), function computeDeadValues
;
; See PR13375 for details.

%values = type { i16, [2 x i8], [40 x i32], [0 x i8] }

@POW10TO128 = external constant [14 x i32], align 4

; CHECK: core_num_flt2dec_strategy_dragon
define nonnull dereferenceable(164) ptr @core_num_flt2dec_strategy_dragon(ptr returned dereferenceable(164) %arg, i16 %arg1) unnamed_addr {
start:
  %ret.i = alloca [40 x i32], align 4
  %tmp = icmp eq i16 undef, 0
  br i1 %tmp, label %bb5, label %bb1

bb1:
  unreachable

bb5:
  br i1 undef, label %bb14, label %bb11

bb11:
  unreachable

bb14:
  %tmp2 = bitcast ptr %ret.i to ptr
  call void @llvm.memset.p0.i16(ptr align 4 %tmp2, i8 0, i16 160, i1 false)
  %tmp3 = getelementptr inbounds %values, ptr %arg, i16 0, i32 0
  %tmp4 = load i16, ptr %tmp3, align 2
  %tmp5 = icmp ult i16 %tmp4, 14
  %tmp6 = getelementptr inbounds %values, ptr %arg, i16 0, i32 2, i16 0
  br i1 %tmp5, label %bb2.i38, label %bb3.i39

bb2.i38:
  %tmp7 = getelementptr inbounds %values, ptr %arg, i16 0, i32 2, i16 %tmp4
  %tmp8 = ptrtoint ptr %tmp6 to i16
  br label %bb4.outer.i122

bb4.outer.i122:
  %iter.sroa.0.0.ph.i119 = phi i16 [ %tmp12, %bb24.i141 ], [ %tmp8, %bb2.i38 ]
  %retsz.0.ph.i121 = phi i16 [ %.retsz.0.i140, %bb24.i141 ], [ 0, %bb2.i38 ]
  br label %bb4.i125

bb4.i125:
  %iter.sroa.0.0.i123 = phi i16 [ %tmp12, %core.iter.Enumerate.ALPHA ], [ %iter.sroa.0.0.ph.i119, %bb4.outer.i122 ]
  %tmp9 = inttoptr i16 %iter.sroa.0.0.i123 to ptr
  %tmp10 = icmp eq ptr %tmp9, %tmp7
  br i1 %tmp10, label %core.num.bignum.Big32x40.exit44, label %core.iter.Enumerate.ALPHA

core.iter.Enumerate.ALPHA:
  %tmp11 = getelementptr inbounds i32, ptr %tmp9, i16 1
  %tmp12 = ptrtoint ptr %tmp11 to i16
  %tmp13 = load i32, ptr %tmp9, align 4
  %tmp14 = icmp eq i32 %tmp13, 0
  br i1 %tmp14, label %bb4.i125, label %core..iter..Enumerate.exit17

core..iter..Enumerate.exit17:
  %tmp15 = zext i32 %tmp13 to i64
  br label %core..iter..Enumerate.exit17.i132

core..iter..Enumerate.exit17.i132:
  %carry.085.i129 = phi i32 [ 0, %core..iter..Enumerate.exit17 ], [ %tmp28, %core_slice_IndexMut.exit13 ]
  %tmp16 = icmp ult i16 undef, 40
  br i1 %tmp16, label %core_slice_IndexMut.exit13, label %panic.i.i14.i134

bb16.i133:
  %tmp17 = icmp eq i32 %tmp28, 0
  br i1 %tmp17, label %bb24.i141, label %bb21.i136

panic.i.i14.i134:
  unreachable

core_slice_IndexMut.exit13:
  %tmp18 = load i32, ptr null, align 4
  %tmp19 = getelementptr inbounds [40 x i32], ptr %ret.i, i16 0, i16 undef
  %tmp20 = load i32, ptr %tmp19, align 4
  %tmp21 = zext i32 %tmp18 to i64
  %tmp22 = mul nuw i64 %tmp21, %tmp15
  %tmp23 = zext i32 %tmp20 to i64
  %tmp24 = zext i32 %carry.085.i129 to i64
  %tmp25 = add nuw nsw i64 %tmp23, %tmp24
  %tmp26 = add i64 %tmp25, %tmp22
  %tmp27 = lshr i64 %tmp26, 32
  %tmp28 = trunc i64 %tmp27 to i32
  %tmp29 = icmp eq ptr undef, getelementptr inbounds ([14 x i32], ptr @POW10TO128, i16 1, i16 0)
  br i1 %tmp29, label %bb16.i133, label %core..iter..Enumerate.exit17.i132

bb21.i136:
  %tmp30 = icmp ult i16 undef, 40
  br i1 %tmp30, label %"_ZN4core5slice70_$LT$impl$u20$core..ops..IndexMut$LT$I$GT$$u20$for$u20$$u5b$T$u5d$$GT$9index_mut17h8eccc0af1ec6f971E.exit.i138", label %panic.i.i.i137

panic.i.i.i137:
  unreachable

"_ZN4core5slice70_$LT$impl$u20$core..ops..IndexMut$LT$I$GT$$u20$for$u20$$u5b$T$u5d$$GT$9index_mut17h8eccc0af1ec6f971E.exit.i138":
  store i32 %tmp28, ptr undef, align 4
  br label %bb24.i141

bb24.i141:
  %sz.0.i139 = phi i16 [ 15, %"_ZN4core5slice70_$LT$impl$u20$core..ops..IndexMut$LT$I$GT$$u20$for$u20$$u5b$T$u5d$$GT$9index_mut17h8eccc0af1ec6f971E.exit.i138" ], [ 14, %bb16.i133 ]
  %tmp31 = add i16 %sz.0.i139, 0
  %tmp32 = icmp ult i16 %retsz.0.ph.i121, %tmp31
  %.retsz.0.i140 = select i1 %tmp32, i16 %tmp31, i16 %retsz.0.ph.i121
  br label %bb4.outer.i122

bb3.i39:
  %tmp33 = call fastcc i16 @_ZN4core3num6bignum8Big32x4010mul_digits9mul_inner17h5d3461bce04d16ccE(ptr nonnull dereferenceable(160) %ret.i, ptr noalias nonnull readonly @POW10TO128, i16 14, ptr noalias nonnull readonly %tmp6, i16 %tmp4)
  br label %core.num.bignum.Big32x40.exit44

core.num.bignum.Big32x40.exit44:
  %retsz.0.i40 = phi i16 [ %tmp33, %bb3.i39 ], [ %retsz.0.ph.i121, %bb4.i125 ]
  call void @llvm.memcpy.p0.p0.i16(ptr align 4 undef, ptr align 4 %tmp2, i16 160, i1 false)
  store i16 %retsz.0.i40, ptr %tmp3, align 2
  %tmp34 = and i16 %arg1, 256
  %tmp35 = icmp eq i16 %tmp34, 0
  br i1 %tmp35, label %bb30, label %bb27

bb27:
  unreachable

bb30:
  ret ptr %arg
}

declare fastcc i16 @_ZN4core3num6bignum8Big32x4010mul_digits9mul_inner17h5d3461bce04d16ccE(ptr nocapture dereferenceable(160), ptr noalias nonnull readonly, i16, ptr noalias nonnull readonly, i16)

declare void @llvm.memset.p0.i16(ptr nocapture writeonly, i8, i16, i1)

declare void @llvm.memcpy.p0.p0.i16(ptr nocapture writeonly, ptr nocapture readonly, i16, i1)

