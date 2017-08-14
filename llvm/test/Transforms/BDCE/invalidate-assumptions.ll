; RUN: opt -bdce %s -S | FileCheck %s

; The 'nuw' on the subtract allows us to deduce that %setbit is not demanded.
; But if we change that value to '0', then the 'nuw' is no longer valid. If we don't
; remove the 'nuw', another pass (-instcombine) may make a transform based on an
; that incorrect assumption and we can miscompile:
; https://bugs.llvm.org/show_bug.cgi?id=33695

define i1 @PR33695(i1 %b, i8 %x) {
; CHECK-LABEL: @PR33695(
; CHECK-NEXT:    [[SETBIT:%.*]] = or i8 %x, 64
; CHECK-NEXT:    [[LITTLE_NUMBER:%.*]] = zext i1 %b to i8
; CHECK-NEXT:    [[BIG_NUMBER:%.*]] = shl i8 0, 1
; CHECK-NEXT:    [[SUB:%.*]] = sub i8 [[BIG_NUMBER]], [[LITTLE_NUMBER]]
; CHECK-NEXT:    [[TRUNC:%.*]] = trunc i8 [[SUB]] to i1
; CHECK-NEXT:    ret i1 [[TRUNC]]
;
  %setbit = or i8 %x, 64
  %little_number = zext i1 %b to i8
  %big_number = shl i8 %setbit, 1
  %sub = sub nuw i8 %big_number, %little_number
  %trunc = trunc i8 %sub to i1
  ret i1 %trunc
}

; Similar to above, but now with more no-wrap.
; https://bugs.llvm.org/show_bug.cgi?id=34037

define i64 @PR34037(i64 %m, i32 %r, i64 %j, i1 %b, i32 %k, i64 %p) {
; CHECK-LABEL: @PR34037(
; CHECK-NEXT:    [[CONV:%.*]] = zext i32 %r to i64
; CHECK-NEXT:    [[AND:%.*]] = and i64 %m, 0
; CHECK-NEXT:    [[NEG:%.*]] = xor i64 0, 34359738367
; CHECK-NEXT:    [[OR:%.*]] = or i64 %j, 0
; CHECK-NEXT:    [[SHL:%.*]] = shl i64 0, 29
; CHECK-NEXT:    [[CONV1:%.*]] = select i1 %b, i64 7, i64 0
; CHECK-NEXT:    [[SUB:%.*]] = sub i64 [[SHL]], [[CONV1]]
; CHECK-NEXT:    [[CONV2:%.*]] = zext i32 %k to i64
; CHECK-NEXT:    [[MUL:%.*]] = mul i64 [[SUB]], [[CONV2]]
; CHECK-NEXT:    [[CONV4:%.*]] = and i64 %p, 65535
; CHECK-NEXT:    [[AND5:%.*]] = and i64 [[MUL]], [[CONV4]]
; CHECK-NEXT:    ret i64 [[AND5]]
;
  %conv = zext i32 %r to i64
  %and = and i64 %m, %conv
  %neg = xor i64 %and, 34359738367
  %or = or i64 %j, %neg
  %shl = shl i64 %or, 29
  %conv1 = select i1 %b, i64 7, i64 0
  %sub = sub nuw nsw i64 %shl, %conv1
  %conv2 = zext i32 %k to i64
  %mul = mul nsw i64 %sub, %conv2
  %conv4 = and i64 %p, 65535
  %and5 = and i64 %mul, %conv4
  ret i64 %and5
}

; This is a manufactured example based on the 1st test to prove that the
; assumption-killing algorithm stops at the call. Ie, it does not remove
; nsw/nuw from the 'add' because a call demands all bits of its argument.

declare i1 @foo(i1)

define i1 @poison_on_call_user_is_ok(i1 %b, i8 %x) {
; CHECK-LABEL: @poison_on_call_user_is_ok(
; CHECK-NEXT:    [[SETBIT:%.*]] = or i8 %x, 64
; CHECK-NEXT:    [[LITTLE_NUMBER:%.*]] = zext i1 %b to i8
; CHECK-NEXT:    [[BIG_NUMBER:%.*]] = shl i8 0, 1
; CHECK-NEXT:    [[SUB:%.*]] = sub i8 [[BIG_NUMBER]], [[LITTLE_NUMBER]]
; CHECK-NEXT:    [[TRUNC:%.*]] = trunc i8 [[SUB]] to i1
; CHECK-NEXT:    [[CALL_RESULT:%.*]] = call i1 @foo(i1 [[TRUNC]])
; CHECK-NEXT:    [[ADD:%.*]] = add nuw nsw i1 [[CALL_RESULT]], true
; CHECK-NEXT:    [[MUL:%.*]] = mul i1 [[TRUNC]], [[ADD]]
; CHECK-NEXT:    ret i1 [[MUL]]
;
  %setbit = or i8 %x, 64
  %little_number = zext i1 %b to i8
  %big_number = shl i8 %setbit, 1
  %sub = sub nuw i8 %big_number, %little_number
  %trunc = trunc i8 %sub to i1
  %call_result = call i1 @foo(i1 %trunc)
  %add = add nsw nuw i1 %call_result, 1
  %mul = mul i1 %trunc, %add
  ret i1 %mul
}


; We were asserting that all users of a trivialized integer-type instruction were
; also integer-typed, but that's too strong. The alloca has a pointer-type result.

define void @PR34179(i32* %a) {
; CHECK-LABEL: @PR34179(
; CHECK-NEXT:    [[T0:%.*]] = load volatile i32, i32* %a
; CHECK-NEXT:    ret void
;
  %t0 = load volatile i32, i32* %a
  %vla = alloca i32, i32 %t0
  ret void
}

