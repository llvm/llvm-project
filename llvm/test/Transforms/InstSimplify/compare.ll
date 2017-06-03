; RUN: opt < %s -instsimplify -S | FileCheck %s
target datalayout = "p:32:32"

define i1 @ptrtoint() {
; CHECK-LABEL: @ptrtoint(
  %a = alloca i8
  %tmp = ptrtoint i8* %a to i32
  %r = icmp eq i32 %tmp, 0
  ret i1 %r
; CHECK: ret i1 false
}

define i1 @bitcast() {
; CHECK-LABEL: @bitcast(
  %a = alloca i32
  %b = alloca i64
  %x = bitcast i32* %a to i8*
  %y = bitcast i64* %b to i8*
  %cmp = icmp eq i8* %x, %y
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @gep() {
; CHECK-LABEL: @gep(
  %a = alloca [3 x i8], align 8
  %x = getelementptr inbounds [3 x i8], [3 x i8]* %a, i32 0, i32 0
  %cmp = icmp eq i8* %x, null
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @gep2() {
; CHECK-LABEL: @gep2(
  %a = alloca [3 x i8], align 8
  %x = getelementptr inbounds [3 x i8], [3 x i8]* %a, i32 0, i32 0
  %y = getelementptr inbounds [3 x i8], [3 x i8]* %a, i32 0, i32 0
  %cmp = icmp eq i8* %x, %y
  ret i1 %cmp
; CHECK-NEXT: ret i1 true
}

; PR11238
%gept = type { i32, i32 }
@gepy = global %gept zeroinitializer, align 8
@gepz = extern_weak global %gept

define i1 @gep3() {
; CHECK-LABEL: @gep3(
  %x = alloca %gept, align 8
  %a = getelementptr %gept, %gept* %x, i64 0, i32 0
  %b = getelementptr %gept, %gept* %x, i64 0, i32 1
  %equal = icmp eq i32* %a, %b
  ret i1 %equal
; CHECK-NEXT: ret i1 false
}

define i1 @gep4() {
; CHECK-LABEL: @gep4(
  %x = alloca %gept, align 8
  %a = getelementptr %gept, %gept* @gepy, i64 0, i32 0
  %b = getelementptr %gept, %gept* @gepy, i64 0, i32 1
  %equal = icmp eq i32* %a, %b
  ret i1 %equal
; CHECK-NEXT: ret i1 false
}

@a = common global [1 x i32] zeroinitializer, align 4

define i1 @PR31262() {
; CHECK-LABEL: @PR31262(
; CHECK-NEXT:    ret i1 icmp uge (i32* getelementptr ([1 x i32], [1 x i32]* @a, i64 0, i64 undef), i32* getelementptr inbounds ([1 x i32], [1 x i32]* @a, i32 0, i32 0))
;
  %idx = getelementptr inbounds [1 x i32], [1 x i32]* @a, i64 0, i64 undef
  %cmp = icmp uge i32* %idx, getelementptr inbounds ([1 x i32], [1 x i32]* @a, i32 0, i32 0)
  ret i1 %cmp
}

define i1 @gep5() {
; CHECK-LABEL: @gep5(
  %x = alloca %gept, align 8
  %a = getelementptr inbounds %gept, %gept* %x, i64 0, i32 1
  %b = getelementptr %gept, %gept* @gepy, i64 0, i32 0
  %equal = icmp eq i32* %a, %b
  ret i1 %equal
; CHECK-NEXT: ret i1 false
}

define i1 @gep6(%gept* %x) {
; Same as @gep3 but potentially null.
; CHECK-LABEL: @gep6(
  %a = getelementptr %gept, %gept* %x, i64 0, i32 0
  %b = getelementptr %gept, %gept* %x, i64 0, i32 1
  %equal = icmp eq i32* %a, %b
  ret i1 %equal
; CHECK-NEXT: ret i1 false
}

define i1 @gep7(%gept* %x) {
; CHECK-LABEL: @gep7(
  %a = getelementptr %gept, %gept* %x, i64 0, i32 0
  %b = getelementptr %gept, %gept* @gepz, i64 0, i32 0
  %equal = icmp eq i32* %a, %b
  ret i1 %equal
; CHECK: ret i1 %equal
}

define i1 @gep8(%gept* %x) {
; CHECK-LABEL: @gep8(
  %a = getelementptr %gept, %gept* %x, i32 1
  %b = getelementptr %gept, %gept* %x, i32 -1
  %equal = icmp ugt %gept* %a, %b
  ret i1 %equal
; CHECK: ret i1 %equal
}

define i1 @gep9(i8* %ptr) {
; CHECK-LABEL: @gep9(
; CHECK-NOT: ret
; CHECK: ret i1 true

entry:
  %first1 = getelementptr inbounds i8, i8* %ptr, i32 0
  %first2 = getelementptr inbounds i8, i8* %first1, i32 1
  %first3 = getelementptr inbounds i8, i8* %first2, i32 2
  %first4 = getelementptr inbounds i8, i8* %first3, i32 4
  %last1 = getelementptr inbounds i8, i8* %first2, i32 48
  %last2 = getelementptr inbounds i8, i8* %last1, i32 8
  %last3 = getelementptr inbounds i8, i8* %last2, i32 -4
  %last4 = getelementptr inbounds i8, i8* %last3, i32 -4
  %first.int = ptrtoint i8* %first4 to i32
  %last.int = ptrtoint i8* %last4 to i32
  %cmp = icmp ne i32 %last.int, %first.int
  ret i1 %cmp
}

define i1 @gep10(i8* %ptr) {
; CHECK-LABEL: @gep10(
; CHECK-NOT: ret
; CHECK: ret i1 true

entry:
  %first1 = getelementptr inbounds i8, i8* %ptr, i32 -2
  %first2 = getelementptr inbounds i8, i8* %first1, i32 44
  %last1 = getelementptr inbounds i8, i8* %ptr, i32 48
  %last2 = getelementptr inbounds i8, i8* %last1, i32 -6
  %first.int = ptrtoint i8* %first2 to i32
  %last.int = ptrtoint i8* %last2 to i32
  %cmp = icmp eq i32 %last.int, %first.int
  ret i1 %cmp
}

define i1 @gep11(i8* %ptr) {
; CHECK-LABEL: @gep11(
; CHECK-NOT: ret
; CHECK: ret i1 true

entry:
  %first1 = getelementptr inbounds i8, i8* %ptr, i32 -2
  %last1 = getelementptr inbounds i8, i8* %ptr, i32 48
  %last2 = getelementptr inbounds i8, i8* %last1, i32 -6
  %cmp = icmp ult i8* %first1, %last2
  ret i1 %cmp
}

define i1 @gep12(i8* %ptr) {
; CHECK-LABEL: @gep12(
; CHECK-NOT: ret
; CHECK: ret i1 %cmp

entry:
  %first1 = getelementptr inbounds i8, i8* %ptr, i32 -2
  %last1 = getelementptr inbounds i8, i8* %ptr, i32 48
  %last2 = getelementptr inbounds i8, i8* %last1, i32 -6
  %cmp = icmp slt i8* %first1, %last2
  ret i1 %cmp
}

define i1 @gep13(i8* %ptr) {
; CHECK-LABEL: @gep13(
; We can prove this GEP is non-null because it is inbounds.
  %x = getelementptr inbounds i8, i8* %ptr, i32 1
  %cmp = icmp eq i8* %x, null
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @gep14({ {}, i8 }* %ptr) {
; CHECK-LABEL: @gep14(
; We can't simplify this because the offset of one in the GEP actually doesn't
; move the pointer.
  %x = getelementptr inbounds { {}, i8 }, { {}, i8 }* %ptr, i32 0, i32 1
  %cmp = icmp eq i8* %x, null
  ret i1 %cmp
; CHECK-NOT: ret i1 false
}

define i1 @gep15({ {}, [4 x {i8, i8}]}* %ptr, i32 %y) {
; CHECK-LABEL: @gep15(
; We can prove this GEP is non-null even though there is a user value, as we
; would necessarily violate inbounds on one side or the other.
  %x = getelementptr inbounds { {}, [4 x {i8, i8}]}, { {}, [4 x {i8, i8}]}* %ptr, i32 0, i32 1, i32 %y, i32 1
  %cmp = icmp eq i8* %x, null
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @gep16(i8* %ptr, i32 %a) {
; CHECK-LABEL: @gep16(
; We can prove this GEP is non-null because it is inbounds and because we know
; %b is non-zero even though we don't know its value.
  %b = or i32 %a, 1
  %x = getelementptr inbounds i8, i8* %ptr, i32 %b
  %cmp = icmp eq i8* %x, null
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @gep17() {
; CHECK-LABEL: @gep17(
  %alloca = alloca i32, align 4
  %bc = bitcast i32* %alloca to [4 x i8]*
  %gep1 = getelementptr inbounds i32, i32* %alloca, i32 1
  %pti1 = ptrtoint i32* %gep1 to i32
  %gep2 = getelementptr inbounds [4 x i8], [4 x i8]* %bc, i32 0, i32 1
  %pti2 = ptrtoint i8* %gep2 to i32
  %cmp = icmp ugt i32 %pti1, %pti2
  ret i1 %cmp
; CHECK-NEXT: ret i1 true
}

define i1 @zext(i32 %x) {
; CHECK-LABEL: @zext(
  %e1 = zext i32 %x to i64
  %e2 = zext i32 %x to i64
  %r = icmp eq i64 %e1, %e2
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @zext2(i1 %x) {
; CHECK-LABEL: @zext2(
  %e = zext i1 %x to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 %x
}

define i1 @zext3() {
; CHECK-LABEL: @zext3(
  %e = zext i1 1 to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @sext(i32 %x) {
; CHECK-LABEL: @sext(
  %e1 = sext i32 %x to i64
  %e2 = sext i32 %x to i64
  %r = icmp eq i64 %e1, %e2
  ret i1 %r
; CHECK: ret i1 true
}

define i1 @sext2(i1 %x) {
; CHECK-LABEL: @sext2(
  %e = sext i1 %x to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 %x
}

define i1 @sext3() {
; CHECK-LABEL: @sext3(
  %e = sext i1 1 to i32
  %c = icmp ne i32 %e, 0
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @add(i32 %x, i32 %y) {
; CHECK-LABEL: @add(
  %l = lshr i32 %x, 1
  %q = lshr i32 %y, 1
  %r = or i32 %q, 1
  %s = add i32 %l, %r
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @add2(i8 %x, i8 %y) {
; CHECK-LABEL: @add2(
  %l = or i8 %x, 128
  %r = or i8 %y, 129
  %s = add i8 %l, %r
  %c = icmp eq i8 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @add3(i8 %x, i8 %y) {
; CHECK-LABEL: @add3(
  %l = zext i8 %x to i32
  %r = zext i8 %y to i32
  %s = add i32 %l, %r
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %c
}

define i1 @add4(i32 %x, i32 %y) {
; CHECK-LABEL: @add4(
  %z = add nsw i32 %y, 1
  %s1 = add nsw i32 %x, %y
  %s2 = add nsw i32 %x, %z
  %c = icmp slt i32 %s1, %s2
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @add5(i32 %x, i32 %y) {
; CHECK-LABEL: @add5(
  %z = add nuw i32 %y, 1
  %s1 = add nuw i32 %x, %z
  %s2 = add nuw i32 %x, %y
  %c = icmp ugt i32 %s1, %s2
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @add6(i64 %A, i64 %B) {
; CHECK-LABEL: @add6(
  %s1 = add i64 %A, %B
  %s2 = add i64 %B, %A
  %cmp = icmp eq i64 %s1, %s2
  ret i1 %cmp
; CHECK: ret i1 true
}

define i1 @addpowtwo(i32 %x, i32 %y) {
; CHECK-LABEL: @addpowtwo(
  %l = lshr i32 %x, 1
  %r = shl i32 1, %y
  %s = add i32 %l, %r
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @or(i32 %x) {
; CHECK-LABEL: @or(
  %o = or i32 %x, 1
  %c = icmp eq i32 %o, 0
  ret i1 %c
; CHECK: ret i1 false
}

; Do not simplify if we cannot guarantee that the ConstantExpr is a non-zero
; constant.
@GV = common global i32* null
define i1 @or_constexp(i32 %x) {
; CHECK-LABEL: @or_constexp(
entry:
  %0 = and i32 ptrtoint (i32** @GV to i32), 32
  %o = or i32 %x, %0
  %c = icmp eq i32 %o, 0
  ret i1 %c
; CHECK: or
; CHECK-NEXT: icmp eq
; CHECK-NOT: ret i1 false
}

define i1 @shl1(i32 %x) {
; CHECK-LABEL: @shl1(
  %s = shl i32 1, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @shl3(i32 %X) {
; CHECK: @shl3
  %sub = shl nuw i32 4, %X
  %cmp = icmp eq i32 %sub, 31
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @lshr1(i32 %x) {
; CHECK-LABEL: @lshr1(
  %s = lshr i32 -1, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @lshr3(i32 %x) {
; CHECK-LABEL: @lshr3(
  %s = lshr i32 %x, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @lshr4(i32 %X, i32 %Y) {
; CHECK-LABEL: @lshr4(
  %A = lshr i32 %X, %Y
  %C = icmp ule i32 %A, %X
  ret i1 %C
; CHECK: ret i1 true
}

define i1 @lshr5(i32 %X, i32 %Y) {
; CHECK-LABEL: @lshr5(
  %A = lshr i32 %X, %Y
  %C = icmp ugt i32 %A, %X
  ret i1 %C
; CHECK: ret i1 false
}

define i1 @lshr6(i32 %X, i32 %Y) {
; CHECK-LABEL: @lshr6(
  %A = lshr i32 %X, %Y
  %C = icmp ult i32 %X, %A
  ret i1 %C
; CHECK: ret i1 false
}

define i1 @lshr7(i32 %X, i32 %Y) {
; CHECK-LABEL: @lshr7(
  %A = lshr i32 %X, %Y
  %C = icmp uge i32 %X, %A
  ret i1 %C
; CHECK: ret i1 true
}

define i1 @ashr1(i32 %x) {
; CHECK-LABEL: @ashr1(
  %s = ashr i32 -1, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 false
}

define i1 @ashr3(i32 %x) {
; CHECK-LABEL: @ashr3(
  %s = ashr i32 %x, %x
  %c = icmp eq i32 %s, 0
  ret i1 %c
; CHECK: ret i1 true
}

define i1 @select1(i1 %cond) {
; CHECK-LABEL: @select1(
  %s = select i1 %cond, i32 1, i32 0
  %c = icmp eq i32 %s, 1
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select2(i1 %cond) {
; CHECK-LABEL: @select2(
  %x = zext i1 %cond to i32
  %s = select i1 %cond, i32 %x, i32 0
  %c = icmp ne i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select3(i1 %cond) {
; CHECK-LABEL: @select3(
  %x = zext i1 %cond to i32
  %s = select i1 %cond, i32 1, i32 %x
  %c = icmp ne i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select4(i1 %cond) {
; CHECK-LABEL: @select4(
  %invert = xor i1 %cond, 1
  %s = select i1 %invert, i32 0, i32 1
  %c = icmp ne i32 %s, 0
  ret i1 %c
; CHECK: ret i1 %cond
}

define i1 @select5(i32 %x) {
; CHECK-LABEL: @select5(
  %c = icmp eq i32 %x, 0
  %s = select i1 %c, i32 1, i32 %x
  %c2 = icmp eq i32 %s, 0
  ret i1 %c2
; CHECK: ret i1 false
}

define i1 @select6(i32 %x) {
; CHECK-LABEL: @select6(
  %c = icmp sgt i32 %x, 0
  %s = select i1 %c, i32 %x, i32 4
  %c2 = icmp eq i32 %s, 0
  ret i1 %c2
; CHECK: ret i1 %c2
}

define i1 @urem1(i32 %X, i32 %Y) {
; CHECK-LABEL: @urem1(
  %A = urem i32 %X, %Y
  %B = icmp ult i32 %A, %Y
  ret i1 %B
; CHECK: ret i1 true
}

define i1 @urem2(i32 %X, i32 %Y) {
; CHECK-LABEL: @urem2(
  %A = urem i32 %X, %Y
  %B = icmp eq i32 %A, %Y
  ret i1 %B
; CHECK: ret i1 false
}

define i1 @urem4(i32 %X) {
; CHECK-LABEL: @urem4(
  %A = urem i32 %X, 15
  %B = icmp ult i32 %A, 10
  ret i1 %B
; CHECK: ret i1 %B
}

define i1 @urem5(i16 %X, i32 %Y) {
; CHECK-LABEL: @urem5(
  %A = zext i16 %X to i32
  %B = urem i32 %A, %Y
  %C = icmp slt i32 %B, %Y
  ret i1 %C
; CHECK-NOT: ret i1 true
}

define i1 @urem6(i32 %X, i32 %Y) {
; CHECK-LABEL: @urem6(
  %A = urem i32 %X, %Y
  %B = icmp ugt i32 %Y, %A
  ret i1 %B
; CHECK: ret i1 true
}

define i1 @urem7(i32 %X) {
; CHECK-LABEL: @urem7(
  %A = urem i32 1, %X
  %B = icmp sgt i32 %A, %X
  ret i1 %B
; CHECK-NOT: ret i1 false
}

; PR9343 #15
; CHECK-LABEL: @srem2(
; CHECK: ret i1 false
define i1 @srem2(i16 %X, i32 %Y) {
  %A = zext i16 %X to i32
  %B = add nsw i32 %A, 1
  %C = srem i32 %B, %Y
  %D = icmp slt i32 %C, 0
  ret i1 %D
}

; CHECK-LABEL: @srem3(
; CHECK-NEXT: ret i1 false
define i1 @srem3(i16 %X, i32 %Y) {
  %A = zext i16 %X to i32
  %B = or i32 2147483648, %A
  %C = sub nsw i32 1, %B
  %D = srem i32 %C, %Y
  %E = icmp slt i32 %D, 0
  ret i1 %E
}

define i1 @udiv2(i32 %Z) {
; CHECK-LABEL: @udiv2(
; CHECK-NEXT:    ret i1 true
;
  %A = udiv exact i32 10, %Z
  %B = udiv exact i32 20, %Z
  %C = icmp ult i32 %A, %B
  ret i1 %C
}

; Exact sdiv and equality preds can simplify.

define i1 @sdiv_exact_equality(i32 %Z) {
; CHECK-LABEL: @sdiv_exact_equality(
; CHECK-NEXT:    ret i1 false
;
  %A = sdiv exact i32 10, %Z
  %B = sdiv exact i32 20, %Z
  %C = icmp eq i32 %A, %B
  ret i1 %C
}

; But not other preds: PR32949 - https://bugs.llvm.org/show_bug.cgi?id=32949

define i1 @sdiv_exact_not_equality(i32 %Z) {
; CHECK-LABEL: @sdiv_exact_not_equality(
; CHECK-NEXT:    [[A:%.*]] = sdiv exact i32 10, %Z
; CHECK-NEXT:    [[B:%.*]] = sdiv exact i32 20, %Z
; CHECK-NEXT:    [[C:%.*]] = icmp ult i32 [[A]], [[B]]
; CHECK-NEXT:    ret i1 [[C]]
;
  %A = sdiv exact i32 10, %Z
  %B = sdiv exact i32 20, %Z
  %C = icmp ult i32 %A, %B
  ret i1 %C
}

define i1 @udiv3(i32 %X, i32 %Y) {
; CHECK-LABEL: @udiv3(
  %A = udiv i32 %X, %Y
  %C = icmp ugt i32 %A, %X
  ret i1 %C
; CHECK: ret i1 false
}

define i1 @udiv4(i32 %X, i32 %Y) {
; CHECK-LABEL: @udiv4(
  %A = udiv i32 %X, %Y
  %C = icmp ule i32 %A, %X
  ret i1 %C
; CHECK: ret i1 true
}

; PR11340
define i1 @udiv6(i32 %X) nounwind {
; CHECK-LABEL: @udiv6(
  %A = udiv i32 1, %X
  %C = icmp eq i32 %A, 0
  ret i1 %C
; CHECK: ret i1 %C
}

define i1 @udiv7(i32 %X, i32 %Y) {
; CHECK-LABEL: @udiv7(
  %A = udiv i32 %X, %Y
  %C = icmp ult i32 %X, %A
  ret i1 %C
; CHECK: ret i1 false
}

define i1 @udiv8(i32 %X, i32 %Y) {
; CHECK-LABEL: @udiv8(
  %A = udiv i32 %X, %Y
  %C = icmp uge i32 %X, %A
  ret i1 %C
; CHECK: ret i1 true
}

define i1 @mul1(i32 %X) {
; CHECK-LABEL: @mul1(
; Square of a non-zero number is non-zero if there is no overflow.
  %Y = or i32 %X, 1
  %M = mul nuw i32 %Y, %Y
  %C = icmp eq i32 %M, 0
  ret i1 %C
; CHECK: ret i1 false
}

define i1 @mul2(i32 %X) {
; CHECK-LABEL: @mul2(
; Square of a non-zero number is positive if there is no signed overflow.
  %Y = or i32 %X, 1
  %M = mul nsw i32 %Y, %Y
  %C = icmp sgt i32 %M, 0
  ret i1 %C
; CHECK: ret i1 true
}

define i1 @mul3(i32 %X, i32 %Y) {
; CHECK-LABEL: @mul3(
; Product of non-negative numbers is non-negative if there is no signed overflow.
  %XX = mul nsw i32 %X, %X
  %YY = mul nsw i32 %Y, %Y
  %M = mul nsw i32 %XX, %YY
  %C = icmp sge i32 %M, 0
  ret i1 %C
; CHECK: ret i1 true
}

define <2 x i1> @vectorselect1(<2 x i1> %cond) {
; CHECK-LABEL: @vectorselect1(
  %invert = xor <2 x i1> %cond, <i1 1, i1 1>
  %s = select <2 x i1> %invert, <2 x i32> <i32 0, i32 0>, <2 x i32> <i32 1, i32 1>
  %c = icmp ne <2 x i32> %s, <i32 0, i32 0>
  ret <2 x i1> %c
; CHECK: ret <2 x i1> %cond
}

; PR11948
define <2 x i1> @vectorselectcrash(i32 %arg1) {
  %tobool40 = icmp ne i32 %arg1, 0
  %cond43 = select i1 %tobool40, <2 x i16> <i16 -5, i16 66>, <2 x i16> <i16 46, i16 1>
  %cmp45 = icmp ugt <2 x i16> %cond43, <i16 73, i16 21>
  ret <2 x i1> %cmp45
}

; PR12013
define i1 @alloca_compare(i64 %idx) {
  %sv = alloca { i32, i32, [124 x i32] }
  %1 = getelementptr inbounds { i32, i32, [124 x i32] }, { i32, i32, [124 x i32] }* %sv, i32 0, i32 2, i64 %idx
  %2 = icmp eq i32* %1, null
  ret i1 %2
  ; CHECK: alloca_compare
  ; CHECK: ret i1 false
}

; PR12075
define i1 @infinite_gep() {
  ret i1 1

unreachableblock:
  %X = getelementptr i32, i32 *%X, i32 1
  %Y = icmp eq i32* %X, null
  ret i1 %Y
}

; It's not valid to fold a comparison of an argument with an alloca, even though
; that's tempting. An argument can't *alias* an alloca, however the aliasing rule
; relies on restrictions against guessing an object's address and dereferencing.
; There are no restrictions against guessing an object's address and comparing.

define i1 @alloca_argument_compare(i64* %arg) {
  %alloc = alloca i64
  %cmp = icmp eq i64* %arg, %alloc
  ret i1 %cmp
  ; CHECK: alloca_argument_compare
  ; CHECK: ret i1 %cmp
}

; As above, but with the operands reversed.

define i1 @alloca_argument_compare_swapped(i64* %arg) {
  %alloc = alloca i64
  %cmp = icmp eq i64* %alloc, %arg
  ret i1 %cmp
  ; CHECK: alloca_argument_compare_swapped
  ; CHECK: ret i1 %cmp
}

; Don't assume that a noalias argument isn't equal to a global variable's
; address. This is an example where AliasAnalysis' NoAlias concept is
; different from actual pointer inequality.

@y = external global i32
define zeroext i1 @external_compare(i32* noalias %x) {
  %cmp = icmp eq i32* %x, @y
  ret i1 %cmp
  ; CHECK: external_compare
  ; CHECK: ret i1 %cmp
}

define i1 @alloca_gep(i64 %a, i64 %b) {
; CHECK-LABEL: @alloca_gep(
; We can prove this GEP is non-null because it is inbounds and the pointer
; is non-null.
  %strs = alloca [1000 x [1001 x i8]], align 16
  %x = getelementptr inbounds [1000 x [1001 x i8]], [1000 x [1001 x i8]]* %strs, i64 0, i64 %a, i64 %b
  %cmp = icmp eq i8* %x, null
  ret i1 %cmp
; CHECK-NEXT: ret i1 false
}

define i1 @non_inbounds_gep_compare(i64* %a) {
; CHECK-LABEL: @non_inbounds_gep_compare(
; Equality compares with non-inbounds GEPs can be folded.
  %x = getelementptr i64, i64* %a, i64 42
  %y = getelementptr inbounds i64, i64* %x, i64 -42
  %z = getelementptr i64, i64* %a, i64 -42
  %w = getelementptr inbounds i64, i64* %z, i64 42
  %cmp = icmp eq i64* %y, %w
  ret i1 %cmp
; CHECK-NEXT: ret i1 true
}

define i1 @non_inbounds_gep_compare2(i64* %a) {
; CHECK-LABEL: @non_inbounds_gep_compare2(
; Equality compares with non-inbounds GEPs can be folded.
  %x = getelementptr i64, i64* %a, i64 4294967297
  %y = getelementptr i64, i64* %a, i64 1
  %cmp = icmp eq i64* %y, %y
  ret i1 %cmp
; CHECK-NEXT: ret i1 true
}

define <4 x i8> @vectorselectfold(<4 x i8> %a, <4 x i8> %b) {
  %false = icmp ne <4 x i8> zeroinitializer, zeroinitializer
  %sel = select <4 x i1> %false, <4 x i8> %a, <4 x i8> %b
  ret <4 x i8> %sel

; CHECK-LABEL: @vectorselectfold
; CHECK-NEXT: ret <4 x i8> %b
}

define <4 x i8> @vectorselectfold2(<4 x i8> %a, <4 x i8> %b) {
  %true = icmp eq <4 x i8> zeroinitializer, zeroinitializer
  %sel = select <4 x i1> %true, <4 x i8> %a, <4 x i8> %b
  ret <4 x i8> %sel

; CHECK-LABEL: @vectorselectfold
; CHECK-NEXT: ret <4 x i8> %a
}

define i1 @compare_always_true_slt(i16 %a) {
  %1 = zext i16 %a to i32
  %2 = sub nsw i32 0, %1
  %3 = icmp slt i32 %2, 1
  ret i1 %3

; CHECK-LABEL: @compare_always_true_slt
; CHECK-NEXT: ret i1 true
}

define i1 @compare_always_true_sle(i16 %a) {
  %1 = zext i16 %a to i32
  %2 = sub nsw i32 0, %1
  %3 = icmp sle i32 %2, 0
  ret i1 %3

; CHECK-LABEL: @compare_always_true_sle
; CHECK-NEXT: ret i1 true
}

define i1 @compare_always_false_sgt(i16 %a) {
  %1 = zext i16 %a to i32
  %2 = sub nsw i32 0, %1
  %3 = icmp sgt i32 %2, 0
  ret i1 %3

; CHECK-LABEL: @compare_always_false_sgt
; CHECK-NEXT: ret i1 false
}

define i1 @compare_always_false_sge(i16 %a) {
  %1 = zext i16 %a to i32
  %2 = sub nsw i32 0, %1
  %3 = icmp sge i32 %2, 1
  ret i1 %3

; CHECK-LABEL: @compare_always_false_sge
; CHECK-NEXT: ret i1 false
}

define i1 @compare_always_false_eq(i16 %a) {
  %1 = zext i16 %a to i32
  %2 = sub nsw i32 0, %1
  %3 = icmp eq i32 %2, 1
  ret i1 %3

; CHECK-LABEL: @compare_always_false_eq
; CHECK-NEXT: ret i1 false
}

define i1 @compare_always_false_ne(i16 %a) {
  %1 = zext i16 %a to i32
  %2 = sub nsw i32 0, %1
  %3 = icmp ne i32 %2, 1
  ret i1 %3

; CHECK-LABEL: @compare_always_false_ne
; CHECK-NEXT: ret i1 true
}

define i1 @lshr_ugt_false(i32 %a) {
  %shr = lshr i32 1, %a
  %cmp = icmp ugt i32 %shr, 1
  ret i1 %cmp
; CHECK-LABEL: @lshr_ugt_false
; CHECK-NEXT: ret i1 false
}

define i1 @nonnull_arg(i32* nonnull %i) {
  %cmp = icmp eq i32* %i, null
  ret i1 %cmp
; CHECK-LABEL: @nonnull_arg
; CHECK: ret i1 false
}

define i1 @nonnull_deref_arg(i32* dereferenceable(4) %i) {
  %cmp = icmp eq i32* %i, null
  ret i1 %cmp
; CHECK-LABEL: @nonnull_deref_arg
; CHECK: ret i1 false
}

define i1 @nonnull_deref_as_arg(i32 addrspace(1)* dereferenceable(4) %i) {
  %cmp = icmp eq i32 addrspace(1)* %i, null
  ret i1 %cmp
; CHECK-LABEL: @nonnull_deref_as_arg
; CHECK: icmp
; CHECK: ret
}

declare nonnull i32* @returns_nonnull_helper()
define i1 @returns_nonnull() {
  %call = call nonnull i32* @returns_nonnull_helper()
  %cmp = icmp eq i32* %call, null
  ret i1 %cmp
; CHECK-LABEL: @returns_nonnull
; CHECK: ret i1 false
}

declare dereferenceable(4) i32* @returns_nonnull_deref_helper()
define i1 @returns_nonnull_deref() {
  %call = call dereferenceable(4) i32* @returns_nonnull_deref_helper()
  %cmp = icmp eq i32* %call, null
  ret i1 %cmp
; CHECK-LABEL: @returns_nonnull_deref
; CHECK: ret i1 false
}

declare dereferenceable(4) i32 addrspace(1)* @returns_nonnull_deref_as_helper()
define i1 @returns_nonnull_as_deref() {
  %call = call dereferenceable(4) i32 addrspace(1)* @returns_nonnull_deref_as_helper()
  %cmp = icmp eq i32 addrspace(1)* %call, null
  ret i1 %cmp
; CHECK-LABEL: @returns_nonnull_as_deref
; CHECK: icmp
; CHECK: ret
}

define i1 @nonnull_load(i32** %addr) {
  %ptr = load i32*, i32** %addr, !nonnull !{}
  %cmp = icmp eq i32* %ptr, null
  ret i1 %cmp
; CHECK-LABEL: @nonnull_load
; CHECK: ret i1 false
}

define i1 @nonnull_load_as_outer(i32* addrspace(1)* %addr) {
  %ptr = load i32*, i32* addrspace(1)* %addr, !nonnull !{}
  %cmp = icmp eq i32* %ptr, null
  ret i1 %cmp
; CHECK-LABEL: @nonnull_load_as_outer
; CHECK: ret i1 false
}
define i1 @nonnull_load_as_inner(i32 addrspace(1)** %addr) {
  %ptr = load i32 addrspace(1)*, i32 addrspace(1)** %addr, !nonnull !{}
  %cmp = icmp eq i32 addrspace(1)* %ptr, null
  ret i1 %cmp
; CHECK-LABEL: @nonnull_load_as_inner
; CHECK: ret i1 false
}

; If a bit is known to be zero for A and known to be one for B,
; then A and B cannot be equal.
define i1 @icmp_eq_const(i32 %a) {
; CHECK-LABEL: @icmp_eq_const(
; CHECK-NEXT:    ret i1 false
;
  %b = mul nsw i32 %a, -2
  %c = icmp eq i32 %b, 1
  ret i1 %c
}

define <2 x i1> @icmp_eq_const_vec(<2 x i32> %a) {
; CHECK-LABEL: @icmp_eq_const_vec(
; CHECK-NEXT:    ret <2 x i1> zeroinitializer
;
  %b = mul nsw <2 x i32> %a, <i32 -2, i32 -2>
  %c = icmp eq <2 x i32> %b, <i32 1, i32 1>
  ret <2 x i1> %c
}

define i1 @icmp_ne_const(i32 %a) {
; CHECK-LABEL: @icmp_ne_const(
; CHECK-NEXT:    ret i1 true
;
  %b = mul nsw i32 %a, -2
  %c = icmp ne i32 %b, 1
  ret i1 %c
}

define <2 x i1> @icmp_ne_const_vec(<2 x i32> %a) {
; CHECK-LABEL: @icmp_ne_const_vec(
; CHECK-NEXT:    ret <2 x i1> <i1 true, i1 true>
;
  %b = mul nsw <2 x i32> %a, <i32 -2, i32 -2>
  %c = icmp ne <2 x i32> %b, <i32 1, i32 1>
  ret <2 x i1> %c
}

define i1 @icmp_sdiv_int_min(i32 %a) {
  %div = sdiv i32 -2147483648, %a
  %cmp = icmp ne i32 %div, -1073741824
  ret i1 %cmp

; CHECK-LABEL: @icmp_sdiv_int_min
; CHECK-NEXT: [[DIV:%.*]] = sdiv i32 -2147483648, %a
; CHECK-NEXT: [[CMP:%.*]] = icmp ne i32 [[DIV]], -1073741824
; CHECK-NEXT: ret i1 [[CMP]]
}

define i1 @icmp_sdiv_pr20288(i64 %a) {
   %div = sdiv i64 %a, -8589934592
   %cmp = icmp ne i64 %div, 1073741824
   ret i1 %cmp

; CHECK-LABEL: @icmp_sdiv_pr20288
; CHECK-NEXT: [[DIV:%.*]] = sdiv i64 %a, -8589934592
; CHECK-NEXT: [[CMP:%.*]] = icmp ne i64 [[DIV]], 1073741824
; CHECK-NEXT: ret i1 [[CMP]]
}

define i1 @icmp_sdiv_neg1(i64 %a) {
 %div = sdiv i64 %a, -1
 %cmp = icmp ne i64 %div, 1073741824
 ret i1 %cmp

; CHECK-LABEL: @icmp_sdiv_neg1
; CHECK-NEXT: [[DIV:%.*]] = sdiv i64 %a, -1
; CHECK-NEXT: [[CMP:%.*]] = icmp ne i64 [[DIV]], 1073741824
; CHECK-NEXT: ret i1 [[CMP]]
}

define i1 @icmp_known_bits(i4 %x, i4 %y) {
  %and1 = and i4 %y, -7
  %and2 = and i4 %x, -7
  %or1 = or i4 %and1, 2
  %or2 = or i4 %and2, 2
  %add = add i4 %or1, %or2
  %cmp = icmp eq i4 %add, 0
  ret i1 %cmp

; CHECK-LABEL: @icmp_known_bits
; CHECK-NEXT: ret i1 false
}

define i1 @icmp_shl_nuw_1(i64 %a) {
 %shl = shl nuw i64 1, %a
 %cmp = icmp ne i64 %shl, 0
 ret i1 %cmp

; CHECK-LABEL: @icmp_shl_nuw_1
; CHECK-NEXT: ret i1 true
}

define i1 @icmp_shl_1_V_ugt_2147483648(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ugt i32 %shl, 2147483648
  ret i1 %cmp

; CHECK-LABEL: @icmp_shl_1_V_ugt_2147483648(
; CHECK-NEXT: ret i1 false
}

define i1 @icmp_shl_1_V_ule_2147483648(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ule i32 %shl, 2147483648
  ret i1 %cmp

; CHECK-LABEL: @icmp_shl_1_V_ule_2147483648(
; CHECK-NEXT: ret i1 true
}

define i1 @icmp_shl_1_V_eq_31(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp eq i32 %shl, 31
  ret i1 %cmp

; CHECK-LABEL: @icmp_shl_1_V_eq_31(
; CHECK-NEXT: ret i1 false
}

define i1 @icmp_shl_1_V_ne_31(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ne i32 %shl, 31
  ret i1 %cmp

; CHECK-LABEL: @icmp_shl_1_V_ne_31(
; CHECK-NEXT: ret i1 true
}

define i1 @tautological1(i32 %A, i32 %B) {
  %C = and i32 %A, %B
  %D = icmp ugt i32 %C, %A
  ret i1 %D
; CHECK-LABEL: @tautological1(
; CHECK: ret i1 false
}

define i1 @tautological2(i32 %A, i32 %B) {
  %C = and i32 %A, %B
  %D = icmp ule i32 %C, %A
  ret i1 %D
; CHECK-LABEL: @tautological2(
; CHECK: ret i1 true
}

define i1 @tautological3(i32 %A, i32 %B) {
  %C = or i32 %A, %B
  %D = icmp ule i32 %A, %C
  ret i1 %D
; CHECK-LABEL: @tautological3(
; CHECK: ret i1 true
}

define i1 @tautological4(i32 %A, i32 %B) {
  %C = or i32 %A, %B
  %D = icmp ugt i32 %A, %C
  ret i1 %D
; CHECK-LABEL: @tautological4(
; CHECK: ret i1 false
}

define i1 @tautological5(i32 %A, i32 %B) {
  %C = or i32 %A, %B
  %D = icmp ult i32 %C, %A
  ret i1 %D
; CHECK-LABEL: @tautological5(
; CHECK: ret i1 false
}

define i1 @tautological6(i32 %A, i32 %B) {
  %C = or i32 %A, %B
  %D = icmp uge i32 %C, %A
  ret i1 %D
; CHECK-LABEL: @tautological6(
; CHECK: ret i1 true
}

define i1 @tautological7(i32 %A, i32 %B) {
  %C = and i32 %A, %B
  %D = icmp uge i32 %A, %C
  ret i1 %D
; CHECK-LABEL: @tautological7(
; CHECK: ret i1 true
}

define i1 @tautological8(i32 %A, i32 %B) {
  %C = and i32 %A, %B
  %D = icmp ult i32 %A, %C
  ret i1 %D
; CHECK-LABEL: @tautological8(
; CHECK: ret i1 false
}

declare void @helper_i1(i1)
; Series of tests for icmp s[lt|ge] (or A, B), A and icmp s[gt|le] A, (or A, B)
define void @icmp_slt_sge_or(i32 %Ax, i32 %Bx) {
; 'p' for positive, 'n' for negative, 'x' for potentially either.
; %D is 'icmp slt (or A, B), A'
; %E is 'icmp sge (or A, B), A' making it the not of %D
; %F is 'icmp sgt A, (or A, B)' making it the same as %D
; %G is 'icmp sle A, (or A, B)' making it the not of %D
  %Aneg = or i32 %Ax, 2147483648
  %Apos = and i32 %Ax, 2147483647
  %Bneg = or i32 %Bx, 2147483648
  %Bpos = and i32 %Bx, 2147483647

  %Cpp = or i32 %Apos, %Bpos
  %Dpp = icmp slt i32 %Cpp, %Apos
  %Epp = icmp sge i32 %Cpp, %Apos
  %Fpp = icmp sgt i32 %Apos, %Cpp
  %Gpp = icmp sle i32 %Apos, %Cpp
  %Cpx = or i32 %Apos, %Bx
  %Dpx = icmp slt i32 %Cpx, %Apos
  %Epx = icmp sge i32 %Cpx, %Apos
  %Fpx = icmp sgt i32 %Apos, %Cpx
  %Gpx = icmp sle i32 %Apos, %Cpx
  %Cpn = or i32 %Apos, %Bneg
  %Dpn = icmp slt i32 %Cpn, %Apos
  %Epn = icmp sge i32 %Cpn, %Apos
  %Fpn = icmp sgt i32 %Apos, %Cpn
  %Gpn = icmp sle i32 %Apos, %Cpn

  %Cxp = or i32 %Ax, %Bpos
  %Dxp = icmp slt i32 %Cxp, %Ax
  %Exp = icmp sge i32 %Cxp, %Ax
  %Fxp = icmp sgt i32 %Ax, %Cxp
  %Gxp = icmp sle i32 %Ax, %Cxp
  %Cxx = or i32 %Ax, %Bx
  %Dxx = icmp slt i32 %Cxx, %Ax
  %Exx = icmp sge i32 %Cxx, %Ax
  %Fxx = icmp sgt i32 %Ax, %Cxx
  %Gxx = icmp sle i32 %Ax, %Cxx
  %Cxn = or i32 %Ax, %Bneg
  %Dxn = icmp slt i32 %Cxn, %Ax
  %Exn = icmp sge i32 %Cxn, %Ax
  %Fxn = icmp sgt i32 %Ax, %Cxn
  %Gxn = icmp sle i32 %Ax, %Cxn

  %Cnp = or i32 %Aneg, %Bpos
  %Dnp = icmp slt i32 %Cnp, %Aneg
  %Enp = icmp sge i32 %Cnp, %Aneg
  %Fnp = icmp sgt i32 %Aneg, %Cnp
  %Gnp = icmp sle i32 %Aneg, %Cnp
  %Cnx = or i32 %Aneg, %Bx
  %Dnx = icmp slt i32 %Cnx, %Aneg
  %Enx = icmp sge i32 %Cnx, %Aneg
  %Fnx = icmp sgt i32 %Aneg, %Cnx
  %Gnx = icmp sle i32 %Aneg, %Cnx
  %Cnn = or i32 %Aneg, %Bneg
  %Dnn = icmp slt i32 %Cnn, %Aneg
  %Enn = icmp sge i32 %Cnn, %Aneg
  %Fnn = icmp sgt i32 %Aneg, %Cnn
  %Gnn = icmp sle i32 %Aneg, %Cnn

  call void @helper_i1(i1 %Dpp)
  call void @helper_i1(i1 %Epp)
  call void @helper_i1(i1 %Fpp)
  call void @helper_i1(i1 %Gpp)
  call void @helper_i1(i1 %Dpx)
  call void @helper_i1(i1 %Epx)
  call void @helper_i1(i1 %Fpx)
  call void @helper_i1(i1 %Gpx)
  call void @helper_i1(i1 %Dpn)
  call void @helper_i1(i1 %Epn)
  call void @helper_i1(i1 %Fpn)
  call void @helper_i1(i1 %Gpn)
  call void @helper_i1(i1 %Dxp)
  call void @helper_i1(i1 %Exp)
  call void @helper_i1(i1 %Fxp)
  call void @helper_i1(i1 %Gxp)
  call void @helper_i1(i1 %Dxx)
  call void @helper_i1(i1 %Exx)
  call void @helper_i1(i1 %Fxx)
  call void @helper_i1(i1 %Gxx)
  call void @helper_i1(i1 %Dxn)
  call void @helper_i1(i1 %Exn)
  call void @helper_i1(i1 %Fxn)
  call void @helper_i1(i1 %Gxn)
  call void @helper_i1(i1 %Dnp)
  call void @helper_i1(i1 %Enp)
  call void @helper_i1(i1 %Fnp)
  call void @helper_i1(i1 %Gnp)
  call void @helper_i1(i1 %Dnx)
  call void @helper_i1(i1 %Enx)
  call void @helper_i1(i1 %Fnx)
  call void @helper_i1(i1 %Gnx)
  call void @helper_i1(i1 %Dnn)
  call void @helper_i1(i1 %Enn)
  call void @helper_i1(i1 %Fnn)
  call void @helper_i1(i1 %Gnn)
; CHECK-LABEL: @icmp_slt_sge_or
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 %Dpx)
; CHECK: call void @helper_i1(i1 %Epx)
; CHECK: call void @helper_i1(i1 %Fpx)
; CHECK: call void @helper_i1(i1 %Gpx)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 %Dxx)
; CHECK: call void @helper_i1(i1 %Exx)
; CHECK: call void @helper_i1(i1 %Fxx)
; CHECK: call void @helper_i1(i1 %Gxx)
; CHECK: call void @helper_i1(i1 %Dxn)
; CHECK: call void @helper_i1(i1 %Exn)
; CHECK: call void @helper_i1(i1 %Fxn)
; CHECK: call void @helper_i1(i1 %Gxn)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
; CHECK: call void @helper_i1(i1 false)
; CHECK: call void @helper_i1(i1 true)
  ret void
}

define i1 @constant_fold_inttoptr_null() {
; CHECK-LABEL: @constant_fold_inttoptr_null(
; CHECK-NEXT:    ret i1 false
;
  %x = icmp eq i32* inttoptr (i64 32 to i32*), null
  ret i1 %x
}

define i1 @constant_fold_null_inttoptr() {
; CHECK-LABEL: @constant_fold_null_inttoptr(
; CHECK-NEXT:    ret i1 false
;
  %x = icmp eq i32* null, inttoptr (i64 32 to i32*)
  ret i1 %x
}
