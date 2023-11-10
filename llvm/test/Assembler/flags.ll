; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

@addr = external global i64

define i64 @add_unsigned(i64 %x, i64 %y) {
; CHECK: %z = add nuw i64 %x, %y
	%z = add nuw i64 %x, %y
	ret i64 %z
}

define i64 @sub_unsigned(i64 %x, i64 %y) {
; CHECK: %z = sub nuw i64 %x, %y
	%z = sub nuw i64 %x, %y
	ret i64 %z
}

define i64 @mul_unsigned(i64 %x, i64 %y) {
; CHECK: %z = mul nuw i64 %x, %y
	%z = mul nuw i64 %x, %y
	ret i64 %z
}

define i64 @add_signed(i64 %x, i64 %y) {
; CHECK: %z = add nsw i64 %x, %y
	%z = add nsw i64 %x, %y
	ret i64 %z
}

define i64 @sub_signed(i64 %x, i64 %y) {
; CHECK: %z = sub nsw i64 %x, %y
	%z = sub nsw i64 %x, %y
	ret i64 %z
}

define i64 @mul_signed(i64 %x, i64 %y) {
; CHECK: %z = mul nsw i64 %x, %y
	%z = mul nsw i64 %x, %y
	ret i64 %z
}

define i64 @add_plain(i64 %x, i64 %y) {
; CHECK: %z = add i64 %x, %y
	%z = add i64 %x, %y
	ret i64 %z
}

define i64 @sub_plain(i64 %x, i64 %y) {
; CHECK: %z = sub i64 %x, %y
	%z = sub i64 %x, %y
	ret i64 %z
}

define i64 @mul_plain(i64 %x, i64 %y) {
; CHECK: %z = mul i64 %x, %y
	%z = mul i64 %x, %y
	ret i64 %z
}

define i64 @add_both(i64 %x, i64 %y) {
; CHECK: %z = add nuw nsw i64 %x, %y
	%z = add nuw nsw i64 %x, %y
	ret i64 %z
}

define i64 @sub_both(i64 %x, i64 %y) {
; CHECK: %z = sub nuw nsw i64 %x, %y
	%z = sub nuw nsw i64 %x, %y
	ret i64 %z
}

define i64 @mul_both(i64 %x, i64 %y) {
; CHECK: %z = mul nuw nsw i64 %x, %y
	%z = mul nuw nsw i64 %x, %y
	ret i64 %z
}

define i64 @add_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = add nuw nsw i64 %x, %y
	%z = add nsw nuw i64 %x, %y
	ret i64 %z
}

define i64 @sub_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = sub nuw nsw i64 %x, %y
	%z = sub nsw nuw i64 %x, %y
	ret i64 %z
}

define i64 @mul_both_reversed(i64 %x, i64 %y) {
; CHECK: %z = mul nuw nsw i64 %x, %y
	%z = mul nsw nuw i64 %x, %y
	ret i64 %z
}

define i64 @shl_both(i64 %x, i64 %y) {
; CHECK: %z = shl nuw nsw i64 %x, %y
	%z = shl nuw nsw i64 %x, %y
	ret i64 %z
}

define i64 @sdiv_exact(i64 %x, i64 %y) {
; CHECK: %z = sdiv exact i64 %x, %y
	%z = sdiv exact i64 %x, %y
	ret i64 %z
}

define i64 @sdiv_plain(i64 %x, i64 %y) {
; CHECK: %z = sdiv i64 %x, %y
	%z = sdiv i64 %x, %y
	ret i64 %z
}

define i64 @udiv_exact(i64 %x, i64 %y) {
; CHECK: %z = udiv exact i64 %x, %y
	%z = udiv exact i64 %x, %y
	ret i64 %z
}

define i64 @udiv_plain(i64 %x, i64 %y) {
; CHECK: %z = udiv i64 %x, %y
	%z = udiv i64 %x, %y
	ret i64 %z
}

define i64 @ashr_plain(i64 %x, i64 %y) {
; CHECK: %z = ashr i64 %x, %y
	%z = ashr i64 %x, %y
	ret i64 %z
}

define i64 @ashr_exact(i64 %x, i64 %y) {
; CHECK: %z = ashr exact i64 %x, %y
	%z = ashr exact i64 %x, %y
	ret i64 %z
}

define i64 @lshr_plain(i64 %x, i64 %y) {
; CHECK: %z = lshr i64 %x, %y
	%z = lshr i64 %x, %y
	ret i64 %z
}

define i64 @lshr_exact(i64 %x, i64 %y) {
; CHECK: %z = lshr exact i64 %x, %y
	%z = lshr exact i64 %x, %y
	ret i64 %z
}

define ptr @gep_nw(ptr %p, i64 %x) {
; CHECK: %z = getelementptr inbounds i64, ptr %p, i64 %x
	%z = getelementptr inbounds i64, ptr %p, i64 %x
        ret ptr %z
}

define ptr @gep_plain(ptr %p, i64 %x) {
; CHECK: %z = getelementptr i64, ptr %p, i64 %x
	%z = getelementptr i64, ptr %p, i64 %x
        ret ptr %z
}

define i64 @add_both_ce() {
; CHECK: ret i64 add nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 add nsw nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @sub_both_ce() {
; CHECK: ret i64 sub nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 sub nsw nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @mul_both_ce() {
; CHECK: ret i64 mul nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 mul nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @ashr_exact_ce() {
; CHECK: ret i64 ashr exact (i64 ptrtoint (ptr @addr to i64), i64 9)
	ret i64 ashr exact (i64 ptrtoint (ptr @addr to i64), i64 9)
}

define i64 @lshr_exact_ce() {
; CHECK: ret i64 lshr exact (i64 ptrtoint (ptr @addr to i64), i64 9)
	ret i64 lshr exact (i64 ptrtoint (ptr @addr to i64), i64 9)
}

define ptr @gep_nw_ce() {
; CHECK: ret ptr getelementptr inbounds (i64, ptr @addr, i64 171)
        ret ptr getelementptr inbounds (i64, ptr @addr, i64 171)
}

define i64 @add_plain_ce() {
; CHECK: ret i64 add (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 add (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @sub_plain_ce() {
; CHECK: ret i64 sub (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 sub (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @mul_plain_ce() {
; CHECK: ret i64 mul (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 mul (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define ptr @gep_plain_ce() {
; CHECK: ret ptr getelementptr (i64, ptr @addr, i64 171)
        ret ptr getelementptr (i64, ptr @addr, i64 171)
}

define i64 @add_both_reversed_ce() {
; CHECK: ret i64 add nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 add nsw nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @sub_both_reversed_ce() {
; CHECK: ret i64 sub nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 sub nsw nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @mul_both_reversed_ce() {
; CHECK: ret i64 mul nuw nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 mul nsw nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @add_signed_ce() {
; CHECK: ret i64 add nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 add nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @sub_signed_ce() {
; CHECK: ret i64 sub nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 sub nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @mul_signed_ce() {
; CHECK: ret i64 mul nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 mul nsw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @shl_signed_ce() {
; CHECK: ret i64 shl nsw (i64 ptrtoint (ptr @addr to i64), i64 17)
	ret i64 shl nsw (i64 ptrtoint (ptr @addr to i64), i64 17)
}


define i64 @add_unsigned_ce() {
; CHECK: ret i64 add nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 add nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @sub_unsigned_ce() {
; CHECK: ret i64 sub nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 sub nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @mul_unsigned_ce() {
; CHECK: ret i64 mul nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
	ret i64 mul nuw (i64 ptrtoint (ptr @addr to i64), i64 91)
}

define i64 @test_zext(i32 %a) {
; CHECK: %res = zext nneg i32 %a to i64
  %res = zext nneg i32 %a to i64
  ret i64 %res
}

