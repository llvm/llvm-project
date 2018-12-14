; RUN: llc -mtriple=aarch64-unknown-linux-gnu < %s | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "arm64"

define i1 @test_EQ_IllEbT(i64 %a, i64 %b) {
; CHECK-LABEL: test_EQ_IllEbT
; CHECK:      cmn	x1, x0
; CHECK-NEXT: cset	w0, eq
; CHECK-NEXT: ret
entry:
  %add = sub i64 0, %b
  %cmp = icmp eq i64 %add, %a
  ret i1 %cmp
}

define i1 @test_EQ_IliEbT(i64 %a, i32 %b) {
; CHECK-LABEL: test_EQ_IliEbT
; CHECK:      cmn	x0, w1, sxtw
; CHECK-NEXT: cset	w0, eq
; CHECK-NEXT: ret
entry:
  %conv = sext i32 %b to i64
  %add = sub i64 0, %a
  %cmp = icmp eq i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IlsEbT(i64 %a, i16 %b) {
; CHECK-LABEL: test_EQ_IlsEbT
; CHECK:      cmn	x0, w1, sxth
; CHECK-NEXT: cset	w0, eq
; CHECK-NEXT: ret
entry:
  %conv = sext i16 %b to i64
  %add = sub i64 0, %a
  %cmp = icmp eq i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IlcEbT(i64 %a, i8 %b) {
; CHECK-LABEL: test_EQ_IlcEbT
; CHECK: 	cmn	x0, w1, uxtb
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %b to i64
  %add = sub i64 0, %a
  %cmp = icmp eq i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IilEbT(i32 %a, i64 %b) {
; CHECK-LABEL: test_EQ_IilEbT
; CHECK: 	cmn	x1, w0, sxtw
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = sext i32 %a to i64
  %add = sub i64 0, %b
  %cmp = icmp eq i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IiiEbT(i32 %a, i32 %b) {
; CHECK-LABEL: test_EQ_IiiEbT
; CHECK: 	cmn	w1, w0
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %add = sub i32 0, %b
  %cmp = icmp eq i32 %add, %a
  ret i1 %cmp
}

define i1 @test_EQ_IisEbT(i32 %a, i16 %b) {
; CHECK-LABEL: test_EQ_IisEbT
; CHECK: 	cmn	w0, w1, sxth
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %b to i32
  %add = sub i32 0, %a
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IicEbT(i32 %a, i8 %b) {
; CHECK-LABEL: test_EQ_IicEbT
; CHECK: 	cmn	w0, w1, uxtb
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %b to i32
  %add = sub i32 0, %a
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IslEbT(i16 %a, i64 %b) {
; CHECK-LABEL: test_EQ_IslEbT
; CHECK: 	cmn	x1, w0, sxth
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %a to i64
  %add = sub i64 0, %b
  %cmp = icmp eq i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IsiEbT(i16 %a, i32 %b) {
; CHECK-LABEL: test_EQ_IsiEbT
; CHECK: 	cmn	w1, w0, sxth
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %a to i32
  %add = sub i32 0, %b
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IssEbT(i16 %a, i16 %b) {
; CHECK-LABEL: test_EQ_IssEbT
; CHECK: 	sxth	w8, w1
; CHECK-NEXT: 	cmn	w8, w0, sxth
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT;   ret
entry:
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IscEbT(i16 %a, i8 %b) {
; CHECK-LABEL: test_EQ_IscEbT
; CHECK: 	and	w8, w1, #0xff
; CHECK-NEXT: 	cmn	w8, w0, sxth
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT;   ret
entry:
  %conv = sext i16 %a to i32
  %conv1 = zext i8 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IclEbT(i8 %a, i64 %b) {
; CHECK-LABEL: test_EQ_IclEbT
; CHECK:      	cmn	x1, w0, uxtb
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %a to i64
  %add = sub i64 0, %b
  %cmp = icmp eq i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IciEbT(i8 %a, i32 %b) {
; CHECK-LABEL: test_EQ_IciEbT
; CHECK:      	cmn	w1, w0, uxtb
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %a to i32
  %add = sub i32 0, %b
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IcsEbT(i8 %a, i16 %b) {
; CHECK-LABEL: test_EQ_IcsEbT
; CHECK:      	sxth	w8, w1
; CHECK-NEXT: 	cmn	w8, w0, uxtb
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT:	ret
entry:
  %conv = zext i8 %a to i32
  %conv1 = sext i16 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_EQ_IccEbT(i8 %a, i8 %b) {
; CHECK-LABEL: test_EQ_IccEbT
; CHECK:      	and	w8, w1, #0xff
; CHECK-NEXT: 	cmn	w8, w0, uxtb
; CHECK-NEXT: 	cset	w0, eq
; CHECK-NEXT:	ret
entry:
  %conv = zext i8 %a to i32
  %conv1 = zext i8 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp eq i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IllEbT(i64 %a, i64 %b) {
; CHECK-LABEL: test_NE_IllEbT
; CHECK:      	cmn	x1, x0
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %add = sub i64 0, %b
  %cmp = icmp ne i64 %add, %a
  ret i1 %cmp
}

define i1 @test_NE_IliEbT(i64 %a, i32 %b) {
; CHECK-LABEL: test_NE_IliEbT
; CHECK:      	cmn	x0, w1, sxtw
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = sext i32 %b to i64
  %add = sub i64 0, %a
  %cmp = icmp ne i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IlsEbT(i64 %a, i16 %b) {
; CHECK-LABEL: test_NE_IlsEbT
; CHECK:      	cmn	x0, w1, sxth
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %b to i64
  %add = sub i64 0, %a
  %cmp = icmp ne i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IlcEbT(i64 %a, i8 %b) {
; CHECK-LABEL: test_NE_IlcEbT
; CHECK:      	cmn	x0, w1, uxtb
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %b to i64
  %add = sub i64 0, %a
  %cmp = icmp ne i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IilEbT(i32 %a, i64 %b) {
; CHECK-LABEL: test_NE_IilEbT
; CHECK:      	cmn	x1, w0, sxtw
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = sext i32 %a to i64
  %add = sub i64 0, %b
  %cmp = icmp ne i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IiiEbT(i32 %a, i32 %b) {
; CHECK-LABEL: test_NE_IiiEbT
; CHECK:      	cmn	w1, w0
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %add = sub i32 0, %b
  %cmp = icmp ne i32 %add, %a
  ret i1 %cmp
}

define i1 @test_NE_IisEbT(i32 %a, i16 %b) {
; CHECK-LABEL: test_NE_IisEbT
; CHECK:      	cmn	w0, w1, sxth
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %b to i32
  %add = sub i32 0, %a
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IicEbT(i32 %a, i8 %b) {
; CHECK-LABEL: test_NE_IicEbT
; CHECK:      	cmn	w0, w1, uxtb
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %b to i32
  %add = sub i32 0, %a
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IslEbT(i16 %a, i64 %b) {
; CHECK-LABEL: test_NE_IslEbT
; CHECK:      	cmn	x1, w0, sxth
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %a to i64
  %add = sub i64 0, %b
  %cmp = icmp ne i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IsiEbT(i16 %a, i32 %b) {
; CHECK-LABEL: test_NE_IsiEbT
; CHECK:      	cmn	w1, w0, sxth
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = sext i16 %a to i32
  %add = sub i32 0, %b
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IssEbT(i16 %a, i16 %b) {
; CHECK-LABEL:test_NE_IssEbT
; CHECK:      	sxth	w8, w1
; CHECK-NEXT: 	cmn	w8, w0, sxth
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT:	ret
entry:
  %conv = sext i16 %a to i32
  %conv1 = sext i16 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IscEbT(i16 %a, i8 %b) {
; CHECK-LABEL:test_NE_IscEbT
; CHECK:      	and	w8, w1, #0xff
; CHECK-NEXT: 	cmn	w8, w0, sxth
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT:	ret
entry:
  %conv = sext i16 %a to i32
  %conv1 = zext i8 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IclEbT(i8 %a, i64 %b) {
; CHECK-LABEL:test_NE_IclEbT
; CHECK:      	cmn	x1, w0, uxtb
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %a to i64
  %add = sub i64 0, %b
  %cmp = icmp ne i64 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IciEbT(i8 %a, i32 %b) {
; CHECK-LABEL:test_NE_IciEbT
; CHECK:      	cmn	w1, w0, uxtb
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT: 	ret
entry:
  %conv = zext i8 %a to i32
  %add = sub i32 0, %b
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IcsEbT(i8 %a, i16 %b) {
; CHECK-LABEL:test_NE_IcsEbT
; CHECK:      	sxth	w8, w1
; CHECK-NEXT: 	cmn	w8, w0, uxtb
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT:	ret
entry:
  %conv = zext i8 %a to i32
  %conv1 = sext i16 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}

define i1 @test_NE_IccEbT(i8 %a, i8 %b) {
; CHECK-LABEL:test_NE_IccEbT
; CHECK:      	and	w8, w1, #0xff
; CHECK-NEXT: 	cmn	w8, w0, uxtb
; CHECK-NEXT: 	cset	w0, ne
; CHECK-NEXT:	ret
entry:
  %conv = zext i8 %a to i32
  %conv1 = zext i8 %b to i32
  %add = sub nsw i32 0, %conv1
  %cmp = icmp ne i32 %conv, %add
  ret i1 %cmp
}
