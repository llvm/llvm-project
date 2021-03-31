; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc-unknown-linux-gnu \
; RUN:   -mcpu=pwr10 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | \
; RUN:   FileCheck %s --check-prefix=CHECK32

; These test cases aim to test constant materialization using the pli instruction on Power10.

define  signext i32 @t_16BitsMinRequiring34Bits() {
; CHECK-LABEL: t_16BitsMinRequiring34Bits:
; CHECK:	pli r3, 32768
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_16BitsMinRequiring34Bits:
; CHECK32:	pli r3, 32768
; CHECK32-NEXT:	blr

entry:
  ret i32 32768
}

define  signext i32 @t_16Bits() {
; CHECK-LABEL: t_16Bits:
; CHECK:	pli r3, 62004
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_16Bits:
; CHECK32:	pli r3, 62004
; CHECK32-NEXT:	blr

entry:
  ret i32 62004
}

define  signext i32 @t_lt32gt16BitsNonShiftable() {
; CHECK-LABEL: t_lt32gt16BitsNonShiftable:
; CHECK:	pli r3, 1193046
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_lt32gt16BitsNonShiftable:
; CHECK32:	pli r3, 1193046
; CHECK32-NEXT:	blr

entry:
  ret i32 1193046
}

define  signext i32 @t_32Bits() {
; CHECK-LABEL: t_32Bits:
; CHECK:	pli r3, -231451016
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_32Bits:
; CHECK32:	pli r3, -231451016
; CHECK32-NEXT:	blr

entry:
  ret i32 -231451016
}

define  i64 @t_34BitsLargestPositive() {
; CHECK-LABEL: t_34BitsLargestPositive:
; CHECK:	pli r3, 8589934591
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_34BitsLargestPositive:
; CHECK32:	li r3, 1
; CHECK32-NEXT: li r4, -1
; CHECK32-NEXT:	blr

entry:
  ret i64 8589934591
}

define  i64 @t_neg34Bits() {
; CHECK-LABEL: t_neg34Bits:
; CHECK:	pli r3, -8284514696
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_neg34Bits:
; CHECK32:	li r3, -2
; CHECK32-NEXT: pli r4, 305419896
; CHECK32-NEXT:	blr

entry:
  ret i64 -8284514696
}

define  signext i32 @t_16BitsMinRequiring34BitsMinusOne() {
; CHECK-LABEL: t_16BitsMinRequiring34BitsMinusOne:
; CHECK:	li r3, 32767
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_16BitsMinRequiring34BitsMinusOne:
; CHECK32:	li r3, 32767
; CHECK32-NEXT:	blr

entry:
  ret i32 32767
}

define  signext i32 @t_lt16Bits() {
; CHECK-LABEL: t_lt16Bits:
; CHECK:	li r3, 291
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_lt16Bits:
; CHECK32:	li r3, 291
; CHECK32-NEXT:	blr

entry:
  ret i32 291
}

define  signext i32 @t_neglt16Bits() {
; CHECK-LABEL: t_neglt16Bits:
; CHECK:	li r3, -3805
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_neglt16Bits:
; CHECK32:	li r3, -3805
; CHECK32-NEXT:	blr

entry:
  ret i32 -3805
}

define  signext i32 @t_neg16Bits() {
; CHECK-LABEL: t_neg16Bits:
; CHECK:	li r3, -32204
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_neg16Bits:
; CHECK32:	li r3, -32204
; CHECK32-NEXT:	blr

entry:
  ret i32 -32204
}

define  signext i32 @t_lt32gt16BitsShiftable() {
; CHECK-LABEL: t_lt32gt16BitsShiftable:
; CHECK:	lis r3, 18
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_lt32gt16BitsShiftable:
; CHECK32:	lis r3, 18
; CHECK32-NEXT:	blr

entry:
  ret i32 1179648
}

define  signext i32 @t_32gt16BitsShiftable() {
; CHECK-LABEL: t_32gt16BitsShiftable:
; CHECK:	lis r3, -3532
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_32gt16BitsShiftable:
; CHECK32:	lis r3, -3532
; CHECK32-NEXT:	blr

entry:
  ret i32 -231473152
}

define  signext i32 @t_32BitsZero() {
; CHECK-LABEL: t_32BitsZero:
; CHECK:	li r3, 0
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_32BitsZero:
; CHECK32:	li r3, 0
; CHECK32-NEXT:	blr

entry:
  ret i32 0
}

define  signext i32 @t_32BitsAllOnes() {
; CHECK-LABEL: t_32BitsAllOnes:
; CHECK:	li r3, -1
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_32BitsAllOnes:
; CHECK32:	li r3, -1
; CHECK32-NEXT:	blr

entry:
  ret i32 -1
}

define  i64 @t_34BitsLargestPositivePlus() {
; CHECK-LABEL: t_34BitsLargestPositivePlus:
; CHECK:	li r3, 1
; CHECK-NEXT:	rldic r3, r3, 33, 30
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_34BitsLargestPositivePlus:
; CHECK32:	li r3, 2
; CHECK32-NEXT:	li r4, 0
; CHECK32-NEXT:	blr

entry:
  ret i64 8589934592
}

define  i64 @t_34Bits() {
; CHECK-LABEL: t_34Bits:
; CHECK:	pli r3, 1648790223
; CHECK-NEXT:	rldic r3, r3, 3, 30
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_34Bits:
; CHECK32:	li r3, 3
; CHECK32-NEXT:	pli r4, 305419896
; CHECK32-NEXT:	blr

entry:
  ret i64 13190321784
}

define  i64 @t_35Bits() {
; CHECK-LABEL: t_35Bits:
; CHECK:	pli r3, 4266035919
; CHECK-NEXT:	rldic r3, r3, 3, 29
; CHECK-NEXT:	blr
; CHECK32-LABEL: t_35Bits:
; CHECK32:	li r3, 7
; CHECK32-NEXT:	pli r4, -231451016
; CHECK32-NEXT:	blr

entry:
  ret i64 34128287352
}

; (Value >> Shift) can be expressed in 34 bits
define  i64 @t_Shift() {
; CHECK-LABEL: t_Shift:
; CHECK:         pli r3, 8522759166
; CHECK-NEXT:    rotldi r3, r3, 48
; CHECK-NEXT:    blr

entry:
  ; 0xFBFE00000001FBFE
  ret i64 18157950747604548606
}

; Leading Zeros + Following Ones + Trailing Zeros > 30
define  i64 @t_LZFOTZ() {
; CHECK-LABEL: t_LZFOTZ:
; CHECK:         pli r3, -349233
; CHECK-NEXT:    rldic r3, r3, 4, 12
; CHECK-NEXT:    blr

entry:
  ; 0x000FFFFFFFAABCF0
  ret i64 4503599621782768
}

; Leading Zeros + Trailing Ones > 30
define  i64 @t_LZTO() {
; CHECK-LABEL: t_LZTO:
; CHECK:         pli r3, -2684406441
; CHECK-NEXT:    rldicl r3, r3, 11, 19
; CHECK-NEXT:    blr
entry:
  ; 0x00001AFFF9AABFFF
  ret i64 29686707699711
}

; Leading Zeros + Trailing Ones + Following Zeros > 30
define  i64 @t_LZTOFO() {
; CHECK-LABEL: t_LZTOFO:
; CHECK:         pli r3, -5720033968
; CHECK-NEXT:    rldicl r3, r3, 11, 12
; CHECK-NEXT:    blr
entry:
  ; 0x000FF55879AA87FF
  ret i64 4491884997806079
}

; Requires full expansion
define  i64 @t_Full64Bits1() {
; CHECK-LABEL: t_Full64Bits1:
; CHECK:         pli r4, 2146500607
; CHECK-NEXT:    pli r3, 4043305214
; CHECK-NEXT:    rldimi r3, r4, 32, 0
; CHECK-NEXT:    blr
entry:
  ; 0x7FF0FFFFF0FFF0FE
  ret i64 9219149911952453886
}

; Requires full expansion
define  i64 @t_Ful64Bits2() {
; CHECK-LABEL: t_Ful64Bits2:
; CHECK:         pli r4, 4042326015
; CHECK-NEXT:    pli r3, 4043305214
; CHECK-NEXT:    rldimi r3, r4, 32, 0
; CHECK-NEXT:    blr
entry:
  ; 0xF0F0FFFFF0FFF0FE
  ret i64 17361658038238310654
}

; A splat of 32 bits: 32 Bits Low == 32 Bits High
define  i64 @t_Splat32Bits() {
; CHECK-LABEL: t_Splat32Bits:
; CHECK:         pli r3, 262916796
; CHECK-NEXT:    rldimi r3, r3, 32, 0
; CHECK-NEXT:    blr
entry:
  ; 0x0FABCABC0FABCABC
  ret i64 1129219040652020412
}

; The load immediates resulting from phi-nodes are needed to test whether
; li/lis is preferred to pli by the instruction selector.
define dso_local void @t_phiNode() {
; CHECK-LABEL: t_phiNode:
; CHECK:	lis r6, 18
; CHECK-NEXT:	li r5, 291
; CHECK-NEXT:	li r4, 0
; CHECK-NEXT:   cmpwi r3, 1
; CHECK-NEXT:	li r3, -1
; CHECK:	pli r6, 2147483647
; CHECK-NEXT:	pli r5, 1193046
; CHECK-NEXT:	pli r4, 32768
; CHECK-NEXT:	pli r3, -231451016
; CHECK32-LABEL: t_phiNode:
; CHECK32:	lis r6, 18
; CHECK32-NEXT:	li r5, 291
; CHECK32-NEXT:	li r4, 0
; CHECK32-NEXT:   cmpwi r3, 1
; CHECK32-NEXT:	li r3, -1
; CHECK32:	pli r6, 2147483647
; CHECK32-NEXT:	pli r5, 1193046
; CHECK32-NEXT:	pli r4, 32768
; CHECK32-NEXT:	pli r3, -231451016

entry:
  br label %while.body

while.body:                                       ; preds = %if.else.i, %entry
  br label %while.body.i

while.body.i:                                     ; preds = %sw.epilog.i, %while.body
  %a.1.i = phi i32 [ %a.2.i, %sw.epilog.i ], [ -1, %while.body ]
  %b.1.i = phi i32 [ %b.2.i, %sw.epilog.i ], [ 0, %while.body ]
  %c.1.i = phi i32 [ %c.2.i, %sw.epilog.i ], [ 291, %while.body ]
  %d.1.i = phi i32 [ %d.2.i, %sw.epilog.i ], [ 1179648, %while.body ]
  %0 = load i8, i8* null, align 1
  %cmp1.i = icmp eq i8 %0, 1
  br i1 %cmp1.i, label %if.then.i, label %if.else.i

if.then.i:                                        ; preds = %while.body.i
  switch i8 undef, label %sw.default.i [
    i8 3, label %sw.epilog.i
    i8 2, label %sw.bb1.i
  ]

sw.bb1.i:                                        ; preds = %if.then.i
  br label %sw.epilog.i

sw.default.i:                                     ; preds = %if.then.i
  unreachable

sw.epilog.i:                                      ; preds = %sw.bb2.i, %sw.bb1.i, %if.then.i
  %a.2.i = phi i32 [ -231451016, %sw.bb1.i ], [ %a.1.i, %if.then.i ]
  %b.2.i = phi i32 [ 32768, %sw.bb1.i ], [ %b.1.i, %if.then.i ]
  %c.2.i = phi i32 [ 1193046, %sw.bb1.i ], [ %c.1.i, %if.then.i ]
  %d.2.i = phi i32 [ 2147483647, %sw.bb1.i ], [ %d.1.i, %if.then.i ]
  br label %while.body.i

if.else.i:                                     ; preds = %while.body.i
  call void @func2(i32 signext %a.1.i, i32 signext %b.1.i, i32 signext %c.1.i, i32 signext %d.1.i)
  br label %while.body
}

declare void @func2(i32, i32, i32, i32)
