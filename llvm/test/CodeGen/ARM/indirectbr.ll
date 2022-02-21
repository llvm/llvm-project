; RUN: llc < %s -relocation-model=pic -mtriple=armv6-apple-darwin | FileCheck %s -check-prefix=ARM
; RUN: llc < %s -relocation-model=pic -mtriple=thumbv6-apple-darwin | FileCheck %s -check-prefix=THUMB
; RUN: llc < %s -relocation-model=static -mtriple=thumbv7-apple-darwin | FileCheck %s -check-prefix=THUMB2
; RUN: llc < %s -relocation-model=static -mtriple=thumbv8-apple-darwin | FileCheck %s -check-prefix=THUMB2

@nextaddr = global i8* null                       ; <i8**> [#uses=2]
@C.0.2070 = private constant [5 x i8*] [i8* blockaddress(@foo, %L1), i8* blockaddress(@foo, %L2), i8* blockaddress(@foo, %L3), i8* blockaddress(@foo, %L4), i8* blockaddress(@foo, %L5)] ; <[5 x i8*]*> [#uses=1]

define internal i32 @foo(i32 %i) nounwind {
; ARM-LABEL: foo:
; THUMB-LABEL: foo:
; THUMB2-LABEL: foo:
entry:
  ; _nextaddr gets CSEed for use later on.
; THUMB: ldr r[[NEXTADDR_REG:[0-9]+]], [[NEXTADDR_CPI:LCPI0_[0-9]+]]
; THUMB: [[NEXTADDR_PCBASE:LPC0_[0-9]]]:
; THUMB: add r[[NEXTADDR_REG]], pc

  %0 = load i8*, i8** @nextaddr, align 4               ; <i8*> [#uses=2]
  %1 = icmp eq i8* %0, null                       ; <i1> [#uses=1]
; indirect branch gets duplicated here
; ARM: bx
; THUMB: mov pc,
; THUMB2: mov pc,
  br i1 %1, label %bb3, label %bb2

bb2:                                              ; preds = %entry, %bb3
  %gotovar.4.0 = phi i8* [ %gotovar.4.0.pre, %bb3 ], [ %0, %entry ] ; <i8*> [#uses=1]
; ARM: bx
; THUMB: mov pc,
  indirectbr i8* %gotovar.4.0, [label %L5, label %L4, label %L3, label %L2, label %L1]

bb3:                                              ; preds = %entry
  %2 = getelementptr inbounds [5 x i8*], [5 x i8*]* @C.0.2070, i32 0, i32 %i ; <i8**> [#uses=1]
  %gotovar.4.0.pre = load i8*, i8** %2, align 4        ; <i8*> [#uses=1]
  br label %bb2

L5:                                               ; preds = %bb2
  br label %L4

L4:                                               ; preds = %L5, %bb2
  %res.0 = phi i32 [ 385, %L5 ], [ 35, %bb2 ]     ; <i32> [#uses=1]
  br label %L3

L3:                                               ; preds = %L4, %bb2
  %res.1 = phi i32 [ %res.0, %L4 ], [ 5, %bb2 ]   ; <i32> [#uses=1]
  br label %L2

L2:                                               ; preds = %L3, %bb2
; THUMB-LABEL: %.split4
; THUMB: muls
  %res.2 = phi i32 [ %res.1, %L3 ], [ 1, %bb2 ]   ; <i32> [#uses=1]
  %phitmp = mul i32 %res.2, 6                     ; <i32> [#uses=1]
  br label %L1

L1:                                               ; preds = %L2, %bb2
  %res.3 = phi i32 [ %phitmp, %L2 ], [ 2, %bb2 ]  ; <i32> [#uses=1]
; ARM-LABEL: %L1
; ARM: ldr [[R_NEXTADDR:r[0-9]+]], LCPI
; ARM: ldr [[R1:r[0-9]+]], LCPI
; ARM: add [[R_NEXTADDR_b:r[0-9]+]], pc, [[R_NEXTADDR]]
; ARM: add [[R1b:r[0-9]+]], pc, [[R1]]
; ARM: str [[R1b]], [[[R_NEXTADDR_b]]]

; THUMB-LABEL: %L1
; THUMB: ldr [[R2:r[0-9]+]], LCPI
; THUMB: add [[R2]], pc
; THUMB: str [[R2]], [r[[NEXTADDR_REG]]]
; THUMB2-LABEL: %L1
; THUMB2: ldr [[R2:r[0-9]+]], LCPI
; THUMB2-NEXT: str{{(.w)?}} [[R2]]
  store i8* blockaddress(@foo, %L5), i8** @nextaddr, align 4
  ret i32 %res.3
}
; ARM: .long Ltmp0-(LPC{{.*}}+8)
; THUMB: .long Ltmp0-(LPC{{.*}}+4)
; THUMB: .long _nextaddr-([[NEXTADDR_PCBASE]]+4)
; THUMB2: .long Ltmp0
