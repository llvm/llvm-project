; RUN: llc -mcpu=corei7 -no-stack-coloring=false < %s | FileCheck %s --check-prefix=YESCOLOR --check-prefix=CHECK
; RUN: llc -mcpu=corei7 -no-stack-coloring=false -stackcoloring-lifetime-start-on-first-use=false < %s | FileCheck %s --check-prefix=NOFIRSTUSE --check-prefix=CHECK
; RUN: llc -mcpu=corei7 -no-stack-coloring=true  < %s | FileCheck %s --check-prefix=NOCOLOR --check-prefix=CHECK

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;CHECK-LABEL: myCall_w2:
;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp

define i32 @myCall_w2(i32 %in) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
}


;CHECK-LABEL: myCall2_no_merge
;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp

define i32 @myCall2_no_merge(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  ret i32 %t7
bb3:
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  ret i32 0
}

;CHECK-LABEL: myCall2_w2
;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp

define i32 @myCall2_w2(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}

;CHECK-LABEL: myCall_w4:
;YESCOLOR: subq  $112, %rsp
;NOFIRSTUSE: subq  $208, %rsp
;NOCOLOR: subq  $400, %rsp

define i32 @myCall_w4(i32 %in) {
entry:
  %a1 = alloca [14 x ptr], align 8
  %a2 = alloca [13 x ptr], align 8
  %a3 = alloca [12 x ptr], align 8
  %a4 = alloca [11 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a4)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a1)
  %t1 = call i32 @foo(i32 %in, ptr %a1)
  %t2 = call i32 @foo(i32 %in, ptr %a1)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a1)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t9 = call i32 @foo(i32 %in, ptr %a2)
  %t8 = call i32 @foo(i32 %in, ptr %a2)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a2)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a3)
  %t3 = call i32 @foo(i32 %in, ptr %a3)
  %t4 = call i32 @foo(i32 %in, ptr %a3)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a3)
  %t11 = call i32 @foo(i32 %in, ptr %a4)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a4)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
}

;CHECK-LABEL: myCall2_w4:
;YESCOLOR: subq  $112, %rsp
;NOCOLOR: subq  $400, %rsp

define i32 @myCall2_w4(i32 %in) {
entry:
  %a1 = alloca [14 x ptr], align 8
  %a2 = alloca [13 x ptr], align 8
  %a3 = alloca [12 x ptr], align 8
  %a4 = alloca [11 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a1)
  %t1 = call i32 @foo(i32 %in, ptr %a1)
  %t2 = call i32 @foo(i32 %in, ptr %a1)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a1)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t9 = call i32 @foo(i32 %in, ptr %a2)
  %t8 = call i32 @foo(i32 %in, ptr %a2)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a2)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a3)
  %t3 = call i32 @foo(i32 %in, ptr %a3)
  %t4 = call i32 @foo(i32 %in, ptr %a3)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a3)
  br i1 poison, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a4)
  %t11 = call i32 @foo(i32 %in, ptr %a4)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a4)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}


;CHECK-LABEL: myCall2_noend:
;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp


define i32 @myCall2_noend(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}

;CHECK-LABEL: myCall2_noend2:
;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @myCall2_noend2(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}


;CHECK-LABEL: myCall2_nostart:
;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @myCall2_nostart(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}

; Adopt the test from Transforms/Inline/array_merge.ll'
;CHECK-LABEL: array_merge:
;YESCOLOR: subq  $808, %rsp
;NOCOLOR: subq  $1608, %rsp
define void @array_merge() nounwind ssp {
entry:
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 -1, ptr %A.i) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %B.i) nounwind
  call void @bar(ptr %A.i, ptr %B.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %A.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %B.i) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %A.i1) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %B.i2) nounwind
  call void @bar(ptr %A.i1, ptr %B.i2) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %A.i1) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %B.i2) nounwind
  ret void
}

;CHECK-LABEL: func_phi_lifetime:
;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @func_phi_lifetime(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  br i1 %d, label %bb0, label %bb1

bb0:
  br label %bb2

bb1:
  br label %bb2

bb2:
  %split = phi ptr [ %a, %bb0 ], [ %a2, %bb1 ]
  call void @llvm.lifetime.start.p0(i64 -1, ptr %split)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  call void @llvm.lifetime.end.p0(i64 -1, ptr %split)
  ret i32 %t7
bb3:
  ret i32 0
}


;CHECK-LABEL: multi_region_bb:
;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp

define void @multi_region_bb() nounwind ssp {
entry:
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 -1, ptr %A.i) nounwind ; <---- start #1
  call void @llvm.lifetime.start.p0(i64 -1, ptr %B.i) nounwind
  call void @bar(ptr %A.i, ptr %B.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %A.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %B.i) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %A.i1) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %B.i2) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %A.i) nounwind  ; <---- start #2
  call void @bar(ptr %A.i1, ptr %B.i2) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %A.i1) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %A.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %B.i2) nounwind
  ret void
}

define i32 @myCall_end_before_begin(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x ptr], align 8
  %a2 = alloca [16 x ptr], align 8
  %t1 = call i32 @foo(i32 %in, ptr %a)
  %t2 = call i32 @foo(i32 %in, ptr %a)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a2)
  %t3 = call i32 @foo(i32 %in, ptr %a2)
  %t4 = call i32 @foo(i32 %in, ptr %a2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}


; Regression test for PR15707.  %buf1 and %buf2 should not be merged
; in this test case.
;CHECK-LABEL: myCall_pr15707:
;NOFIRSTUSE: subq $200008, %rsp
;NOCOLOR: subq $200008, %rsp
define void @myCall_pr15707() {
  %buf1 = alloca i8, i32 100000, align 16
  %buf2 = alloca i8, i32 100000, align 16

  call void @llvm.lifetime.start.p0(i64 -1, ptr %buf1)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %buf1)

  call void @llvm.lifetime.start.p0(i64 -1, ptr %buf1)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %buf2)
  %result1 = call i32 @foo(i32 0, ptr %buf1)
  %result2 = call i32 @foo(i32 0, ptr %buf2)
  ret void
}


; Check that we don't assert and crash even when there are allocas
; outside the declared lifetime regions.
;CHECK-LABEL: bad_range:
define void @bad_range() nounwind ssp {
entry:
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 -1, ptr %A.i) nounwind
  call void @llvm.lifetime.start.p0(i64 -1, ptr %B.i) nounwind
  call void @bar(ptr %A.i, ptr %B.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %A.i) nounwind
  call void @llvm.lifetime.end.p0(i64 -1, ptr %B.i) nounwind
  br label %block2

block2:
  ; I am used outside the marked lifetime.
  call void @bar(ptr %A.i, ptr %B.i) nounwind
  ret void
}


; Check that we don't assert and crash even when there are usages
; of allocas which do not read or write outside the declared lifetime regions.
;CHECK-LABEL: shady_range:

%struct.Klass = type { i32, i32 }

define i32 @shady_range(i32 %argc, ptr nocapture %argv) uwtable {
  %a.i = alloca [4 x %struct.Klass], align 16
  %b.i = alloca [4 x %struct.Klass], align 16
  ; I am used outside the lifetime zone below:
  call void @llvm.lifetime.start.p0(i64 -1, ptr %a.i)
  call void @llvm.lifetime.start.p0(i64 -1, ptr %b.i)
  %z3 = load i32, ptr %a.i, align 16
  %r = call i32 @foo(i32 %z3, ptr %a.i)
  %r2 = call i32 @foo(i32 %z3, ptr %b.i)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %a.i)
  call void @llvm.lifetime.end.p0(i64 -1, ptr %b.i)
  ret i32 9
}

; In this case 'itar1' and 'itar2' can't be overlapped if we treat
; lifetime.start as the beginning of the lifetime, but we can
; overlap if we consider first use of the slot as lifetime
; start. See llvm bug 25776.

;CHECK-LABEL: ifthen_twoslots:
;YESCOLOR: subq  $1544, %rsp
;NOFIRSTUSE: subq $2056, %rsp
;NOCOLOR: subq  $2568, %rsp

define i32 @ifthen_twoslots(i32 %x) #0 {
entry:
  %b1 = alloca [128 x i32], align 16
  %b2 = alloca [128 x i32], align 16
  %b3 = alloca [128 x i32], align 16
  %b4 = alloca [128 x i32], align 16
  %b5 = alloca [128 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 512, ptr %b1)
  call void @llvm.lifetime.start.p0(i64 512, ptr %b2)
  %and = and i32 %x, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 512, ptr %b3)
  call void @initb(ptr %b1, ptr %b3, ptr null)
  call void @llvm.lifetime.end.p0(i64 512, ptr %b3)
  br label %if.end

if.else:                                          ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 512, ptr %b4)
  call void @llvm.lifetime.start.p0(i64 512, ptr %b5)
  call void @initb(ptr %b2, ptr %b4, ptr %b5) #3
  call void @llvm.lifetime.end.p0(i64 512, ptr %b5)
  call void @llvm.lifetime.end.p0(i64 512, ptr %b4)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  call void @llvm.lifetime.end.p0(i64 512, ptr %b2)
  call void @llvm.lifetime.end.p0(i64 512, ptr %b1)
  ret i32 0

}

; This function is intended to test the case where you
; have a reference to a stack slot that lies outside of
; the START/END lifetime markers-- the flow analysis
; should catch this and build the lifetime based on the
; markers only.

;CHECK-LABEL: while_loop:
;YESCOLOR: subq  $1032, %rsp
;NOFIRSTUSE: subq  $1544, %rsp
;NOCOLOR: subq  $1544, %rsp

define i32 @while_loop(i32 %x) #0 {
entry:
  %b1 = alloca [128 x i32], align 16
  %b2 = alloca [128 x i32], align 16
  %b3 = alloca [128 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 512, ptr %b1) #3
  call void @llvm.lifetime.start.p0(i64 512, ptr %b2) #3
  %and = and i32 %x, 1
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  call void @inita(ptr %b2) #3
  br label %if.end

if.else:                                          ; preds = %entry
  call void @inita(ptr %b1) #3
  call void @inita(ptr %b3) #3
  %tobool25 = icmp eq i32 %x, 0
  br i1 %tobool25, label %if.end, label %while.body.lr.ph

while.body.lr.ph:                                 ; preds = %if.else
  br label %while.body

while.body:                                       ; preds = %while.body.lr.ph, %while.body
  %x.addr.06 = phi i32 [ %x, %while.body.lr.ph ], [ %dec, %while.body ]
  %dec = add nsw i32 %x.addr.06, -1
  call void @llvm.lifetime.start.p0(i64 512, ptr %b3) #3
  call void @inita(ptr %b3) #3
  call void @llvm.lifetime.end.p0(i64 512, ptr %b3) #3
  %tobool2 = icmp eq i32 %dec, 0
  br i1 %tobool2, label %if.end.loopexit, label %while.body

if.end.loopexit:                                  ; preds = %while.body
  br label %if.end

if.end:                                           ; preds = %if.end.loopexit, %if.else, %if.then
  call void @llvm.lifetime.end.p0(i64 512, ptr %b2) #3
  call void @llvm.lifetime.end.p0(i64 512, ptr %b1) #3
  ret i32 0
}

; Test case motivated by PR27903. Same routine inlined multiple times
; into a caller results in a multi-segment lifetime, but the second
; lifetime has no explicit references to the stack slot. Such slots
; have to be treated conservatively.

;CHECK-LABEL: twobod_b27903:
;YESCOLOR: subq  $96, %rsp
;NOFIRSTUSE: subq  $96, %rsp
;NOCOLOR: subq  $96, %rsp

define i32 @twobod_b27903(i32 %y, i32 %x) {
entry:
  %buffer.i = alloca [12 x i32], align 16
  %abc = alloca [12 x i32], align 16
  call void @llvm.lifetime.start.p0(i64 48, ptr %buffer.i)
  %idxprom.i = sext i32 %y to i64
  %arrayidx.i = getelementptr inbounds [12 x i32], ptr %buffer.i, i64 0, i64 %idxprom.i
  call void @inita(ptr %arrayidx.i)
  %add.i = add nsw i32 %x, %y
  call void @llvm.lifetime.end.p0(i64 48, ptr %buffer.i)
  %tobool = icmp eq i32 %y, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  call void @llvm.lifetime.start.p0(i64 48, ptr %abc)
  %arrayidx = getelementptr inbounds [12 x i32], ptr %abc, i64 0, i64 %idxprom.i
  call void @inita(ptr %arrayidx)
  call void @llvm.lifetime.start.p0(i64 48, ptr %buffer.i)
  call void @inita(ptr %arrayidx.i)
  %add.i9 = add nsw i32 %add.i, %y
  call void @llvm.lifetime.end.p0(i64 48, ptr %buffer.i)
  call void @llvm.lifetime.end.p0(i64 48, ptr %abc)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %x.addr.0 = phi i32 [ %add.i9, %if.then ], [ %add.i, %entry ]
  ret i32 %x.addr.0
}

;CHECK-LABEL: multi_segment:
;YESCOLOR: subq  $256, %rsp
;NOFIRSTUSE: subq  $256, %rsp
;NOCOLOR: subq  $512, %rsp
define i1 @multi_segment(i1, i1)
{
entry-block:
  %foo = alloca [32 x i64]
  %bar = alloca [32 x i64]
  call void @llvm.lifetime.start.p0(i64 256, ptr %bar)
  call void @baz(ptr %bar, i32 1)
  call void @llvm.lifetime.end.p0(i64 256, ptr %bar)
  call void @llvm.lifetime.start.p0(i64 256, ptr %foo)
  call void @baz(ptr %foo, i32 1)
  call void @llvm.lifetime.end.p0(i64 256, ptr %foo)
  call void @llvm.lifetime.start.p0(i64 256, ptr %bar)
  call void @baz(ptr %bar, i32 1)
  call void @llvm.lifetime.end.p0(i64 256, ptr %bar)
  ret i1 true
}

;CHECK-LABEL: pr32488:
;YESCOLOR: subq  $256, %rsp
;NOFIRSTUSE: subq  $256, %rsp
;NOCOLOR: subq  $512, %rsp
define i1 @pr32488(i1, i1)
{
entry-block:
  %foo = alloca [32 x i64]
  %bar = alloca [32 x i64]
  br i1 %0, label %if_false, label %if_true
if_false:
  call void @llvm.lifetime.start.p0(i64 256, ptr %bar)
  call void @baz(ptr %bar, i32 0)
  br i1 %1, label %if_false.1, label %onerr
if_false.1:
  call void @llvm.lifetime.end.p0(i64 256, ptr %bar)
  br label %merge
if_true:
  call void @llvm.lifetime.start.p0(i64 256, ptr %foo)
  call void @baz(ptr %foo, i32 1)
  br i1 %1, label %if_true.1, label %onerr
if_true.1:
  call void @llvm.lifetime.end.p0(i64 256, ptr %foo)
  br label %merge
merge:
  ret i1 false
onerr:
  call void @llvm.lifetime.end.p0(i64 256, ptr %foo)
  call void @llvm.lifetime.end.p0(i64 256, ptr %bar)
  call void @destructor()
  ret i1 true
}

%Data = type { [32 x i64] }

declare void @destructor()

declare void @inita(ptr)

declare void @initb(ptr,ptr,ptr)

declare void @bar(ptr , ptr) nounwind

declare void @baz(ptr, i32)

declare void @llvm.lifetime.start.p0(i64, ptr nocapture) nounwind

declare void @llvm.lifetime.end.p0(i64, ptr nocapture) nounwind

declare i32 @foo(i32, ptr)
