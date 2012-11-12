; RUN: llc -mcpu=corei7 -no-stack-coloring=false < %s | FileCheck %s --check-prefix=YESCOLOR
; RUN: llc -mcpu=corei7 -no-stack-coloring=true  < %s | FileCheck %s --check-prefix=NOCOLOR

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

;YESCOLOR: subq  $136, %rsp
;NOCOLOR: subq  $264, %rsp

define i32 @myCall_w2(i32 %in) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b)
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  call void @llvm.lifetime.end(i64 -1, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
}


;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp

define i32 @myCall2_no_merge(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b)
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  call void @llvm.lifetime.end(i64 -1, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  ret i32 %t7
bb3:
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  ret i32 0
}

;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp

define i32 @myCall2_w2(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b)
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  call void @llvm.lifetime.end(i64 -1, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}
;YESCOLOR: subq  $208, %rsp
;NOCOLOR: subq  $400, %rsp




define i32 @myCall_w4(i32 %in) {
entry:
  %a1 = alloca [14 x i8*], align 8
  %a2 = alloca [13 x i8*], align 8
  %a3 = alloca [12 x i8*], align 8
  %a4 = alloca [11 x i8*], align 8
  %b1 = bitcast [14 x i8*]* %a1 to i8*
  %b2 = bitcast [13 x i8*]* %a2 to i8*
  %b3 = bitcast [12 x i8*]* %a3 to i8*
  %b4 = bitcast [11 x i8*]* %a4 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b4)
  call void @llvm.lifetime.start(i64 -1, i8* %b1)
  %t1 = call i32 @foo(i32 %in, i8* %b1)
  %t2 = call i32 @foo(i32 %in, i8* %b1)
  call void @llvm.lifetime.end(i64 -1, i8* %b1)
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t9 = call i32 @foo(i32 %in, i8* %b2)
  %t8 = call i32 @foo(i32 %in, i8* %b2)
  call void @llvm.lifetime.end(i64 -1, i8* %b2)
  call void @llvm.lifetime.start(i64 -1, i8* %b3)
  %t3 = call i32 @foo(i32 %in, i8* %b3)
  %t4 = call i32 @foo(i32 %in, i8* %b3)
  call void @llvm.lifetime.end(i64 -1, i8* %b3)
  %t11 = call i32 @foo(i32 %in, i8* %b4)
  call void @llvm.lifetime.end(i64 -1, i8* %b4)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
}

;YESCOLOR: subq  $112, %rsp
;NOCOLOR: subq  $400, %rsp

define i32 @myCall2_w4(i32 %in) {
entry:
  %a1 = alloca [14 x i8*], align 8
  %a2 = alloca [13 x i8*], align 8
  %a3 = alloca [12 x i8*], align 8
  %a4 = alloca [11 x i8*], align 8
  %b1 = bitcast [14 x i8*]* %a1 to i8*
  %b2 = bitcast [13 x i8*]* %a2 to i8*
  %b3 = bitcast [12 x i8*]* %a3 to i8*
  %b4 = bitcast [11 x i8*]* %a4 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b1)
  %t1 = call i32 @foo(i32 %in, i8* %b1)
  %t2 = call i32 @foo(i32 %in, i8* %b1)
  call void @llvm.lifetime.end(i64 -1, i8* %b1)
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t9 = call i32 @foo(i32 %in, i8* %b2)
  %t8 = call i32 @foo(i32 %in, i8* %b2)
  call void @llvm.lifetime.end(i64 -1, i8* %b2)
  call void @llvm.lifetime.start(i64 -1, i8* %b3)
  %t3 = call i32 @foo(i32 %in, i8* %b3)
  %t4 = call i32 @foo(i32 %in, i8* %b3)
  call void @llvm.lifetime.end(i64 -1, i8* %b3)
  br i1 undef, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start(i64 -1, i8* %b4)
  %t11 = call i32 @foo(i32 %in, i8* %b4)
  call void @llvm.lifetime.end(i64 -1, i8* %b4)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}


;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp


define i32 @myCall2_noend(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b)
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}

;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @myCall2_noend2(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %b)
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}


;YESCOLOR: subq  $144, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @myCall2_nostart(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}

; Adopt the test from Transforms/Inline/array_merge.ll'
;YESCOLOR: subq  $816, %rsp
;NOCOLOR: subq  $1616, %rsp
define void @array_merge() nounwind ssp {
entry:
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  %0 = bitcast [100 x i32]* %A.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %0) nounwind
  %1 = bitcast [100 x i32]* %B.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %1) nounwind
  call void @bar([100 x i32]* %A.i, [100 x i32]* %B.i) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %1) nounwind
  %2 = bitcast [100 x i32]* %A.i1 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %2) nounwind
  %3 = bitcast [100 x i32]* %B.i2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %3) nounwind
  call void @bar([100 x i32]* %A.i1, [100 x i32]* %B.i2) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %2) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %3) nounwind
  ret void
}

;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @func_phi_lifetime(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  br i1 %d, label %bb0, label %bb1

bb0:
  %I1 = bitcast [17 x i8*]* %a to i8*
  br label %bb2

bb1:
  %I2 = bitcast [16 x i8*]* %a2 to i8*
  br label %bb2

bb2:
  %split = phi i8* [ %I1, %bb0 ], [ %I2, %bb1 ]
  call void @llvm.lifetime.start(i64 -1, i8* %split)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  call void @llvm.lifetime.end(i64 -1, i8* %split)
  ret i32 %t7
bb3:
  ret i32 0
}


;YESCOLOR: multi_region_bb
;NOCOLOR: multi_region_bb
define void @multi_region_bb() nounwind ssp {
entry:
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  %0 = bitcast [100 x i32]* %A.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %0) nounwind ; <---- start #1
  %1 = bitcast [100 x i32]* %B.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %1) nounwind
  call void @bar([100 x i32]* %A.i, [100 x i32]* %B.i) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %1) nounwind
  %2 = bitcast [100 x i32]* %A.i1 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %2) nounwind
  %3 = bitcast [100 x i32]* %B.i2 to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %3) nounwind
  call void @llvm.lifetime.start(i64 -1, i8* %0) nounwind  ; <---- start #2
  call void @bar([100 x i32]* %A.i1, [100 x i32]* %B.i2) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %2) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %3) nounwind
  ret void
}


;YESCOLOR: subq  $272, %rsp
;NOCOLOR: subq  $272, %rsp
define i32 @myCall_end_before_begin(i32 %in, i1 %d) {
entry:
  %a = alloca [17 x i8*], align 8
  %a2 = alloca [16 x i8*], align 8
  %b = bitcast [17 x i8*]* %a to i8*
  %b2 = bitcast [16 x i8*]* %a2 to i8*
  %t1 = call i32 @foo(i32 %in, i8* %b)
  %t2 = call i32 @foo(i32 %in, i8* %b)
  call void @llvm.lifetime.end(i64 -1, i8* %b)
  call void @llvm.lifetime.start(i64 -1, i8* %b)
  br i1 %d, label %bb2, label %bb3
bb2:
  call void @llvm.lifetime.start(i64 -1, i8* %b2)
  %t3 = call i32 @foo(i32 %in, i8* %b2)
  %t4 = call i32 @foo(i32 %in, i8* %b2)
  %t5 = add i32 %t1, %t2
  %t6 = add i32 %t3, %t4
  %t7 = add i32 %t5, %t6
  ret i32 %t7
bb3:
  ret i32 0
}

; Check that we don't assert and crash even when there are allocas
; outside the declared lifetime regions.
;YESCOLOR: bad_range
;NOCOLOR:  bad_range
define void @bad_range() nounwind ssp {
entry:
  %A.i1 = alloca [100 x i32], align 4
  %B.i2 = alloca [100 x i32], align 4
  %A.i = alloca [100 x i32], align 4
  %B.i = alloca [100 x i32], align 4
  %0 = bitcast [100 x i32]* %A.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %0) nounwind
  %1 = bitcast [100 x i32]* %B.i to i8*
  call void @llvm.lifetime.start(i64 -1, i8* %1) nounwind
  call void @bar([100 x i32]* %A.i, [100 x i32]* %B.i) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %0) nounwind
  call void @llvm.lifetime.end(i64 -1, i8* %1) nounwind
  br label %block2

block2:
  ; I am used outside the marked lifetime.
  call void @bar([100 x i32]* %A.i, [100 x i32]* %B.i) nounwind
  ret void
}


; Check that we don't assert and crash even when there are usages
; of allocas which do not read or write outside the declared lifetime regions.
;YESCOLOR: shady_range
;NOCOLOR:  shady_range

%struct.Klass = type { i32, i32 }

define i32 @shady_range(i32 %argc, i8** nocapture %argv) uwtable {
  %a.i = alloca [4 x %struct.Klass], align 16
  %b.i = alloca [4 x %struct.Klass], align 16
  %a8 = bitcast [4 x %struct.Klass]* %a.i to i8*
  %b8 = bitcast [4 x %struct.Klass]* %b.i to i8*
  ; I am used outside the lifetime zone below:
  %z2 = getelementptr inbounds [4 x %struct.Klass]* %a.i, i64 0, i64 0, i32 0
  call void @llvm.lifetime.start(i64 -1, i8* %a8)
  call void @llvm.lifetime.start(i64 -1, i8* %b8)
  %z3 = load i32* %z2, align 16
  %r = call i32 @foo(i32 %z3, i8* %a8)
  %r2 = call i32 @foo(i32 %z3, i8* %b8)
  call void @llvm.lifetime.end(i64 -1, i8* %a8)
  call void @llvm.lifetime.end(i64 -1, i8* %b8)
  ret i32 9
}

declare void @bar([100 x i32]* , [100 x i32]*) nounwind

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind

declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind

declare i32 @foo(i32, i8*)

