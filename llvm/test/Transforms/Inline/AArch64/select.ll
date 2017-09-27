; RUN: opt -inline -mtriple=aarch64--linux-gnu -S -o - < %s -inline-threshold=0 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

declare void @pad()
@glbl = external global i32

define i32 @outer1(i1 %cond) {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call i32 @inner1
  %C = call i32 @inner1(i1 %cond, i32 1)
  ret i32 %C
}

define i32 @inner1(i1 %cond, i32 %val) {
  %select = select i1 %cond, i32 1, i32 %val       ; Simplified to 1
  call void @pad()
  store i32 0, i32* @glbl
  ret i32 %select                                  ; Simplifies to ret i32 1
}


define i32 @outer2(i32 %val) {
; CHECK-LABEL: @outer2(
; CHECK-NOT: call i32 @inner2
  %C = call i32 @inner2(i1 true, i32 %val)
  ret i32 %C
}

define i32 @inner2(i1 %cond, i32 %val) {
  %select = select i1 %cond, i32 1, i32 %val       ; Simplifies to 1
  call void @pad()
  store i32 0, i32* @glbl
  ret i32 %select                                  ; Simplifies to ret i32 1
}


define i32 @outer3(i32 %val) {
; CHECK-LABEL: @outer3(
; CHECK-NOT: call i32 @inner3
  %C = call i32 @inner3(i1 false, i32 %val)
  ret i32 %C
}

define i32 @inner3(i1 %cond, i32 %val) {
  %select = select i1 %cond, i32 %val, i32 -1      ; Simplifies to -1
  call void @pad()
  store i32 0, i32* @glbl
  ret i32 %select                                  ; Simplifies to ret i32 -1
}


define i32 @outer4() {
; CHECK-LABEL: @outer4(
; CHECK-NOT: call i32 @inner4
  %C = call i32 @inner4(i1 true, i32 1, i32 -1)
  ret i32 %C
}

define i32 @inner4(i1 %cond, i32 %val1, i32 %val2) {
  %select = select i1 %cond, i32 %val1, i32 %val2  ; Simplifies to 1
  call void @pad()
  store i32 0, i32* @glbl
  store i32 1, i32* @glbl
  ret i32 %select                                  ; Simplifies to ret i32 1
}


define i1 @outer5() {
; CHECK-LABEL: @outer5(
; CHECK-NOT: call i1 @inner5
  %C = call i1 @inner5(i1 true, i1 true, i1 false)
  ret i1 %C
}

declare void @dead()

define i1 @inner5(i1 %cond, i1 %val1, i1 %val2) {
  %select = select i1 %cond, i1 %val1, i1 %val2    ; Simplifies to true
  br i1 %select, label %exit, label %isfalse       ; Simplifies to br label %end

isfalse:                                           ; This block is unreachable once inlined
  call void @dead()
  br label %exit

exit:
  store i32 0, i32* @glbl
  ret i1 %select                                   ; Simplifies to ret i1 true
}


define i32 @outer6(i1 %cond) {
; CHECK-LABEL: @outer6(
; CHECK-NOT: call i32 @inner6
  %A = alloca i32
  %C = call i32 @inner6(i1 %cond, i32* %A)
  ret i32 %C
}

define i32 @inner6(i1 %cond, i32* %ptr) {
  %G1 = getelementptr inbounds i32, i32* %ptr, i32 1
  %G2 = getelementptr inbounds i32, i32* %G1, i32 1
  %G3 = getelementptr inbounds i32, i32* %ptr, i32 2
  %select = select i1 %cond, i32* %G2, i32* %G3    ; Simplified to %A[2]
  %load = load i32, i32* %select                   ; SROA'ed
  call void @pad()
  ret i32 %load                                    ; Simplified
}


define i32 @outer7(i32* %ptr) {
; CHECK-LABEL: @outer7(
; CHECK-NOT: call i32 @inner7
  %A = alloca i32
  %C = call i32 @inner7(i1 true, i32* %A, i32* %ptr)
  ret i32 %C
}

define i32 @inner7(i1 %cond, i32* %p1, i32* %p2) {
  %select = select i1 %cond, i32* %p1, i32* %p2    ; Simplifies to %A
  %load = load i32, i32* %select                   ; SROA'ed
  call void @pad()
  store i32 0, i32* @glbl
  ret i32 %load                                    ; Simplified
}


define i32 @outer8(i32* %ptr) {
; CHECK-LABEL: @outer8(
; CHECK-NOT: call i32 @inner8
  %A = alloca i32
  %C = call i32 @inner8(i1 false, i32* %ptr, i32* %A)
  ret i32 %C
}

define i32 @inner8(i1 %cond, i32* %p1, i32* %p2) {
  %select = select i1 %cond, i32* %p1, i32* %p2    ; Simplifies to %A
  %load = load i32, i32* %select                   ; SROA'ed
  call void @pad()
  store i32 0, i32* @glbl
  ret i32 %load                                    ; Simplified
}


define <2 x i32> @outer9(<2 x i32> %val) {
; CHECK-LABEL: @outer9(
; CHECK-NOT: call <2 x i32> @inner9
  %C = call <2 x i32> @inner9(<2 x i1> <i1 true, i1 true>, <2 x i32> %val)
  ret <2 x i32> %C
}

define <2 x i32> @inner9(<2 x i1> %cond, <2 x i32> %val) {
  %select = select <2 x i1> %cond, <2 x i32> <i32 1, i32 1>, <2 x i32> %val              ; Simplifies to <1, 1>
  call void @pad()
  store i32 0, i32* @glbl
  ret <2 x i32> %select                                                                  ; Simplifies to ret <2 x i32> <1, 1>
}


define <2 x i32> @outer10(<2 x i32> %val) {
; CHECK-LABEL: @outer10(
; CHECK-NOT: call <2 x i32> @inner10
  %C = call <2 x i32> @inner10(<2 x i1> <i1 false, i1 false>, <2 x i32> %val)
  ret <2 x i32> %C
}

define <2 x i32> @inner10(<2 x i1> %cond, <2 x i32> %val) {
  %select = select <2 x i1> %cond, < 2 x i32> %val, <2 x i32> <i32 -1, i32 -1>           ; Simplifies to <-1, -1>
  call void @pad()
  store i32 0, i32* @glbl
  ret <2 x i32> %select                                                                  ; Simplifies to ret <2 x i32> <-1, -1>
}


define <2 x i32> @outer11() {
; CHECK-LABEL: @outer11(
; CHECK-NOT: call <2 x i32> @inner11
  %C = call <2 x i32> @inner11(<2 x i1> <i1 true, i1 false>)
  ret <2 x i32> %C
}

define <2 x i32> @inner11(<2 x i1> %cond) {
  %select = select <2 x i1> %cond, <2 x i32> <i32 1, i32 1>, < 2 x i32> <i32 -1, i32 -1> ; Simplifies to <1, -1>
  call void @pad()
  ret <2 x i32> %select                                                                  ; Simplifies to ret <2 x i32> <1, -1>
}


define i1 @outer12(i32* %ptr) {
; CHECK-LABEL: @outer12(
; CHECK-NOT: call i1 @inner12
  %C = call i1 @inner12(i1 true, i32* @glbl, i32* %ptr)
  ret i1 %C
}

define i1 @inner12(i1 %cond, i32* %ptr1, i32* %ptr2) {
  %select = select i1 %cond, i32* %ptr1, i32* %ptr2 ; Simplified to @glbl
  %cmp = icmp eq i32* %select, @glbl                ; Simplified to true
  call void @pad()
  store i32 0, i32* @glbl
  ret i1 %cmp                                       ; Simplifies to ret i1 true
}


define <2 x i32> @outer13(<2 x i32> %val1, <2 x i32> %val2) {
; CHECK-LABEL: @outer13(
; CHECK: call <2 x i32> @inner13
  %C = call <2 x i32> @inner13(<2 x i1> <i1 true, i1 false>, <2 x i32> %val1, <2 x i32> %val2)
  ret <2 x i32> %C
}

define <2 x i32> @inner13(<2 x i1> %cond, <2 x i32> %val1, < 2 x i32> %val2) {
  %select = select <2 x i1> %cond, <2 x i32> %val1, < 2 x i32> %val2 ; Cannot be Simplified
  call void @pad()
  store i32 0, i32* @glbl
  store i32 1, i32* @glbl
  ret <2 x i32> %select                                              ; Simplified
}


define i32 @outer14(i32 %val1, i32 %val2) {
; CHECK-LABEL: @outer14(
; CHECK-NOT: call i32 @inner14
  %C = call i32 @inner14(i1 true, i32 %val1, i32 %val2)
  ret i32 %C
}

define i32 @inner14(i1 %cond, i32 %val1, i32 %val2) {
  %select = select i1 %cond, i32 %val1, i32 %val2   ; Simplified to %val1
  call void @pad()
  store i32 0, i32* @glbl
  store i32 1, i32* @glbl
  ret i32 %select                                   ; Simplifies to ret i32 %val1
}


define i32 @outer15(i32 %val1, i32 %val2) {
; CHECK-LABEL: @outer15(
; CHECK-NOT: call i32 @inner15
  %C = call i32 @inner15(i1 false, i32 %val1, i32 %val2)
  ret i32 %C
}

define i32 @inner15(i1 %cond, i32 %val1, i32 %val2) {
  %select = select i1 %cond, i32 %val1, i32 %val2   ; Simplified to %val2
  call void @pad()
  store i32 0, i32* @glbl
  store i32 1, i32* @glbl
  ret i32 %select                                   ; Simplifies to ret i32 %val2
}
