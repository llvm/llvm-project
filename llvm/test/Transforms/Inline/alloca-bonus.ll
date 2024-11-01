; RUN: opt -passes=inline < %s -S -o - -inline-threshold=8 | FileCheck %s
; RUN: opt -passes='cgscc(inline)' < %s -S -o - -inline-threshold=8 | FileCheck %s

target datalayout = "p:32:32"

declare void @llvm.lifetime.start.p0i8(i64 %size, i8* nocapture %ptr)

@glbl = external global i32

define void @outer1() {
; CHECK-LABEL: @outer1(
; CHECK-NOT: call void @inner1
  %ptr = alloca i32
  call void @inner1(i32* %ptr)
  ret void
}

define void @inner1(i32 *%ptr) {
  %A = load i32, i32* %ptr
  store i32 0, i32* %ptr
  %C = getelementptr inbounds i32, i32* %ptr, i32 0
  %D = getelementptr inbounds i32, i32* %ptr, i32 1
  %E = bitcast i32* %ptr to i8*
  %F = select i1 false, i32* %ptr, i32* @glbl
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %E)
  call void @extern()
  ret void
}

define void @outer2() {
; CHECK-LABEL: @outer2(
; CHECK: call void @inner2
  %ptr = alloca i32
  call void @inner2(i32* %ptr)
  ret void
}

; %D poisons this call, scalar-repl can't handle that instruction.
define void @inner2(i32 *%ptr) {
  %A = load i32, i32* %ptr
  store i32 0, i32* %ptr
  %C = getelementptr inbounds i32, i32* %ptr, i32 0
  %D = getelementptr inbounds i32, i32* %ptr, i32 %A
  %E = bitcast i32* %ptr to i8*
  %F = select i1 false, i32* %ptr, i32* @glbl
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %E)
  call void @extern()
  ret void
}

define void @outer3() {
; CHECK-LABEL: @outer3(
; CHECK-NOT: call void @inner3
  %ptr = alloca i32
  call void @inner3(i32* %ptr, i1 undef)
  ret void
}

define void @inner3(i32 *%ptr, i1 %x) {
  %A = icmp eq i32* %ptr, null
  %B = and i1 %x, %A
  call void @extern()
  br i1 %A, label %bb.true, label %bb.false
bb.true:
  ; This block musn't be counted in the inline cost.
  %t1 = load i32, i32* %ptr
  %t2 = add i32 %t1, 1
  %t3 = add i32 %t2, 1
  %t4 = add i32 %t3, 1
  %t5 = add i32 %t4, 1
  %t6 = add i32 %t5, 1
  %t7 = add i32 %t6, 1
  %t8 = add i32 %t7, 1
  %t9 = add i32 %t8, 1
  %t10 = add i32 %t9, 1
  %t11 = add i32 %t10, 1
  %t12 = add i32 %t11, 1
  %t13 = add i32 %t12, 1
  %t14 = add i32 %t13, 1
  %t15 = add i32 %t14, 1
  %t16 = add i32 %t15, 1
  %t17 = add i32 %t16, 1
  %t18 = add i32 %t17, 1
  %t19 = add i32 %t18, 1
  %t20 = add i32 %t19, 1
  ret void
bb.false:
  ret void
}

define void @outer4(i32 %A) {
; CHECK-LABEL: @outer4(
; CHECK-NOT: call void @inner4
  %ptr = alloca i32
  call void @inner4(i32* %ptr, i32 %A)
  ret void
}

; %B poisons this call, scalar-repl can't handle that instruction. However, we
; still want to detect that the icmp and branch *can* be handled.
define void @inner4(i32 *%ptr, i32 %A) {
  %B = getelementptr inbounds i32, i32* %ptr, i32 %A
  %C = icmp eq i32* %ptr, null
  call void @extern()
  br i1 %C, label %bb.true, label %bb.false
bb.true:
  ; This block musn't be counted in the inline cost.
  %t1 = load i32, i32* %ptr
  %t2 = add i32 %t1, 1
  %t3 = add i32 %t2, 1
  %t4 = add i32 %t3, 1
  %t5 = add i32 %t4, 1
  %t6 = add i32 %t5, 1
  %t7 = add i32 %t6, 1
  %t8 = add i32 %t7, 1
  %t9 = add i32 %t8, 1
  %t10 = add i32 %t9, 1
  %t11 = add i32 %t10, 1
  %t12 = add i32 %t11, 1
  %t13 = add i32 %t12, 1
  %t14 = add i32 %t13, 1
  %t15 = add i32 %t14, 1
  %t16 = add i32 %t15, 1
  %t17 = add i32 %t16, 1
  %t18 = add i32 %t17, 1
  %t19 = add i32 %t18, 1
  %t20 = add i32 %t19, 1
  ret void
bb.false:
  ret void
}

define void @outer5() {
; CHECK-LABEL: @outer5(
; CHECK-NOT: call void @inner5
  %ptr = alloca i32
  call void @inner5(i1 false, i32* %ptr)
  ret void
}

; %D poisons this call, scalar-repl can't handle that instruction. However, if
; the flag is set appropriately, the poisoning instruction is inside of dead
; code, and so shouldn't be counted.
define void @inner5(i1 %flag, i32 *%ptr) {
  %A = load i32, i32* %ptr
  store i32 0, i32* %ptr
  call void @extern()
  %C = getelementptr inbounds i32, i32* %ptr, i32 0
  br i1 %flag, label %if.then, label %exit

if.then:
  %D = getelementptr inbounds i32, i32* %ptr, i32 %A
  %E = bitcast i32* %ptr to i8*
  %F = select i1 false, i32* %ptr, i32* @glbl
  call void @llvm.lifetime.start.p0i8(i64 0, i8* %E)
  ret void

exit:
  ret void
}

declare void @extern()
