; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; Make sure that basicaa thinks R and r are must aliases.
define i32 @test1(ptr %P) {
entry:
	%R = getelementptr {i32, i32}, ptr %P, i32 0, i32 1
	%S = load i32, ptr %R

	%r = getelementptr {i32, i32}, ptr %P, i32 0, i32 1
	%s = load i32, ptr %r

	%t = sub i32 %S, %s
	ret i32 %t
; CHECK-LABEL: @test1(
; CHECK: ret i32 0
}

define i32 @test2(ptr %P) {
entry:
	%R = getelementptr {i32, i32, i32}, ptr %P, i32 0, i32 1
	%S = load i32, ptr %R

	%r = getelementptr {i32, i32, i32}, ptr %P, i32 0, i32 2
  store i32 42, ptr %r

	%s = load i32, ptr %R

	%t = sub i32 %S, %s
	ret i32 %t
; CHECK-LABEL: @test2(
; CHECK: ret i32 0
}


; This was a miscompilation.
define i32 @test3(ptr %P) {
entry:
  %P2 = getelementptr {float, {i32, i32, i32}}, ptr %P, i32 0, i32 1
	%R = getelementptr {i32, i32, i32}, ptr %P2, i32 0, i32 1
	%S = load i32, ptr %R

	%r = getelementptr {i32, i32, i32}, ptr %P2, i32 0, i32 2
  store i32 42, ptr %r

	%s = load i32, ptr %R

	%t = sub i32 %S, %s
	ret i32 %t
; CHECK-LABEL: @test3(
; CHECK: ret i32 0
}


;; This is reduced from the SmallPtrSet constructor.
%SmallPtrSetImpl = type { ptr, i32, i32, i32, [1 x ptr] }
%SmallPtrSet64 = type { %SmallPtrSetImpl, [64 x ptr] }

define i32 @test4(ptr %P) {
entry:
  %tmp2 = getelementptr inbounds %SmallPtrSet64, ptr %P, i64 0, i32 0, i32 1
  store i32 64, ptr %tmp2, align 8
  %tmp3 = getelementptr inbounds %SmallPtrSet64, ptr %P, i64 0, i32 0, i32 4, i64 64
  store ptr null, ptr %tmp3, align 8
  %tmp4 = load i32, ptr %tmp2, align 8
	ret i32 %tmp4
; CHECK-LABEL: @test4(
; CHECK: ret i32 64
}

; P[i] != p[i+1]
define i32 @test5(ptr %p, i64 %i) {
  %pi = getelementptr i32, ptr %p, i64 %i
  %i.next = add i64 %i, 1
  %pi.next = getelementptr i32, ptr %p, i64 %i.next
  %x = load i32, ptr %pi
  store i32 42, ptr %pi.next
  %y = load i32, ptr %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test5(
; CHECK: ret i32 0
}

define i32 @test5_as1_smaller_size(ptr addrspace(1) %p, i8 %i) {
  %pi = getelementptr i32, ptr addrspace(1) %p, i8 %i
  %i.next = add i8 %i, 1
  %pi.next = getelementptr i32, ptr addrspace(1) %p, i8 %i.next
  %x = load i32, ptr addrspace(1) %pi
  store i32 42, ptr addrspace(1) %pi.next
  %y = load i32, ptr addrspace(1) %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test5_as1_smaller_size(
; CHECK: sext
; CHECK: ret i32 0
}

define i32 @test5_as1_same_size(ptr addrspace(1) %p, i16 %i) {
  %pi = getelementptr i32, ptr addrspace(1) %p, i16 %i
  %i.next = add i16 %i, 1
  %pi.next = getelementptr i32, ptr addrspace(1) %p, i16 %i.next
  %x = load i32, ptr addrspace(1) %pi
  store i32 42, ptr addrspace(1) %pi.next
  %y = load i32, ptr addrspace(1) %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test5_as1_same_size(
; CHECK: ret i32 0
}

; P[i] != p[(i*4)|1]
define i32 @test6(ptr %p, i64 %i1) {
  %i = shl i64 %i1, 2
  %pi = getelementptr i32, ptr %p, i64 %i
  %i.next = or i64 %i, 1
  %pi.next = getelementptr i32, ptr %p, i64 %i.next
  %x = load i32, ptr %pi
  store i32 42, ptr %pi.next
  %y = load i32, ptr %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test6(
; CHECK: ret i32 0
}

; P[1] != P[i*4]
define i32 @test7(ptr %p, i64 %i) {
  %pi = getelementptr i32, ptr %p, i64 1
  %i.next = shl i64 %i, 2
  %pi.next = getelementptr i32, ptr %p, i64 %i.next
  %x = load i32, ptr %pi
  store i32 42, ptr %pi.next
  %y = load i32, ptr %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test7(
; CHECK: ret i32 0
}

; P[zext(i)] != p[zext(i+1)]
; PR1143
define i32 @test8(ptr %p, i16 %i) {
  %i1 = zext i16 %i to i32
  %pi = getelementptr i32, ptr %p, i32 %i1
  %i.next = add i16 %i, 1
  %i.next2 = zext i16 %i.next to i32
  %pi.next = getelementptr i32, ptr %p, i32 %i.next2
  %x = load i32, ptr %pi
  store i32 42, ptr %pi.next
  %y = load i32, ptr %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test8(
; CHECK: ret i32 0
}

define i8 @test9(ptr %P, i32 %i, i32 %j) {
  %i2 = shl i32 %i, 2
  %i3 = add i32 %i2, 1
  ; P2 = P + 1 + 4*i
  %P2 = getelementptr [4 x i8], ptr %P, i32 0, i32 %i3

  %j2 = shl i32 %j, 2

  ; P4 = P + 4*j
  %P4 = getelementptr [4 x i8], ptr %P, i32 0, i32 %j2

  %x = load i8, ptr %P2
  store i8 42, ptr %P4
  %y = load i8, ptr %P2
  %z = sub i8 %x, %y
  ret i8 %z
; CHECK-LABEL: @test9(
; CHECK: ret i8 0
}

define i8 @test10(ptr %P, i32 %i) {
  %i2 = shl i32 %i, 2
  %i3 = add i32 %i2, 4
  ; P2 = P + 4 + 4*i
  %P2 = getelementptr [4 x i8], ptr %P, i32 0, i32 %i3

  ; P4 = P + 4*i
  %P4 = getelementptr [4 x i8], ptr %P, i32 0, i32 %i2

  %x = load i8, ptr %P2
  store i8 42, ptr %P4
  %y = load i8, ptr %P2
  %z = sub i8 %x, %y
  ret i8 %z
; CHECK-LABEL: @test10(
; CHECK: ret i8 0
}

; (This was a miscompilation.)
define float @test11(i32 %indvar, ptr %q) nounwind ssp {
  %tmp = mul i32 %indvar, -1
  %dec = add i32 %tmp, 3
  %scevgep = getelementptr [4 x [2 x float]], ptr %q, i32 0, i32 %dec
  %y29 = getelementptr inbounds [2 x float], ptr %q, i32 0, i32 1
  store float 1.0, ptr %y29, align 4
  store i64 0, ptr %scevgep, align 4
  %tmp30 = load float, ptr %y29, align 4
  ret float %tmp30
; CHECK-LABEL: @test11(
; CHECK: ret float %tmp30
}

; (This was a miscompilation.)
define i32 @test12(i32 %x, i32 %y, ptr %p) nounwind {
  %b = getelementptr [13 x i8], ptr %p, i32 %x
  %d = getelementptr [15 x i8], ptr %b, i32 %y, i32 8
  store i32 1, ptr %p
  store i32 0, ptr %d
  %r = load i32, ptr %p
  ret i32 %r
; CHECK-LABEL: @test12(
; CHECK: ret i32 %r
}

@P = internal global i32 715827882, align 4
@Q = internal global i32 715827883, align 4
@.str = private unnamed_addr constant [7 x i8] c"%u %u\0A\00", align 1

; Make sure we recognize that u[0] and u[Global + Cst] may alias
; when the addition has wrapping semantic.
; PR24468.
; CHECK-LABEL: @test13(
; Make sure the stores appear before the related loads.
; CHECK: store i8 42,
; CHECK: store i8 99,
; Find the loads and make sure they are used in the arguments to the printf.
; CHECK: [[T0:%[a-zA-Z0-9_]+]] = load i8, ptr %t, align 1
; CHECK: [[T0ARG:%[a-zA-Z0-9_]+]] = zext i8 [[T0]] to i32
; CHECK: [[U0:%[a-zA-Z0-9_]+]] = load i8, ptr %u, align 1
; CHECK: [[U0ARG:%[a-zA-Z0-9_]+]] = zext i8 [[U0]] to i32
; CHECK: call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 [[T0ARG]], i32 [[U0ARG]])
; CHECK: ret
define void @test13() {
entry:
  %t = alloca [3 x i8], align 1
  %u = alloca [3 x i8], align 1
  %tmp = load i32, ptr @P, align 4
  %tmp1 = mul i32 %tmp, 3
  %mul = add i32 %tmp1, -2147483646
  %idxprom = zext i32 %mul to i64
  %arrayidx = getelementptr inbounds [3 x i8], ptr %t, i64 0, i64 %idxprom
  store i8 42, ptr %arrayidx, align 1
  %tmp2 = load i32, ptr @Q, align 4
  %tmp3 = mul i32 %tmp2, 3
  %mul2 = add i32 %tmp3, 2147483647
  %idxprom3 = zext i32 %mul2 to i64
  %arrayidx4 = getelementptr inbounds [3 x i8], ptr %u, i64 0, i64 %idxprom3
  store i8 99, ptr %arrayidx4, align 1
  %tmp4 = load i8, ptr %t, align 1
  %conv = zext i8 %tmp4 to i32
  %tmp5 = load i8, ptr %u, align 1
  %conv7 = zext i8 %tmp5 to i32
  %call = call i32 (ptr, ...) @printf(ptr @.str, i32 %conv, i32 %conv7)
  ret void
}

declare i32 @printf(ptr, ...)
