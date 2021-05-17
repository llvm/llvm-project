; RUN: opt < %s -basic-aa -gvn -instcombine -S 2>&1 | FileCheck %s

target datalayout = "e-p:32:32:32-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

; Make sure that basicaa thinks R and r are must aliases.
define i32 @test1(i8 * %P) {
entry:
	%Q = bitcast i8* %P to {i32, i32}*
	%R = getelementptr {i32, i32}, {i32, i32}* %Q, i32 0, i32 1
	%S = load i32, i32* %R

	%q = bitcast i8* %P to {i32, i32}*
	%r = getelementptr {i32, i32}, {i32, i32}* %q, i32 0, i32 1
	%s = load i32, i32* %r

	%t = sub i32 %S, %s
	ret i32 %t
; CHECK-LABEL: @test1(
; CHECK: ret i32 0
}

define i32 @test2(i8 * %P) {
entry:
	%Q = bitcast i8* %P to {i32, i32, i32}*
	%R = getelementptr {i32, i32, i32}, {i32, i32, i32}* %Q, i32 0, i32 1
	%S = load i32, i32* %R

	%r = getelementptr {i32, i32, i32}, {i32, i32, i32}* %Q, i32 0, i32 2
  store i32 42, i32* %r

	%s = load i32, i32* %R

	%t = sub i32 %S, %s
	ret i32 %t
; CHECK-LABEL: @test2(
; CHECK: ret i32 0
}


; This was a miscompilation.
define i32 @test3({float, {i32, i32, i32}}* %P) {
entry:
  %P2 = getelementptr {float, {i32, i32, i32}}, {float, {i32, i32, i32}}* %P, i32 0, i32 1
	%R = getelementptr {i32, i32, i32}, {i32, i32, i32}* %P2, i32 0, i32 1
	%S = load i32, i32* %R

	%r = getelementptr {i32, i32, i32}, {i32, i32, i32}* %P2, i32 0, i32 2
  store i32 42, i32* %r

	%s = load i32, i32* %R

	%t = sub i32 %S, %s
	ret i32 %t
; CHECK-LABEL: @test3(
; CHECK: ret i32 0
}


;; This is reduced from the SmallPtrSet constructor.
%SmallPtrSetImpl = type { i8**, i32, i32, i32, [1 x i8*] }
%SmallPtrSet64 = type { %SmallPtrSetImpl, [64 x i8*] }

define i32 @test4(%SmallPtrSet64* %P) {
entry:
  %tmp2 = getelementptr inbounds %SmallPtrSet64, %SmallPtrSet64* %P, i64 0, i32 0, i32 1
  store i32 64, i32* %tmp2, align 8
  %tmp3 = getelementptr inbounds %SmallPtrSet64, %SmallPtrSet64* %P, i64 0, i32 0, i32 4, i64 64
  store i8* null, i8** %tmp3, align 8
  %tmp4 = load i32, i32* %tmp2, align 8
	ret i32 %tmp4
; CHECK-LABEL: @test4(
; CHECK: ret i32 64
}

; P[i] != p[i+1]
define i32 @test5(i32* %p, i64 %i) {
  %pi = getelementptr i32, i32* %p, i64 %i
  %i.next = add i64 %i, 1
  %pi.next = getelementptr i32, i32* %p, i64 %i.next
  %x = load i32, i32* %pi
  store i32 42, i32* %pi.next
  %y = load i32, i32* %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test5(
; CHECK: ret i32 0
}

define i32 @test5_as1_smaller_size(i32 addrspace(1)* %p, i8 %i) {
  %pi = getelementptr i32, i32 addrspace(1)* %p, i8 %i
  %i.next = add i8 %i, 1
  %pi.next = getelementptr i32, i32 addrspace(1)* %p, i8 %i.next
  %x = load i32, i32 addrspace(1)* %pi
  store i32 42, i32 addrspace(1)* %pi.next
  %y = load i32, i32 addrspace(1)* %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test5_as1_smaller_size(
; CHECK: sext
; CHECK: ret i32 0
}

define i32 @test5_as1_same_size(i32 addrspace(1)* %p, i16 %i) {
  %pi = getelementptr i32, i32 addrspace(1)* %p, i16 %i
  %i.next = add i16 %i, 1
  %pi.next = getelementptr i32, i32 addrspace(1)* %p, i16 %i.next
  %x = load i32, i32 addrspace(1)* %pi
  store i32 42, i32 addrspace(1)* %pi.next
  %y = load i32, i32 addrspace(1)* %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test5_as1_same_size(
; CHECK: ret i32 0
}

; P[i] != p[(i*4)|1]
define i32 @test6(i32* %p, i64 %i1) {
  %i = shl i64 %i1, 2
  %pi = getelementptr i32, i32* %p, i64 %i
  %i.next = or i64 %i, 1
  %pi.next = getelementptr i32, i32* %p, i64 %i.next
  %x = load i32, i32* %pi
  store i32 42, i32* %pi.next
  %y = load i32, i32* %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test6(
; CHECK: ret i32 0
}

; P[1] != P[i*4]
define i32 @test7(i32* %p, i64 %i) {
  %pi = getelementptr i32, i32* %p, i64 1
  %i.next = shl i64 %i, 2
  %pi.next = getelementptr i32, i32* %p, i64 %i.next
  %x = load i32, i32* %pi
  store i32 42, i32* %pi.next
  %y = load i32, i32* %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test7(
; CHECK: ret i32 0
}

; P[zext(i)] != p[zext(i+1)]
; PR1143
define i32 @test8(i32* %p, i16 %i) {
  %i1 = zext i16 %i to i32
  %pi = getelementptr i32, i32* %p, i32 %i1
  %i.next = add i16 %i, 1
  %i.next2 = zext i16 %i.next to i32
  %pi.next = getelementptr i32, i32* %p, i32 %i.next2
  %x = load i32, i32* %pi
  store i32 42, i32* %pi.next
  %y = load i32, i32* %pi
  %z = sub i32 %x, %y
  ret i32 %z
; CHECK-LABEL: @test8(
; CHECK: ret i32 0
}

define i8 @test9([4 x i8] *%P, i32 %i, i32 %j) {
  %i2 = shl i32 %i, 2
  %i3 = add i32 %i2, 1
  ; P2 = P + 1 + 4*i
  %P2 = getelementptr [4 x i8], [4 x i8] *%P, i32 0, i32 %i3

  %j2 = shl i32 %j, 2

  ; P4 = P + 4*j
  %P4 = getelementptr [4 x i8], [4 x i8]* %P, i32 0, i32 %j2

  %x = load i8, i8* %P2
  store i8 42, i8* %P4
  %y = load i8, i8* %P2
  %z = sub i8 %x, %y
  ret i8 %z
; CHECK-LABEL: @test9(
; CHECK: ret i8 0
}

define i8 @test10([4 x i8] *%P, i32 %i) {
  %i2 = shl i32 %i, 2
  %i3 = add i32 %i2, 4
  ; P2 = P + 4 + 4*i
  %P2 = getelementptr [4 x i8], [4 x i8] *%P, i32 0, i32 %i3

  ; P4 = P + 4*i
  %P4 = getelementptr [4 x i8], [4 x i8]* %P, i32 0, i32 %i2

  %x = load i8, i8* %P2
  store i8 42, i8* %P4
  %y = load i8, i8* %P2
  %z = sub i8 %x, %y
  ret i8 %z
; CHECK-LABEL: @test10(
; CHECK: ret i8 0
}

; (This was a miscompilation.)
define float @test11(i32 %indvar, [4 x [2 x float]]* %q) nounwind ssp {
  %tmp = mul i32 %indvar, -1
  %dec = add i32 %tmp, 3
  %scevgep = getelementptr [4 x [2 x float]], [4 x [2 x float]]* %q, i32 0, i32 %dec
  %scevgep35 = bitcast [2 x float]* %scevgep to i64*
  %arrayidx28 = getelementptr inbounds [4 x [2 x float]], [4 x [2 x float]]* %q, i32 0, i32 0
  %y29 = getelementptr inbounds [2 x float], [2 x float]* %arrayidx28, i32 0, i32 1
  store float 1.0, float* %y29, align 4
  store i64 0, i64* %scevgep35, align 4
  %tmp30 = load float, float* %y29, align 4
  ret float %tmp30
; CHECK-LABEL: @test11(
; CHECK: ret float %tmp30
}

; (This was a miscompilation.)
define i32 @test12(i32 %x, i32 %y, i8* %p) nounwind {
  %a = bitcast i8* %p to [13 x i8]*
  %b = getelementptr [13 x i8], [13 x i8]* %a, i32 %x
  %c = bitcast [13 x i8]* %b to [15 x i8]*
  %d = getelementptr [15 x i8], [15 x i8]* %c, i32 %y, i32 8
  %castd = bitcast i8* %d to i32*
  %castp = bitcast i8* %p to i32*
  store i32 1, i32* %castp
  store i32 0, i32* %castd
  %r = load i32, i32* %castp
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
; CHECK: [[T0ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds [3 x i8], [3 x i8]* %t, i32 0, i32 0
; CHECK: [[T0:%[a-zA-Z0-9_]+]] = load i8, i8* [[T0ADDR]], align 1
; CHECK: [[T0ARG:%[a-zA-Z0-9_]+]] = zext i8 [[T0]] to i32
; CHECK: [[U0ADDR:%[a-zA-Z0-9_]+]] = getelementptr inbounds [3 x i8], [3 x i8]* %u, i32 0, i32 0
; CHECK: [[U0:%[a-zA-Z0-9_]+]] = load i8, i8* [[U0ADDR]], align 1
; CHECK: [[U0ARG:%[a-zA-Z0-9_]+]] = zext i8 [[U0]] to i32
; CHECK: call i32 (i8*, ...) @printf(i8* noundef nonnull dereferenceable(1) getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0), i32 [[T0ARG]], i32 [[U0ARG]])
; CHECK: ret
define void @test13() {
entry:
  %t = alloca [3 x i8], align 1
  %u = alloca [3 x i8], align 1
  %tmp = load i32, i32* @P, align 4
  %tmp1 = mul i32 %tmp, 3
  %mul = add i32 %tmp1, -2147483646
  %idxprom = zext i32 %mul to i64
  %arrayidx = getelementptr inbounds [3 x i8], [3 x i8]* %t, i64 0, i64 %idxprom
  store i8 42, i8* %arrayidx, align 1
  %tmp2 = load i32, i32* @Q, align 4
  %tmp3 = mul i32 %tmp2, 3
  %mul2 = add i32 %tmp3, 2147483647
  %idxprom3 = zext i32 %mul2 to i64
  %arrayidx4 = getelementptr inbounds [3 x i8], [3 x i8]* %u, i64 0, i64 %idxprom3
  store i8 99, i8* %arrayidx4, align 1
  %arrayidx5 = getelementptr inbounds [3 x i8], [3 x i8]* %t, i64 0, i64 0
  %tmp4 = load i8, i8* %arrayidx5, align 1
  %conv = zext i8 %tmp4 to i32
  %arrayidx6 = getelementptr inbounds [3 x i8], [3 x i8]* %u, i64 0, i64 0
  %tmp5 = load i8, i8* %arrayidx6, align 1
  %conv7 = zext i8 %tmp5 to i32
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i64 0, i64 0), i32 %conv, i32 %conv7)
  ret void
}

declare i32 @printf(i8*, ...)
