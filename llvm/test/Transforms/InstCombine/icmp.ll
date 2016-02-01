; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-p1:16:16:16-p2:32:32:32-p3:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

define i32 @test1(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK-LABEL: @test1(
; CHECK: lshr i32 %X, 31
; CHECK-NEXT: ret i32
}

define i32 @test2(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        zext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK-LABEL: @test2(
; CHECK: lshr i32 %X, 31
; CHECK-NEXT: xor i32
; CHECK-NEXT: ret i32
}

define i32 @test3(i32 %X) {
entry:
        icmp slt i32 %X, 0              ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK-LABEL: @test3(
; CHECK: ashr i32 %X, 31
; CHECK-NEXT: ret i32
}

define i32 @test4(i32 %X) {
entry:
        icmp ult i32 %X, -2147483648            ; <i1>:0 [#uses=1]
        sext i1 %0 to i32               ; <i32>:1 [#uses=1]
        ret i32 %1
; CHECK-LABEL: @test4(
; CHECK: ashr i32 %X, 31
; CHECK-NEXT: xor i32
; CHECK-NEXT: ret i32
}

; PR4837
define <2 x i1> @test5(<2 x i64> %x) {
entry:
  %V = icmp eq <2 x i64> %x, undef
  ret <2 x i1> %V
; CHECK-LABEL: @test5(
; CHECK: ret <2 x i1> <i1 true, i1 true>
}

define i32 @test6(i32 %a, i32 %b) {
        %c = icmp sle i32 %a, -1
        %d = zext i1 %c to i32
        %e = sub i32 0, %d
        %f = and i32 %e, %b
        ret i32 %f
; CHECK-LABEL: @test6(
; CHECK-NEXT: ashr i32 %a, 31
; CHECK-NEXT: %f = and i32 %e, %b
; CHECK-NEXT: ret i32 %f
}


define i1 @test7(i32 %x) {
entry:
  %a = add i32 %x, -1
  %b = icmp ult i32 %a, %x
  ret i1 %b
; CHECK-LABEL: @test7(
; CHECK: %b = icmp ne i32 %x, 0
; CHECK: ret i1 %b
}

define i1 @test8(i32 %x){
entry:
  %a = add i32 %x, -1
  %b = icmp eq i32 %a, %x
  ret i1 %b
; CHECK-LABEL: @test8(
; CHECK: ret i1 false
}

define i1 @test9(i32 %x)  {
entry:
  %a = add i32 %x, -2
  %b = icmp ugt i32 %x, %a
  ret i1 %b
; CHECK-LABEL: @test9(
; CHECK: icmp ugt i32 %x, 1
; CHECK: ret i1 %b
}

define i1 @test10(i32 %x){
entry:
  %a = add i32 %x, -1
  %b = icmp slt i32 %a, %x
  ret i1 %b
; CHECK-LABEL: @test10(
; CHECK: %b = icmp ne i32 %x, -2147483648
; CHECK: ret i1 %b
}

define i1 @test11(i32 %x) {
  %a = add nsw i32 %x, 8
  %b = icmp slt i32 %x, %a
  ret i1 %b
; CHECK-LABEL: @test11(
; CHECK: ret i1 true
}

; PR6195
define i1 @test12(i1 %A) {
  %S = select i1 %A, i64 -4294967295, i64 8589934591
  %B = icmp ne i64 bitcast (<2 x i32> <i32 1, i32 -1> to i64), %S
  ret i1 %B
; CHECK-LABEL: @test12(
; CHECK-NEXT: = xor i1 %A, true
; CHECK-NEXT: ret i1
}

; PR6481
define i1 @test13(i8 %X) nounwind readnone {
entry:
        %cmp = icmp slt i8 undef, %X
        ret i1 %cmp
; CHECK-LABEL: @test13(
; CHECK: ret i1 false
}

define i1 @test14(i8 %X) nounwind readnone {
entry:
        %cmp = icmp slt i8 undef, -128
        ret i1 %cmp
; CHECK-LABEL: @test14(
; CHECK: ret i1 false
}

define i1 @test15() nounwind readnone {
entry:
        %cmp = icmp eq i8 undef, -128
        ret i1 %cmp
; CHECK-LABEL: @test15(
; CHECK: ret i1 undef
}

define i1 @test16() nounwind readnone {
entry:
        %cmp = icmp ne i8 undef, -128
        ret i1 %cmp
; CHECK-LABEL: @test16(
; CHECK: ret i1 undef
}

define i1 @test17(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 8
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
; CHECK-LABEL: @test17(
; CHECK-NEXT: %cmp = icmp ne i32 %x, 3
}

define i1 @test17a(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 7
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
; CHECK-LABEL: @test17a(
; CHECK-NEXT: %cmp = icmp ugt i32 %x, 2
}

define i1 @test18(i32 %x) nounwind {
  %sh = lshr i32 8, %x
  %and = and i32 %sh, 1
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
; CHECK-LABEL: @test18(
; CHECK-NEXT: %cmp = icmp ne i32 %x, 3
}

define i1 @test19(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 8
  %cmp = icmp eq i32 %and, 8
  ret i1 %cmp
; CHECK-LABEL: @test19(
; CHECK-NEXT: %cmp = icmp eq i32 %x, 3
}

define i1 @test20(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 8
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
; CHECK-LABEL: @test20(
; CHECK-NEXT: %cmp = icmp eq i32 %x, 3
}

define i1 @test20a(i32 %x) nounwind {
  %shl = shl i32 1, %x
  %and = and i32 %shl, 7
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
; CHECK-LABEL: @test20a(
; CHECK-NEXT: %cmp = icmp ult i32 %x, 3
}

define i1 @test21(i8 %x, i8 %y) {
; CHECK-LABEL: @test21(
; CHECK-NOT: or i8
; CHECK: icmp ugt
  %A = or i8 %x, 1
  %B = icmp ugt i8 %A, 3
  ret i1 %B
}

define i1 @test22(i8 %x, i8 %y) {
; CHECK-LABEL: @test22(
; CHECK-NOT: or i8
; CHECK: icmp ult
  %A = or i8 %x, 1
  %B = icmp ult i8 %A, 4
  ret i1 %B
}

; PR2740
; CHECK-LABEL: @test23(
; CHECK: icmp sgt i32 %x, 1328634634
define i1 @test23(i32 %x) nounwind {
	%i3 = sdiv i32 %x, -1328634635
	%i4 = icmp eq i32 %i3, -1
	ret i1 %i4
}

@X = global [1000 x i32] zeroinitializer

; PR8882
; CHECK-LABEL: @test24(
; CHECK:    %cmp = icmp eq i64 %i, 1000
; CHECK:   ret i1 %cmp
define i1 @test24(i64 %i) {
  %p1 = getelementptr inbounds i32, i32* getelementptr inbounds ([1000 x i32], [1000 x i32]* @X, i64 0, i64 0), i64 %i
  %cmp = icmp eq i32* %p1, getelementptr inbounds ([1000 x i32], [1000 x i32]* @X, i64 1, i64 0)
  ret i1 %cmp
}

@X_as1 = addrspace(1) global [1000 x i32] zeroinitializer

; CHECK: @test24_as1
; CHECK: trunc i64 %i to i16
; CHECK: %cmp = icmp eq i16 %1, 1000
; CHECK: ret i1 %cmp
define i1 @test24_as1(i64 %i) {
  %p1 = getelementptr inbounds i32, i32 addrspace(1)* getelementptr inbounds ([1000 x i32], [1000 x i32] addrspace(1)* @X_as1, i64 0, i64 0), i64 %i
  %cmp = icmp eq i32 addrspace(1)* %p1, getelementptr inbounds ([1000 x i32], [1000 x i32] addrspace(1)* @X_as1, i64 1, i64 0)
  ret i1 %cmp
}

; CHECK-LABEL: @test25(
; X + Z > Y + Z -> X > Y if there is no overflow.
; CHECK: %c = icmp sgt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test25(i32 %x, i32 %y, i32 %z) {
  %lhs = add nsw i32 %x, %z
  %rhs = add nsw i32 %y, %z
  %c = icmp sgt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test26(
; X + Z > Y + Z -> X > Y if there is no overflow.
; CHECK: %c = icmp ugt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test26(i32 %x, i32 %y, i32 %z) {
  %lhs = add nuw i32 %x, %z
  %rhs = add nuw i32 %y, %z
  %c = icmp ugt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test27(
; X - Z > Y - Z -> X > Y if there is no overflow.
; CHECK: %c = icmp sgt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test27(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nsw i32 %x, %z
  %rhs = sub nsw i32 %y, %z
  %c = icmp sgt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test28(
; X - Z > Y - Z -> X > Y if there is no overflow.
; CHECK: %c = icmp ugt i32 %x, %y
; CHECK: ret i1 %c
define i1 @test28(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nuw i32 %x, %z
  %rhs = sub nuw i32 %y, %z
  %c = icmp ugt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test29(
; X + Y > X -> Y > 0 if there is no overflow.
; CHECK: %c = icmp sgt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test29(i32 %x, i32 %y) {
  %lhs = add nsw i32 %x, %y
  %c = icmp sgt i32 %lhs, %x
  ret i1 %c
}

; CHECK-LABEL: @test30(
; X + Y > X -> Y > 0 if there is no overflow.
; CHECK: %c = icmp ne i32 %y, 0
; CHECK: ret i1 %c
define i1 @test30(i32 %x, i32 %y) {
  %lhs = add nuw i32 %x, %y
  %c = icmp ugt i32 %lhs, %x
  ret i1 %c
}

; CHECK-LABEL: @test31(
; X > X + Y -> 0 > Y if there is no overflow.
; CHECK: %c = icmp slt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test31(i32 %x, i32 %y) {
  %rhs = add nsw i32 %x, %y
  %c = icmp sgt i32 %x, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test32(
; X > X + Y -> 0 > Y if there is no overflow.
; CHECK: ret i1 false
define i1 @test32(i32 %x, i32 %y) {
  %rhs = add nuw i32 %x, %y
  %c = icmp ugt i32 %x, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test33(
; X - Y > X -> 0 > Y if there is no overflow.
; CHECK: %c = icmp slt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test33(i32 %x, i32 %y) {
  %lhs = sub nsw i32 %x, %y
  %c = icmp sgt i32 %lhs, %x
  ret i1 %c
}

; CHECK-LABEL: @test34(
; X - Y > X -> 0 > Y if there is no overflow.
; CHECK: ret i1 false
define i1 @test34(i32 %x, i32 %y) {
  %lhs = sub nuw i32 %x, %y
  %c = icmp ugt i32 %lhs, %x
  ret i1 %c
}

; CHECK-LABEL: @test35(
; X > X - Y -> Y > 0 if there is no overflow.
; CHECK: %c = icmp sgt i32 %y, 0
; CHECK: ret i1 %c
define i1 @test35(i32 %x, i32 %y) {
  %rhs = sub nsw i32 %x, %y
  %c = icmp sgt i32 %x, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test36(
; X > X - Y -> Y > 0 if there is no overflow.
; CHECK: %c = icmp ne i32 %y, 0
; CHECK: ret i1 %c
define i1 @test36(i32 %x, i32 %y) {
  %rhs = sub nuw i32 %x, %y
  %c = icmp ugt i32 %x, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test37(
; X - Y > X - Z -> Z > Y if there is no overflow.
; CHECK: %c = icmp sgt i32 %z, %y
; CHECK: ret i1 %c
define i1 @test37(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nsw i32 %x, %y
  %rhs = sub nsw i32 %x, %z
  %c = icmp sgt i32 %lhs, %rhs
  ret i1 %c
}

; CHECK-LABEL: @test38(
; X - Y > X - Z -> Z > Y if there is no overflow.
; CHECK: %c = icmp ugt i32 %z, %y
; CHECK: ret i1 %c
define i1 @test38(i32 %x, i32 %y, i32 %z) {
  %lhs = sub nuw i32 %x, %y
  %rhs = sub nuw i32 %x, %z
  %c = icmp ugt i32 %lhs, %rhs
  ret i1 %c
}

; PR9343 #1
; CHECK-LABEL: @test39(
; CHECK: %B = icmp eq i32 %X, 0
define i1 @test39(i32 %X, i32 %Y) {
  %A = ashr exact i32 %X, %Y
  %B = icmp eq i32 %A, 0
  ret i1 %B
}

; CHECK-LABEL: @test40(
; CHECK: %B = icmp ne i32 %X, 0
define i1 @test40(i32 %X, i32 %Y) {
  %A = lshr exact i32 %X, %Y
  %B = icmp ne i32 %A, 0
  ret i1 %B
}

; PR9343 #3
; CHECK-LABEL: @test41(
; CHECK: ret i1 true
define i1 @test41(i32 %X, i32 %Y) {
  %A = urem i32 %X, %Y
  %B = icmp ugt i32 %Y, %A
  ret i1 %B
}

; CHECK-LABEL: @test42(
; CHECK: %B = icmp sgt i32 %Y, -1
define i1 @test42(i32 %X, i32 %Y) {
  %A = srem i32 %X, %Y
  %B = icmp slt i32 %A, %Y
  ret i1 %B
}

; CHECK-LABEL: @test43(
; CHECK: %B = icmp slt i32 %Y, 0
define i1 @test43(i32 %X, i32 %Y) {
  %A = srem i32 %X, %Y
  %B = icmp slt i32 %Y, %A
  ret i1 %B
}

; CHECK-LABEL: @test44(
; CHECK: %B = icmp sgt i32 %Y, -1
define i1 @test44(i32 %X, i32 %Y) {
  %A = srem i32 %X, %Y
  %B = icmp slt i32 %A, %Y
  ret i1 %B
}

; CHECK-LABEL: @test45(
; CHECK: %B = icmp slt i32 %Y, 0
define i1 @test45(i32 %X, i32 %Y) {
  %A = srem i32 %X, %Y
  %B = icmp slt i32 %Y, %A
  ret i1 %B
}

; PR9343 #4
; CHECK-LABEL: @test46(
; CHECK: %C = icmp ult i32 %X, %Y
define i1 @test46(i32 %X, i32 %Y, i32 %Z) {
  %A = ashr exact i32 %X, %Z
  %B = ashr exact i32 %Y, %Z
  %C = icmp ult i32 %A, %B
  ret i1 %C
}

; PR9343 #5
; CHECK-LABEL: @test47(
; CHECK: %C = icmp ugt i32 %X, %Y
define i1 @test47(i32 %X, i32 %Y, i32 %Z) {
  %A = ashr exact i32 %X, %Z
  %B = ashr exact i32 %Y, %Z
  %C = icmp ugt i32 %A, %B
  ret i1 %C
}

; PR9343 #8
; CHECK-LABEL: @test48(
; CHECK: %C = icmp eq i32 %X, %Y
define i1 @test48(i32 %X, i32 %Y, i32 %Z) {
  %A = sdiv exact i32 %X, %Z
  %B = sdiv exact i32 %Y, %Z
  %C = icmp eq i32 %A, %B
  ret i1 %C
}

; PR8469
; CHECK-LABEL: @test49(
; CHECK: ret <2 x i1> <i1 true, i1 true>
define <2 x i1> @test49(<2 x i32> %tmp3) {
entry:
  %tmp11 = and <2 x i32> %tmp3, <i32 3, i32 3>
  %cmp = icmp ult <2 x i32> %tmp11, <i32 4, i32 4>
  ret <2 x i1> %cmp
}

; PR9343 #7
; CHECK-LABEL: @test50(
; CHECK: ret i1 true
define i1 @test50(i16 %X, i32 %Y) {
  %A = zext i16 %X to i32
  %B = srem i32 %A, %Y
  %C = icmp sgt i32 %B, -1
  ret i1 %C
}

; CHECK-LABEL: @test51(
; CHECK: ret i1 %C
define i1 @test51(i32 %X, i32 %Y) {
  %A = and i32 %X, 2147483648
  %B = srem i32 %A, %Y
  %C = icmp sgt i32 %B, -1
  ret i1 %C
}

; CHECK-LABEL: @test52(
; CHECK-NEXT: and i32 %x1, 16711935
; CHECK-NEXT: icmp eq i32 {{.*}}, 4980863
; CHECK-NEXT: ret i1
define i1 @test52(i32 %x1) nounwind {
  %conv = and i32 %x1, 255
  %cmp = icmp eq i32 %conv, 127
  %tmp2 = lshr i32 %x1, 16
  %tmp3 = trunc i32 %tmp2 to i8
  %cmp15 = icmp eq i8 %tmp3, 76

  %A = and i1 %cmp, %cmp15
  ret i1 %A
}

; PR9838
; CHECK-LABEL: @test53(
; CHECK-NEXT: sdiv exact
; CHECK-NEXT: sdiv
; CHECK-NEXT: icmp
define i1 @test53(i32 %a, i32 %b) nounwind {
 %x = sdiv exact i32 %a, 30
 %y = sdiv i32 %b, 30
 %z = icmp eq i32 %x, %y
 ret i1 %z
}

; CHECK-LABEL: @test54(
; CHECK-NEXT: %and = and i8 %a, -64
; CHECK-NEXT: icmp eq i8 %and, -128
define i1 @test54(i8 %a) nounwind {
  %ext = zext i8 %a to i32
  %and = and i32 %ext, 192
  %ret = icmp eq i32 %and, 128
  ret i1 %ret
}

; CHECK-LABEL: @test55(
; CHECK-NEXT: icmp eq i32 %a, -123
define i1 @test55(i32 %a) {
  %sub = sub i32 0, %a
  %cmp = icmp eq i32 %sub, 123
  ret i1 %cmp
}

; CHECK-LABEL: @test56(
; CHECK-NEXT: icmp eq i32 %a, -113
define i1 @test56(i32 %a) {
  %sub = sub i32 10, %a
  %cmp = icmp eq i32 %sub, 123
  ret i1 %cmp
}

; PR10267 Don't make icmps more expensive when no other inst is subsumed.
declare void @foo(i32)
; CHECK-LABEL: @test57(
; CHECK: %and = and i32 %a, -2
; CHECK: %cmp = icmp ne i32 %and, 0
define i1 @test57(i32 %a) {
  %and = and i32 %a, -2
  %cmp = icmp ne i32 %and, 0
  call void @foo(i32 %and)
  ret i1 %cmp
}

; rdar://problem/10482509
; CHECK-LABEL: @cmpabs1(
; CHECK-NEXT: icmp ne
define zeroext i1 @cmpabs1(i64 %val) {
  %sub = sub nsw i64 0, %val
  %cmp = icmp slt i64 %val, 0
  %sub.val = select i1 %cmp, i64 %sub, i64 %val
  %tobool = icmp ne i64 %sub.val, 0
  ret i1 %tobool
}

; CHECK-LABEL: @cmpabs2(
; CHECK-NEXT: icmp ne
define zeroext i1 @cmpabs2(i64 %val) {
  %sub = sub nsw i64 0, %val
  %cmp = icmp slt i64 %val, 0
  %sub.val = select i1 %cmp, i64 %val, i64 %sub
  %tobool = icmp ne i64 %sub.val, 0
  ret i1 %tobool
}

; CHECK-LABEL: @test58(
; CHECK-NEXT: call i32 @test58_d(i64 36029346783166592)
define void @test58() nounwind {
  %cast = bitcast <1 x i64> <i64 36029346783166592> to i64
  %call = call i32 @test58_d( i64 %cast) nounwind
  ret void
}
declare i32 @test58_d(i64)

define i1 @test59(i8* %foo) {
  %bit = bitcast i8* %foo to i32*
  %gep1 = getelementptr inbounds i32, i32* %bit, i64 2
  %gep2 = getelementptr inbounds i8, i8* %foo, i64 10
  %cast1 = bitcast i32* %gep1 to i8*
  %cmp = icmp ult i8* %cast1, %gep2
  %use = ptrtoint i8* %cast1 to i64
  %call = call i32 @test58_d(i64 %use) nounwind
  ret i1 %cmp
; CHECK-LABEL: @test59(
; CHECK: ret i1 true
}

define i1 @test59_as1(i8 addrspace(1)* %foo) {
  %bit = bitcast i8 addrspace(1)* %foo to i32 addrspace(1)*
  %gep1 = getelementptr inbounds i32, i32 addrspace(1)* %bit, i64 2
  %gep2 = getelementptr inbounds i8, i8 addrspace(1)* %foo, i64 10
  %cast1 = bitcast i32 addrspace(1)* %gep1 to i8 addrspace(1)*
  %cmp = icmp ult i8 addrspace(1)* %cast1, %gep2
  %use = ptrtoint i8 addrspace(1)* %cast1 to i64
  %call = call i32 @test58_d(i64 %use) nounwind
  ret i1 %cmp
; CHECK: @test59_as1
; CHECK: %[[GEP:.+]] = getelementptr inbounds i8, i8 addrspace(1)* %foo, i16 8
; CHECK: ptrtoint i8 addrspace(1)* %[[GEP]] to i16
; CHECK: ret i1 true
}

define i1 @test60(i8* %foo, i64 %i, i64 %j) {
  %bit = bitcast i8* %foo to i32*
  %gep1 = getelementptr inbounds i32, i32* %bit, i64 %i
  %gep2 = getelementptr inbounds i8, i8* %foo, i64 %j
  %cast1 = bitcast i32* %gep1 to i8*
  %cmp = icmp ult i8* %cast1, %gep2
  ret i1 %cmp
; CHECK-LABEL: @test60(
; CHECK-NEXT: %gep1.idx = shl nuw i64 %i, 2
; CHECK-NEXT: icmp slt i64 %gep1.idx, %j
; CHECK-NEXT: ret i1
}

define i1 @test60_as1(i8 addrspace(1)* %foo, i64 %i, i64 %j) {
  %bit = bitcast i8 addrspace(1)* %foo to i32 addrspace(1)*
  %gep1 = getelementptr inbounds i32, i32 addrspace(1)* %bit, i64 %i
  %gep2 = getelementptr inbounds i8, i8 addrspace(1)* %foo, i64 %j
  %cast1 = bitcast i32 addrspace(1)* %gep1 to i8 addrspace(1)*
  %cmp = icmp ult i8 addrspace(1)* %cast1, %gep2
  ret i1 %cmp
; CHECK: @test60_as1
; CHECK: trunc i64 %i to i16
; CHECK: trunc i64 %j to i16
; CHECK: %gep1.idx = shl nuw i16 %{{.+}}, 2
; CHECK-NEXT: icmp sgt i16 %{{.+}}, %gep1.idx
; CHECK-NEXT: ret i1
}

; Same as test60, but look through an addrspacecast instead of a
; bitcast. This uses the same sized addrspace.
define i1 @test60_addrspacecast(i8* %foo, i64 %i, i64 %j) {
  %bit = addrspacecast i8* %foo to i32 addrspace(3)*
  %gep1 = getelementptr inbounds i32, i32 addrspace(3)* %bit, i64 %i
  %gep2 = getelementptr inbounds i8, i8* %foo, i64 %j
  %cast1 = addrspacecast i32 addrspace(3)* %gep1 to i8*
  %cmp = icmp ult i8* %cast1, %gep2
  ret i1 %cmp
; CHECK-LABEL: @test60_addrspacecast(
; CHECK-NEXT: %gep1.idx = shl nuw i64 %i, 2
; CHECK-NEXT: icmp slt i64 %gep1.idx, %j
; CHECK-NEXT: ret i1
}

define i1 @test60_addrspacecast_smaller(i8* %foo, i16 %i, i64 %j) {
  %bit = addrspacecast i8* %foo to i32 addrspace(1)*
  %gep1 = getelementptr inbounds i32, i32 addrspace(1)* %bit, i16 %i
  %gep2 = getelementptr inbounds i8, i8* %foo, i64 %j
  %cast1 = addrspacecast i32 addrspace(1)* %gep1 to i8*
  %cmp = icmp ult i8* %cast1, %gep2
  ret i1 %cmp
; CHECK-LABEL: @test60_addrspacecast_smaller(
; CHECK-NEXT: %gep1.idx = shl nuw i16 %i, 2
; CHECK-NEXT: trunc i64 %j to i16
; CHECK-NEXT: icmp sgt i16 %1, %gep1.idx
; CHECK-NEXT: ret i1
}

define i1 @test60_addrspacecast_larger(i8 addrspace(1)* %foo, i32 %i, i16 %j) {
  %bit = addrspacecast i8 addrspace(1)* %foo to i32 addrspace(2)*
  %gep1 = getelementptr inbounds i32, i32 addrspace(2)* %bit, i32 %i
  %gep2 = getelementptr inbounds i8, i8 addrspace(1)* %foo, i16 %j
  %cast1 = addrspacecast i32 addrspace(2)* %gep1 to i8 addrspace(1)*
  %cmp = icmp ult i8 addrspace(1)* %cast1, %gep2
  ret i1 %cmp
; CHECK-LABEL: @test60_addrspacecast_larger(
; CHECK-NEXT:  %gep1.idx = shl nuw i32 %i, 2
; CHECK-NEXT:  trunc i32 %gep1.idx to i16
; CHECK-NEXT:  icmp slt i16 %1, %j
; CHECK-NEXT:  ret i1
}

define i1 @test61(i8* %foo, i64 %i, i64 %j) {
  %bit = bitcast i8* %foo to i32*
  %gep1 = getelementptr i32, i32* %bit, i64 %i
  %gep2 = getelementptr  i8,  i8* %foo, i64 %j
  %cast1 = bitcast i32* %gep1 to i8*
  %cmp = icmp ult i8* %cast1, %gep2
  ret i1 %cmp
; Don't transform non-inbounds GEPs.
; CHECK-LABEL: @test61(
; CHECK: icmp ult i8* %cast1, %gep2
; CHECK-NEXT: ret i1
}

define i1 @test61_as1(i8 addrspace(1)* %foo, i16 %i, i16 %j) {
  %bit = bitcast i8 addrspace(1)* %foo to i32 addrspace(1)*
  %gep1 = getelementptr i32, i32 addrspace(1)* %bit, i16 %i
  %gep2 = getelementptr i8, i8 addrspace(1)* %foo, i16 %j
  %cast1 = bitcast i32 addrspace(1)* %gep1 to i8 addrspace(1)*
  %cmp = icmp ult i8 addrspace(1)* %cast1, %gep2
  ret i1 %cmp
; Don't transform non-inbounds GEPs.
; CHECK: @test61_as1
; CHECK: icmp ult i8 addrspace(1)* %cast1, %gep2
; CHECK-NEXT: ret i1
}

define i1 @test62(i8* %a) {
  %arrayidx1 = getelementptr inbounds i8, i8* %a, i64 1
  %arrayidx2 = getelementptr inbounds i8, i8* %a, i64 10
  %cmp = icmp slt i8* %arrayidx1, %arrayidx2
  ret i1 %cmp
; CHECK-LABEL: @test62(
; CHECK-NEXT: ret i1 true
}

define i1 @test62_as1(i8 addrspace(1)* %a) {
; CHECK-LABEL: @test62_as1(
; CHECK-NEXT: ret i1 true
  %arrayidx1 = getelementptr inbounds i8, i8 addrspace(1)* %a, i64 1
  %arrayidx2 = getelementptr inbounds i8, i8 addrspace(1)* %a, i64 10
  %cmp = icmp slt i8 addrspace(1)* %arrayidx1, %arrayidx2
  ret i1 %cmp
}

define i1 @test63(i8 %a, i32 %b) nounwind {
  %z = zext i8 %a to i32
  %t = and i32 %b, 255
  %c = icmp eq i32 %z, %t
  ret i1 %c
; CHECK-LABEL: @test63(
; CHECK-NEXT: %1 = trunc i32 %b to i8
; CHECK-NEXT: %c = icmp eq i8 %1, %a
; CHECK-NEXT: ret i1 %c
}

define i1 @test64(i8 %a, i32 %b) nounwind {
  %t = and i32 %b, 255
  %z = zext i8 %a to i32
  %c = icmp eq i32 %t, %z
  ret i1 %c
; CHECK-LABEL: @test64(
; CHECK-NEXT: %1 = trunc i32 %b to i8
; CHECK-NEXT: %c = icmp eq i8 %1, %a
; CHECK-NEXT: ret i1 %c
}

define i1 @test65(i64 %A, i64 %B) {
  %s1 = add i64 %A, %B
  %s2 = add i64 %A, %B
  %cmp = icmp eq i64 %s1, %s2
; CHECK-LABEL: @test65(
; CHECK-NEXT: ret i1 true
  ret i1 %cmp
}

define i1 @test66(i64 %A, i64 %B) {
  %s1 = add i64 %A, %B
  %s2 = add i64 %B, %A
  %cmp = icmp eq i64 %s1, %s2
; CHECK-LABEL: @test66(
; CHECK-NEXT: ret i1 true
  ret i1 %cmp
}

; CHECK-LABEL: @test67(
; CHECK: %and = and i32 %x, 96
; CHECK: %cmp = icmp ne i32 %and, 0
define i1 @test67(i32 %x) nounwind uwtable {
  %and = and i32 %x, 127
  %cmp = icmp sgt i32 %and, 31
  ret i1 %cmp
}

; CHECK-LABEL: @test68(
; CHECK: %cmp = icmp ugt i32 %and, 30
define i1 @test68(i32 %x) nounwind uwtable {
  %and = and i32 %x, 127
  %cmp = icmp sgt i32 %and, 30
  ret i1 %cmp
}

; PR14708
; CHECK-LABEL: @test69(
; CHECK: %1 = or i32 %c, 32
; CHECK: %2 = icmp eq i32 %1, 97
; CHECK: ret i1 %2
define i1 @test69(i32 %c) nounwind uwtable {
  %1 = icmp eq i32 %c, 97
  %2 = icmp eq i32 %c, 65
  %3 = or i1 %1, %2
  ret i1 %3
}

; PR15940
; CHECK-LABEL: @test70(
; CHECK-NEXT: %A = srem i32 5, %X
; CHECK-NEXT: %C = icmp ne i32 %A, 2
; CHECK-NEXT: ret i1 %C
define i1 @test70(i32 %X) {
  %A = srem i32 5, %X
  %B = add i32 %A, 2
  %C = icmp ne i32 %B, 4
  ret i1 %C
}

; CHECK-LABEL: @icmp_sext16trunc(
; CHECK-NEXT: %1 = trunc i32 %x to i16
; CHECK-NEXT: %cmp = icmp slt i16 %1, 36
define i1 @icmp_sext16trunc(i32 %x) {
  %trunc = trunc i32 %x to i16
  %sext = sext i16 %trunc to i32
  %cmp = icmp slt i32 %sext, 36
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_sext8trunc(
; CHECK-NEXT: %1 = trunc i32 %x to i8
; CHECK-NEXT: %cmp = icmp slt i8 %1, 36
define i1 @icmp_sext8trunc(i32 %x) {
  %trunc = trunc i32 %x to i8
  %sext = sext i8 %trunc to i32
  %cmp = icmp slt i32 %sext, 36
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl16(
; CHECK-NEXT: %1 = trunc i32 %x to i16
; CHECK-NEXT: %cmp = icmp slt i16 %1, 36
define i1 @icmp_shl16(i32 %x) {
  %shl = shl i32 %x, 16
  %cmp = icmp slt i32 %shl, 2359296
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl24(
; CHECK-NEXT: %1 = trunc i32 %x to i8
; CHECK-NEXT: %cmp = icmp slt i8 %1, 36
define i1 @icmp_shl24(i32 %x) {
  %shl = shl i32 %x, 24
  %cmp = icmp slt i32 %shl, 603979776
  ret i1 %cmp
}

; If the (shl x, C) preserved the sign and this is a sign test,
; compare the LHS operand instead
; CHECK-LABEL: @icmp_shl_nsw_sgt(
; CHECK-NEXT: icmp sgt i32 %x, 0
define i1 @icmp_shl_nsw_sgt(i32 %x) {
  %shl = shl nsw i32 %x, 21
  %cmp = icmp sgt i32 %shl, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_nsw_sge0(
; CHECK-NEXT: icmp sgt i32 %x, -1
define i1 @icmp_shl_nsw_sge0(i32 %x) {
  %shl = shl nsw i32 %x, 21
  %cmp = icmp sge i32 %shl, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_nsw_sge1(
; CHECK-NEXT: icmp sgt i32 %x, 0
define i1 @icmp_shl_nsw_sge1(i32 %x) {
  %shl = shl nsw i32 %x, 21
  %cmp = icmp sge i32 %shl, 1
  ret i1 %cmp
}

; Checks for icmp (eq|ne) (shl x, C), 0
; CHECK-LABEL: @icmp_shl_nsw_eq(
; CHECK-NEXT: icmp eq i32 %x, 0
define i1 @icmp_shl_nsw_eq(i32 %x) {
  %mul = shl nsw i32 %x, 5
  %cmp = icmp eq i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_eq(
; CHECK-NOT: icmp eq i32 %mul, 0
define i1 @icmp_shl_eq(i32 %x) {
  %mul = shl i32 %x, 5
  %cmp = icmp eq i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_nsw_ne(
; CHECK-NEXT: icmp ne i32 %x, 0
define i1 @icmp_shl_nsw_ne(i32 %x) {
  %mul = shl nsw i32 %x, 7
  %cmp = icmp ne i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_ne(
; CHECK-NOT: icmp ne i32 %x, 0
define i1 @icmp_shl_ne(i32 %x) {
  %mul = shl i32 %x, 7
  %cmp = icmp ne i32 %mul, 0
  ret i1 %cmp
}

; If the (mul x, C) preserved the sign and this is sign test,
; compare the LHS operand instead
; CHECK-LABEL: @icmp_mul_nsw(
; CHECK-NEXT: icmp sgt i32 %x, 0
define i1 @icmp_mul_nsw(i32 %x) {
  %mul = mul nsw i32 %x, 12
  %cmp = icmp sgt i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul_nsw1(
; CHECK-NEXT: icmp slt i32 %x, 0
define i1 @icmp_mul_nsw1(i32 %x) {
  %mul = mul nsw i32 %x, 12
  %cmp = icmp sle i32 %mul, -1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul_nsw_neg(
; CHECK-NEXT: icmp slt i32 %x, 1
define i1 @icmp_mul_nsw_neg(i32 %x) {
  %mul = mul nsw i32 %x, -12
  %cmp = icmp sge i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul_nsw_neg1(
; CHECK-NEXT: icmp slt i32 %x, 0
define i1 @icmp_mul_nsw_neg1(i32 %x) {
  %mul = mul nsw i32 %x, -12
  %cmp = icmp sge i32 %mul, 1
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul_nsw_0(
; CHECK-NOT: icmp sgt i32 %x, 0
define i1 @icmp_mul_nsw_0(i32 %x) {
  %mul = mul nsw i32 %x, 0
  %cmp = icmp sgt i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul(
; CHECK-NEXT: %mul = mul i32 %x, -12
define i1 @icmp_mul(i32 %x) {
  %mul = mul i32 %x, -12
  %cmp = icmp sge i32 %mul, 0
  ret i1 %cmp
}

; Checks for icmp (eq|ne) (mul x, C), 0
; CHECK-LABEL: @icmp_mul_neq0(
; CHECK-NEXT: icmp ne i32 %x, 0
define i1 @icmp_mul_neq0(i32 %x) {
  %mul = mul nsw i32 %x, -12
  %cmp = icmp ne i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul_eq0(
; CHECK-NEXT: icmp eq i32 %x, 0
define i1 @icmp_mul_eq0(i32 %x) {
  %mul = mul nsw i32 %x, 12
  %cmp = icmp eq i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul0_eq0(
; CHECK-NEXT: ret i1 true
define i1 @icmp_mul0_eq0(i32 %x) {
  %mul = mul i32 %x, 0
  %cmp = icmp eq i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_mul0_ne0(
; CHECK-NEXT: ret i1 false
define i1 @icmp_mul0_ne0(i32 %x) {
  %mul = mul i32 %x, 0
  %cmp = icmp ne i32 %mul, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_sub1_sge(
; CHECK-NEXT: icmp sgt i32 %x, %y
define i1 @icmp_sub1_sge(i32 %x, i32 %y) {
  %sub = add nsw i32 %x, -1
  %cmp = icmp sge i32 %sub, %y
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_add1_sgt(
; CHECK-NEXT: icmp sge i32 %x, %y
define i1 @icmp_add1_sgt(i32 %x, i32 %y) {
  %add = add nsw i32 %x, 1
  %cmp = icmp sgt i32 %add, %y
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_sub1_slt(
; CHECK-NEXT: icmp sle i32 %x, %y
define i1 @icmp_sub1_slt(i32 %x, i32 %y) {
  %sub = add nsw i32 %x, -1
  %cmp = icmp slt i32 %sub, %y
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_add1_sle(
; CHECK-NEXT: icmp slt i32 %x, %y
define i1 @icmp_add1_sle(i32 %x, i32 %y) {
  %add = add nsw i32 %x, 1
  %cmp = icmp sle i32 %add, %y
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_add20_sge_add57(
; CHECK-NEXT: [[ADD:%[a-z0-9]+]] = add nsw i32 %y, 37
; CHECK-NEXT: icmp sle i32 [[ADD]], %x
define i1 @icmp_add20_sge_add57(i32 %x, i32 %y) {
  %1 = add nsw i32 %x, 20
  %2 = add nsw i32 %y, 57
  %cmp = icmp sge i32 %1, %2
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_sub57_sge_sub20(
; CHECK-NEXT: [[SUB:%[a-z0-9]+]] = add nsw i32 %x, -37
; CHECK-NEXT: icmp sge i32 [[SUB]], %y
define i1 @icmp_sub57_sge_sub20(i32 %x, i32 %y) {
  %1 = add nsw i32 %x, -57
  %2 = add nsw i32 %y, -20
  %cmp = icmp sge i32 %1, %2
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_and_shl_neg_ne_0(
; CHECK-NEXT: [[SHL:%[a-z0-9]+]] = shl i32 1, %B
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 [[SHL]], %A
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp eq i32 [[AND]], 0
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_and_shl_neg_ne_0(i32 %A, i32 %B) {
  %neg = xor i32 %A, -1
  %shl = shl i32 1, %B
  %and = and i32 %shl, %neg
  %cmp = icmp ne i32 %and, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_and_shl_neg_eq_0(
; CHECK-NEXT: [[SHL:%[a-z0-9]+]] = shl i32 1, %B
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 [[SHL]], %A
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 [[AND]], 0
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_and_shl_neg_eq_0(i32 %A, i32 %B) {
  %neg = xor i32 %A, -1
  %shl = shl i32 1, %B
  %and = and i32 %shl, %neg
  %cmp = icmp eq i32 %and, 0
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_add_and_shr_ne_0(
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %X, 240
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 [[AND]], 224
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_add_and_shr_ne_0(i32 %X) {
  %shr = lshr i32 %X, 4
  %and = and i32 %shr, 15
  %add = add i32 %and, -14
  %tobool = icmp ne i32 %add, 0
  ret i1 %tobool
}

; PR16244
; CHECK-LABEL: define i1 @test71(
; CHECK-NEXT: ret i1 false
define i1 @test71(i8* %x) {
  %a = getelementptr i8, i8* %x, i64 8
  %b = getelementptr inbounds i8, i8* %x, i64 8
  %c = icmp ugt i8* %a, %b
  ret i1 %c
}

define i1 @test71_as1(i8 addrspace(1)* %x) {
; CHECK-LABEL: @test71_as1(
; CHECK-NEXT: ret i1 false
  %a = getelementptr i8, i8 addrspace(1)* %x, i64 8
  %b = getelementptr inbounds i8, i8 addrspace(1)* %x, i64 8
  %c = icmp ugt i8 addrspace(1)* %a, %b
  ret i1 %c
}

; CHECK-LABEL: @icmp_shl_1_V_ult_32(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ult i32 %V, 5
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_ult_32(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ult i32 %shl, 32
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_eq_32(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp eq i32 %V, 5
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_eq_32(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp eq i32 %shl, 32
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_ult_30(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ult i32 %V, 5
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_ult_30(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ult i32 %shl, 30
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_ugt_30(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ugt i32 %V, 4
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_ugt_30(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ugt i32 %shl, 30
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_ule_30(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ult i32 %V, 5
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_ule_30(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ule i32 %shl, 30
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_uge_30(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ugt i32 %V, 4
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_uge_30(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp uge i32 %shl, 30
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_uge_2147483648(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp eq i32 %V, 31
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_uge_2147483648(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp uge i32 %shl, 2147483648
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_shl_1_V_ult_2147483648(
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 %V, 31
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_shl_1_V_ult_2147483648(i32 %V) {
  %shl = shl i32 1, %V
  %cmp = icmp ult i32 %shl, 2147483648
  ret i1 %cmp
}

; CHECK-LABEL: @or_icmp_eq_B_0_icmp_ult_A_B(
; CHECK-NEXT: [[SUB:%[a-z0-9]+]] = add i64 %b, -1
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp uge i64 [[SUB]], %a
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @or_icmp_eq_B_0_icmp_ult_A_B(i64 %a, i64 %b) {
  %1 = icmp eq i64 %b, 0
  %2 = icmp ult i64 %a, %b
  %3 = or i1 %1, %2
  ret i1 %3
}

; CHECK-LABEL: @icmp_add_ult_2(
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %X, -2
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp eq i32 [[AND]], 14
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_add_ult_2(i32 %X) {
  %add = add i32 %X, -14
  %cmp = icmp ult i32 %add, 2
  ret i1 %cmp
}

; CHECK: @icmp_add_X_-14_ult_2
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %X, -2
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp eq i32 [[AND]], 14
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_add_X_-14_ult_2(i32 %X) {
  %add = add i32 %X, -14
  %cmp = icmp ult i32 %add, 2
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_sub_3_X_ult_2(
; CHECK-NEXT: [[OR:%[a-z0-9]+]] = or i32 %X, 1
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp eq i32 [[OR]], 3
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_sub_3_X_ult_2(i32 %X) {
  %add = sub i32 3, %X
  %cmp = icmp ult i32 %add, 2
  ret i1 %cmp
}

; CHECK: @icmp_add_X_-14_uge_2
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %X, -2
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 [[AND]], 14
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_add_X_-14_uge_2(i32 %X) {
  %add = add i32 %X, -14
  %cmp = icmp uge i32 %add, 2
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_sub_3_X_uge_2(
; CHECK-NEXT: [[OR:%[a-z0-9]+]] = or i32 %X, 1
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 [[OR]], 3
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_sub_3_X_uge_2(i32 %X) {
  %add = sub i32 3, %X
  %cmp = icmp uge i32 %add, 2
  ret i1 %cmp
}

; CHECK: @icmp_and_X_-16_eq-16
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ugt i32 %X, -17
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_and_X_-16_eq-16(i32 %X) {
  %and = and i32 %X, -16
  %cmp = icmp eq i32 %and, -16
  ret i1 %cmp
}

; CHECK: @icmp_and_X_-16_ne-16
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ult i32 %X, -16
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_and_X_-16_ne-16(i32 %X) {
  %and = and i32 %X, -16
  %cmp = icmp ne i32 %and, -16
  ret i1 %cmp
}

; CHECK: @icmp_sub_-1_X_ult_4
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ugt i32 %X, -5
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_sub_-1_X_ult_4(i32 %X) {
  %sub = sub i32 -1, %X
  %cmp = icmp ult i32 %sub, 4
  ret i1 %cmp
}

; CHECK: @icmp_sub_-1_X_uge_4
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ult i32 %X, -4
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_sub_-1_X_uge_4(i32 %X) {
  %sub = sub i32 -1, %X
  %cmp = icmp uge i32 %sub, 4
  ret i1 %cmp
}

; CHECK-LABEL: @icmp_swap_operands_for_cse
; CHECK: [[CMP:%[a-z0-9]+]] = icmp ult i32 %X, %Y
; CHECK-NEXT: br i1 [[CMP]], label %true, label %false
; CHECK: ret i1
define i1 @icmp_swap_operands_for_cse(i32 %X, i32 %Y) {
entry:
  %sub = sub i32 %X, %Y
  %cmp = icmp ugt i32 %Y, %X
  br i1 %cmp, label %true, label %false
true:
  %restrue = trunc i32 %sub to i1
  br label %end
false:
  %shift = lshr i32 %sub, 4
  %resfalse = trunc i32 %shift to i1
  br label %end
end:
  %res = phi i1 [%restrue, %true], [%resfalse, %false]
  ret i1 %res
}

; CHECK-LABEL: @icmp_swap_operands_for_cse2
; CHECK: [[CMP:%[a-z0-9]+]] = icmp ult i32 %X, %Y
; CHECK-NEXT: br i1 [[CMP]], label %true, label %false
; CHECK: ret i1
define i1 @icmp_swap_operands_for_cse2(i32 %X, i32 %Y) {
entry:
  %cmp = icmp ugt i32 %Y, %X
  br i1 %cmp, label %true, label %false
true:
  %sub = sub i32 %X, %Y
  %sub1 = sub i32 %X, %Y
  %add = add i32 %sub, %sub1
  %restrue = trunc i32 %add to i1
  br label %end
false:
  %sub2 = sub i32 %Y, %X
  %resfalse = trunc i32 %sub2 to i1
  br label %end
end:
  %res = phi i1 [%restrue, %true], [%resfalse, %false]
  ret i1 %res
}

; CHECK-LABEL: @icmp_do_not_swap_operands_for_cse
; CHECK: [[CMP:%[a-z0-9]+]] = icmp ugt i32 %Y, %X
; CHECK-NEXT: br i1 [[CMP]], label %true, label %false
; CHECK: ret i1
define i1 @icmp_do_not_swap_operands_for_cse(i32 %X, i32 %Y) {
entry:
  %cmp = icmp ugt i32 %Y, %X
  br i1 %cmp, label %true, label %false
true:
  %sub = sub i32 %X, %Y
  %restrue = trunc i32 %sub to i1
  br label %end
false:
  %sub2 = sub i32 %Y, %X
  %resfalse = trunc i32 %sub2 to i1
  br label %end
end:
  %res = phi i1 [%restrue, %true], [%resfalse, %false]
  ret i1 %res
}

; CHECK-LABEL: @icmp_lshr_lshr_eq
; CHECK: %z.unshifted = xor i32 %a, %b
; CHECK: %z = icmp ult i32 %z.unshifted, 1073741824
define i1 @icmp_lshr_lshr_eq(i32 %a, i32 %b) nounwind {
 %x = lshr i32 %a, 30
 %y = lshr i32 %b, 30
 %z = icmp eq i32 %x, %y
 ret i1 %z
}

; CHECK-LABEL: @icmp_ashr_ashr_ne
; CHECK: %z.unshifted = xor i32 %a, %b
; CHECK: %z = icmp ugt i32 %z.unshifted, 255
define i1 @icmp_ashr_ashr_ne(i32 %a, i32 %b) nounwind {
 %x = ashr i32 %a, 8
 %y = ashr i32 %b, 8
 %z = icmp ne i32 %x, %y
 ret i1 %z
}

; CHECK-LABEL: @icmp_neg_cst_slt
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp sgt i32 %a, 10
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_neg_cst_slt(i32 %a) {
  %1 = sub nsw i32 0, %a
  %2 = icmp slt i32 %1, -10
  ret i1 %2
}

; CHECK-LABEL: @icmp_and_or_lshr
; CHECK-NEXT: [[SHL:%[a-z0-9]+]] = shl nuw i32 1, %y
; CHECK-NEXT: [[OR:%[a-z0-9]+]] = or i32 [[SHL]], 1
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 [[OR]], %x
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 [[AND]], 0
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_and_or_lshr(i32 %x, i32 %y) {
  %shf = lshr i32 %x, %y
  %or = or i32 %shf, %x
  %and = and i32 %or, 1
  %ret = icmp ne i32 %and, 0
  ret i1 %ret
}

; CHECK-LABEL: @icmp_and_or_lshr_cst
; CHECK-NEXT: [[AND:%[a-z0-9]+]] = and i32 %x, 3
; CHECK-NEXT: [[CMP:%[a-z0-9]+]] = icmp ne i32 [[AND]], 0
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @icmp_and_or_lshr_cst(i32 %x) {
  %shf = lshr i32 %x, 1
  %or = or i32 %shf, %x
  %and = and i32 %or, 1
  %ret = icmp ne i32 %and, 0
  ret i1 %ret
}

; CHECK-LABEL: @shl_ap1_zero_ap2_non_zero_2
; CHECK-NEXT: %cmp = icmp ugt i32 %a, 29
; CHECK-NEXT: ret i1 %cmp
define i1 @shl_ap1_zero_ap2_non_zero_2(i32 %a) {
 %shl = shl i32 4, %a
 %cmp = icmp eq i32 %shl, 0
 ret i1 %cmp
}

; CHECK-LABEL: @shl_ap1_zero_ap2_non_zero_4
; CHECK-NEXT: %cmp = icmp ugt i32 %a, 30
; CHECK-NEXT: ret i1 %cmp
define i1 @shl_ap1_zero_ap2_non_zero_4(i32 %a) {
 %shl = shl i32 -2, %a
 %cmp = icmp eq i32 %shl, 0
 ret i1 %cmp
}

; CHECK-LABEL: @shl_ap1_non_zero_ap2_non_zero_both_positive
; CHECK-NEXT: %cmp = icmp eq i32 %a, 0
; CHECK-NEXT: ret i1 %cmp
define i1 @shl_ap1_non_zero_ap2_non_zero_both_positive(i32 %a) {
 %shl = shl i32 50, %a
 %cmp = icmp eq i32 %shl, 50
 ret i1 %cmp
}

; CHECK-LABEL: @shl_ap1_non_zero_ap2_non_zero_both_negative
; CHECK-NEXT: %cmp = icmp eq i32 %a, 0
; CHECK-NEXT: ret i1 %cmp
define i1 @shl_ap1_non_zero_ap2_non_zero_both_negative(i32 %a) {
 %shl = shl i32 -50, %a
 %cmp = icmp eq i32 %shl, -50
 ret i1 %cmp
}

; CHECK-LABEL: @shl_ap1_non_zero_ap2_non_zero_ap1_1
; CHECK-NEXT: ret i1 false
define i1 @shl_ap1_non_zero_ap2_non_zero_ap1_1(i32 %a) {
 %shl = shl i32 50, %a
 %cmp = icmp eq i32 %shl, 25
 ret i1 %cmp
}

; CHECK-LABEL: @shl_ap1_non_zero_ap2_non_zero_ap1_2
; CHECK-NEXT: %cmp = icmp eq i32 %a, 1
; CHECK-NEXT: ret i1 %cmp
define i1 @shl_ap1_non_zero_ap2_non_zero_ap1_2(i32 %a) {
 %shl = shl i32 25, %a
 %cmp = icmp eq i32 %shl, 50
 ret i1 %cmp
}

; CHECK-LABEL: @shl_ap1_non_zero_ap2_non_zero_ap1_3
; CHECK-NEXT: ret i1 false
define i1 @shl_ap1_non_zero_ap2_non_zero_ap1_3(i32 %a) {
 %shl = shl i32 26, %a
 %cmp = icmp eq i32 %shl, 50
 ret i1 %cmp
}

; CHECK-LABEL: @icmp_sgt_zero_add_nsw
; CHECK-NEXT: icmp sgt i32 %a, -1
define i1 @icmp_sgt_zero_add_nsw(i32 %a) {
 %add = add nsw i32 %a, 1
 %cmp = icmp sgt i32 %add, 0
 ret i1 %cmp
}

; CHECK-LABEL: @icmp_sge_zero_add_nsw
; CHECK-NEXT: icmp sgt i32 %a, -2
define i1 @icmp_sge_zero_add_nsw(i32 %a) {
 %add = add nsw i32 %a, 1
 %cmp = icmp sge i32 %add, 0
 ret i1 %cmp
}

; CHECK-LABEL: @icmp_slt_zero_add_nsw
; CHECK-NEXT: icmp slt i32 %a, -1
define i1 @icmp_slt_zero_add_nsw(i32 %a) {
 %add = add nsw i32 %a, 1
 %cmp = icmp slt i32 %add, 0
 ret i1 %cmp
}

; CHECK-LABEL: @icmp_sle_zero_add_nsw
; CHECK-NEXT: icmp slt i32 %a, 0
define i1 @icmp_sle_zero_add_nsw(i32 %a) {
 %add = add nsw i32 %a, 1
 %cmp = icmp sle i32 %add, 0
 ret i1 %cmp
}

; CHECK-LABEL: @icmp_cmpxchg_strong
; CHECK-NEXT: %[[xchg:.*]] = cmpxchg i32* %sc, i32 %old_val, i32 %new_val seq_cst seq_cst
; CHECK-NEXT: %[[icmp:.*]] = extractvalue { i32, i1 } %[[xchg]], 1
; CHECK-NEXT: ret i1 %[[icmp]]
define zeroext i1 @icmp_cmpxchg_strong(i32* %sc, i32 %old_val, i32 %new_val) {
  %xchg = cmpxchg i32* %sc, i32 %old_val, i32 %new_val seq_cst seq_cst
  %xtrc = extractvalue { i32, i1 } %xchg, 0
  %icmp = icmp eq i32 %xtrc, %old_val
  ret i1 %icmp
}

; CHECK-LABEL: @f1
; CHECK-NEXT: %[[cmp:.*]] = icmp sge i64 %a, %b
; CHECK-NEXT: ret i1 %[[cmp]]
define i1 @f1(i64 %a, i64 %b) {
  %t = sub nsw i64 %a, %b
  %v = icmp sge i64 %t, 0
  ret i1 %v
}

; CHECK-LABEL: @f2
; CHECK-NEXT: %[[cmp:.*]] = icmp sgt i64 %a, %b
; CHECK-NEXT: ret i1 %[[cmp]]
define i1 @f2(i64 %a, i64 %b) {
  %t = sub nsw i64 %a, %b
  %v = icmp sgt i64 %t, 0
  ret i1 %v
}

; CHECK-LABEL: @f3
; CHECK-NEXT: %[[cmp:.*]] = icmp slt i64 %a, %b
; CHECK-NEXT: ret i1 %[[cmp]]
define i1 @f3(i64 %a, i64 %b) {
  %t = sub nsw i64 %a, %b
  %v = icmp slt i64 %t, 0
  ret i1 %v
}

; CHECK-LABEL: @f4
; CHECK-NEXT: %[[cmp:.*]] = icmp sle i64 %a, %b
; CHECK-NEXT: ret i1 %[[cmp]]
define i1 @f4(i64 %a, i64 %b) {
  %t = sub nsw i64 %a, %b
  %v = icmp sle i64 %t, 0
  ret i1 %v
}

; CHECK-LABEL: @f5
; CHECK: %[[cmp:.*]] = icmp slt i32 %[[sub:.*]], 0
; CHECK: %[[neg:.*]] = sub nsw i32 0, %[[sub]]
; CHECK: %[[sel:.*]] = select i1 %[[cmp]], i32 %[[neg]], i32 %[[sub]]
; CHECK: ret i32 %[[sel]]
define i32 @f5(i8 %a, i8 %b) {
  %conv = zext i8 %a to i32
  %conv3 = zext i8 %b to i32
  %sub = sub nsw i32 %conv, %conv3
  %cmp4 = icmp slt i32 %sub, 0
  %sub7 = sub nsw i32 0, %sub
  %sub7.sub = select i1 %cmp4, i32 %sub7, i32 %sub
  ret i32 %sub7.sub
}

; CHECK-LABEL: @f6
; CHECK: %cmp.unshifted = xor i32 %a, %b
; CHECK-NEXT: %cmp.mask = and i32 %cmp.unshifted, 255
; CHECK-NEXT: %cmp = icmp eq i32 %cmp.mask, 0
; CHECK-NEXT: %s = select i1 %cmp, i32 10000, i32 0
; CHECK-NEXT: ret i32 %s
define i32 @f6(i32 %a, i32 %b) {
  %sext = shl i32 %a, 24
  %conv = ashr i32 %sext, 24
  %sext6 = shl i32 %b, 24
  %conv4 = ashr i32 %sext6, 24
  %cmp = icmp eq i32 %conv, %conv4
  %s = select i1 %cmp, i32 10000, i32 0
  ret i32 %s
}

; CHECK-LABEL: @f7
; CHECK:   %cmp.unshifted = xor i32 %a, %b
; CHECK-NEXT: %cmp.mask = and i32 %cmp.unshifted, 511
; CHECK-NEXT: %cmp = icmp ne i32 %cmp.mask, 0
; CHECK-NEXT: %s = select i1 %cmp, i32 10000, i32 0
; CHECK-NEXT: ret i32 %s
define i32 @f7(i32 %a, i32 %b) {
  %sext = shl i32 %a, 23
  %sext6 = shl i32 %b, 23
  %cmp = icmp ne i32 %sext, %sext6
  %s = select i1 %cmp, i32 10000, i32 0
  ret i32 %s
}

; CHECK: @f8(
; CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = icmp ne i32 %lim, 0
; CHECK-NEXT: ret i1 [[RESULT]]
define i1 @f8(i32 %val, i32 %lim) {
  %lim.sub = add i32 %lim, -1
  %val.and = and i32 %val, %lim.sub
  %r = icmp ult i32 %val.and, %lim
  ret i1 %r
}

; CHECK: @f9(
; CHECK-NEXT: [[RESULT:%[a-z0-9]+]] = icmp ne i32 %lim, 0
; CHECK-NEXT: ret i1 [[RESULT]]
define i1 @f9(i32 %val, i32 %lim) {
  %lim.sub = sub i32 %lim, 1
  %val.and = and i32 %val, %lim.sub
  %r = icmp ult i32 %val.and, %lim
  ret i1 %r
}

; CHECK: @f10(
; CHECK: [[CMP:%.*]] = icmp uge i16 %p, mul (i16 zext (i8 ptrtoint (i1 (i16)* @f10 to i8) to i16), i16 zext (i8 ptrtoint (i1 (i16)* @f10 to i8) to i16))
; CHECK-NEXT: ret i1 [[CMP]]
define i1 @f10(i16 %p) {
entry:
  %cmp580 = icmp ule i16 mul (i16 zext (i8 ptrtoint (i1 (i16)* @f10 to i8) to i16), i16 zext (i8 ptrtoint (i1 (i16)* @f10 to i8) to i16)), %p
  ret i1 %cmp580
}

; CHECK-LABEL: @cmp_sgt_rhs_dec
; CHECK-NOT: sub
; CHECK: icmp sge i32 %conv, %i
define i1 @cmp_sgt_rhs_dec(float %x, i32 %i) {
  %conv = fptosi float %x to i32
  %dec = sub nsw i32 %i, 1
  %cmp = icmp sgt i32 %conv, %dec
  ret i1 %cmp
}

; CHECK-LABEL: @cmp_sle_rhs_dec
; CHECK-NOT: sub
; CHECK: icmp slt i32 %conv, %i
define i1 @cmp_sle_rhs_dec(float %x, i32 %i) {
  %conv = fptosi float %x to i32
  %dec = sub nsw i32 %i, 1
  %cmp = icmp sle i32 %conv, %dec
  ret i1 %cmp
}

; CHECK-LABEL: @cmp_sge_rhs_inc
; CHECK-NOT: add
; CHECK: icmp sgt i32 %conv, %i
define i1 @cmp_sge_rhs_inc(float %x, i32 %i) {
  %conv = fptosi float %x to i32
  %inc = add nsw i32 %i, 1
  %cmp = icmp sge i32 %conv, %inc
  ret i1 %cmp
}

; CHECK-LABEL: @cmp_slt_rhs_inc
; CHECK-NOT: add
; CHECK: icmp sle i32 %conv, %i
define i1 @cmp_slt_rhs_inc(float %x, i32 %i) {
  %conv = fptosi float %x to i32
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %conv, %inc
  ret i1 %cmp
}

; CHECK-LABEL: @PR26407
; CHECK-NEXT: %[[addx:.*]] = add i32 %x, 2147483647
; CHECK-NEXT: %[[addy:.*]] = add i32 %y, 2147483647
; CHECK-NEXT: %[[cmp:.*]] = icmp uge i32 %[[addx]], %[[addy]]
; CHECK-NEXT: ret i1 %[[cmp]]
define i1 @PR26407(i32 %x, i32 %y) {
  %addx = add i32 %x, 2147483647
  %addy = add i32 %y, 2147483647
  %cmp = icmp uge i32 %addx, %addy
  ret i1 %cmp
}
