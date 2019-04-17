; Ignore stderr, we expect warnings there
; RUN: opt < %s -instcombine 2> /dev/null -S | FileCheck %s

target datalayout = "E-p:64:64:64-p1:16:16:16-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

; Simple case, argument translatable without changing the value
declare void @test1a(i8*)

define void @test1(i32* %A) {
; CHECK-LABEL: @test1(
; CHECK: %1 = bitcast i32* %A to i8*
; CHECK: call void @test1a(i8* %1)
; CHECK: ret void
  call void bitcast (void (i8*)* @test1a to void (i32*)*)( i32* %A )
  ret void
}


; Should not do because of change in address space of the parameter
define void @test1_as1_illegal(i32 addrspace(1)* %A) {
; CHECK-LABEL: @test1_as1_illegal(
; CHECK: call void bitcast
  call void bitcast (void (i8*)* @test1a to void (i32 addrspace(1)*)*)(i32 addrspace(1)* %A)
  ret void
}

; Test1, but the argument has a different sized address-space
declare void @test1a_as1(i8 addrspace(1)*)

; This one is OK to perform
define void @test1_as1(i32 addrspace(1)* %A) {
; CHECK-LABEL: @test1_as1(
; CHECK: %1 = bitcast i32 addrspace(1)* %A to i8 addrspace(1)*
; CHECK: call void @test1a_as1(i8 addrspace(1)* %1)
; CHECK: ret void
  call void bitcast (void (i8 addrspace(1)*)* @test1a_as1 to void (i32 addrspace(1)*)*)(i32 addrspace(1)* %A )
  ret void
}

; More complex case, translate argument because of resolution.  This is safe
; because we have the body of the function
define void @test2a(i8 %A) {
; CHECK-LABEL: @test2a(
; CHECK: ret void
  ret void
}

define i32 @test2(i32 %A) {
; CHECK-LABEL: @test2(
; CHECK: call void bitcast
; CHECK: ret i32 %A
  call void bitcast (void (i8)* @test2a to void (i32)*)( i32 %A )
  ret i32 %A
}


; Resolving this should insert a cast from sbyte to int, following the C
; promotion rules.
define void @test3a(i8, ...) {unreachable }

define void @test3(i8 %A, i8 %B) {
; CHECK-LABEL: @test3(
; CHECK: %1 = zext i8 %B to i32
; CHECK: call void (i8, ...) @test3a(i8 %A, i32 %1)
; CHECK: ret void
  call void bitcast (void (i8, ...)* @test3a to void (i8, i8)*)( i8 %A, i8 %B)
  ret void
}

; test conversion of return value...
define i8 @test4a() {
; CHECK-LABEL: @test4a(
; CHECK: ret i8 0
  ret i8 0
}

define i32 @test4() {
; CHECK-LABEL: @test4(
; CHECK: call i32 bitcast
  %X = call i32 bitcast (i8 ()* @test4a to i32 ()*)( )            ; <i32> [#uses=1]
  ret i32 %X
}

; test conversion of return value... no value conversion occurs so we can do
; this with just a prototype...
declare i32 @test5a()

define i32 @test5() {
; CHECK-LABEL: @test5(
; CHECK: %X = call i32 @test5a()
; CHECK: ret i32 %X
  %X = call i32 @test5a( )                ; <i32> [#uses=1]
  ret i32 %X
}

; test addition of new arguments...
declare i32 @test6a(i32)

define i32 @test6() {
; CHECK-LABEL: @test6(
; CHECK: %X = call i32 @test6a(i32 0)
; CHECK: ret i32 %X
  %X = call i32 bitcast (i32 (i32)* @test6a to i32 ()*)( )
  ret i32 %X
}

; test removal of arguments, only can happen with a function body
define void @test7a() {
; CHECK-LABEL: @test7a(
; CHECK: ret void
  ret void
}

define void @test7() {
; CHECK-LABEL: @test7(
; CHECK: call void @test7a()
; CHECK: ret void
  call void bitcast (void ()* @test7a to void (i32)*)( i32 5 )
  ret void
}


; rdar://7590304
declare void @test8a()

define i8* @test8() personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: @test8(
; CHECK-NEXT: invoke void @test8a()
; Don't turn this into "unreachable": the callee and caller don't agree in
; calling conv, but the implementation of test8a may actually end up using the
; right calling conv.
  invoke void @test8a()
          to label %invoke.cont unwind label %try.handler

invoke.cont:                                      ; preds = %entry
  unreachable

try.handler:                                      ; preds = %entry
  %exn = landingpad {i8*, i32}
            cleanup
  ret i8* null
}

declare i32 @__gxx_personality_v0(...)


; Don't turn this into a direct call, because test9x is just a prototype and
; doing so will make it varargs.
; rdar://9038601
declare i8* @test9x(i8*, i8*, ...) noredzone
define i8* @test9(i8* %arg, i8* %tmp3) nounwind ssp noredzone {
; CHECK-LABEL: @test9
entry:
  %call = call i8* bitcast (i8* (i8*, i8*, ...)* @test9x to i8* (i8*, i8*)*)(i8* %arg, i8* %tmp3) noredzone
  ret i8* %call
; CHECK-LABEL: @test9(
; CHECK: call i8* bitcast
}


; Parameter that's a vector of pointers
declare void @test10a(<2 x i8*>)

define void @test10(<2 x i32*> %A) {
; CHECK-LABEL: @test10(
; CHECK: %1 = bitcast <2 x i32*> %A to <2 x i8*>
; CHECK: call void @test10a(<2 x i8*> %1)
; CHECK: ret void
  call void bitcast (void (<2 x i8*>)* @test10a to void (<2 x i32*>)*)(<2 x i32*> %A)
  ret void
}

; Don't transform because different address spaces
declare void @test10a_mixed_as(<2 x i8 addrspace(1)*>)

define void @test10_mixed_as(<2 x i8*> %A) {
; CHECK-LABEL: @test10_mixed_as(
; CHECK: call void bitcast
  call void bitcast (void (<2 x i8 addrspace(1)*>)* @test10a_mixed_as to void (<2 x i8*>)*)(<2 x i8*> %A)
  ret void
}

; Return type that's a pointer
define i8* @test11a() {
  ret i8* zeroinitializer
}

define i32* @test11() {
; CHECK-LABEL: @test11(
; CHECK: %X = call i8* @test11a()
; CHECK: %1 = bitcast i8* %X to i32*
  %X = call i32* bitcast (i8* ()* @test11a to i32* ()*)()
  ret i32* %X
}

; Return type that's a pointer with a different address space
define i8 addrspace(1)* @test11a_mixed_as() {
  ret i8 addrspace(1)* zeroinitializer
}

define i8* @test11_mixed_as() {
; CHECK-LABEL: @test11_mixed_as(
; CHECK: call i8* bitcast
  %X = call i8* bitcast (i8 addrspace(1)* ()* @test11a_mixed_as to i8* ()*)()
  ret i8* %X
}

; Return type that's a vector of pointers
define <2 x i8*> @test12a() {
  ret <2 x i8*> zeroinitializer
}

define <2 x i32*> @test12() {
; CHECK-LABEL: @test12(
; CHECK: %X = call <2 x i8*> @test12a()
; CHECK: %1 = bitcast <2 x i8*> %X to <2 x i32*>
  %X = call <2 x i32*> bitcast (<2 x i8*> ()* @test12a to <2 x i32*> ()*)()
  ret <2 x i32*> %X
}

define <2 x i8 addrspace(1)*> @test12a_mixed_as() {
  ret <2 x i8 addrspace(1)*> zeroinitializer
}

define <2 x i8*> @test12_mixed_as() {
; CHECK-LABEL: @test12_mixed_as(
; CHECK: call <2 x i8*> bitcast
  %X = call <2 x i8*> bitcast (<2 x i8 addrspace(1)*> ()* @test12a_mixed_as to <2 x i8*> ()*)()
  ret <2 x i8*> %X
}


; Mix parameter that's a vector of integers and pointers of the same size
declare void @test13a(<2 x i64>)

define void @test13(<2 x i32*> %A) {
; CHECK-LABEL: @test13(
; CHECK: call void bitcast
  call void bitcast (void (<2 x i64>)* @test13a to void (<2 x i32*>)*)(<2 x i32*> %A)
  ret void
}

; Mix parameter that's a vector of integers and pointers of the same
; size, but the other way around
declare void @test14a(<2 x i8*>)

define void @test14(<2 x i64> %A) {
; CHECK-LABEL: @test14(
; CHECK: call void bitcast
  call void bitcast (void (<2 x i8*>)* @test14a to void (<2 x i64>)*)(<2 x i64> %A)
  ret void
}


; Return type that's a vector
define <2 x i16> @test15a() {
  ret <2 x i16> zeroinitializer
}

define i32 @test15() {
; CHECK-LABEL: @test15(
; CHECK: %X = call <2 x i16> @test15a()
; CHECK: %1 = bitcast <2 x i16> %X to i32
  %X = call i32 bitcast (<2 x i16> ()* @test15a to i32 ()*)( )
  ret i32 %X
}

define i32 @test16a() {
  ret i32 0
}

define <2 x i16> @test16() {
; CHECK-LABEL: @test16(
; CHECK: %X = call i32 @test16a()
; CHECK: %1 = bitcast i32 %X to <2 x i16>
  %X = call <2 x i16> bitcast (i32 ()* @test16a to <2 x i16> ()*)( )
  ret <2 x i16> %X
}

declare i32 @pr28655(i32 returned %V)

define i32 @test17() {
entry:
  %C = call i32 @pr28655(i32 0)
  ret i32 %C
}
; CHECK-LABEL: @test17(
; CHECK: call i32 @pr28655(i32 0)
; CHECK: ret i32 0

define void @non_vararg(i8*, i32) {
  ret void
}

define void @test_cast_to_vararg(i8* %this) {
; CHECK-LABEL: test_cast_to_vararg
; CHECK:  call void @non_vararg(i8* %this, i32 42)
  call void (i8*, ...) bitcast (void (i8*, i32)* @non_vararg to void (i8*, ...)*)(i8* %this, i32 42)
  ret void
}
