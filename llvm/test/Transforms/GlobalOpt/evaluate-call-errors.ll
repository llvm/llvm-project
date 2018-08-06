; Checks for few bitcasted call evaluation errors

; REQUIRES: asserts
; RUN: opt -globalopt -instcombine -S -debug-only=evaluator %s -o %t 2>&1 | FileCheck %s

; CHECK: Failed to fold bitcast call expr
; CHECK: Can not convert function argument

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.S = type { i32 }
%struct.Q = type { i32 }
%struct.Foo = type { i32 }

@_s = global %struct.S zeroinitializer, align 4
@_q = global %struct.Q zeroinitializer, align 4
@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main2.cpp, i8* null }, { i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main3.cpp, i8* null }]

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @_ZN1SC1Ev(%struct.S* @_s)
  ret void
}

define linkonce_odr void @_ZN1SC1Ev(%struct.S*) unnamed_addr align 2 {
  %2 = alloca %struct.S*, align 8
  store %struct.S* %0, %struct.S** %2, align 8
  %3 = load %struct.S*, %struct.S** %2, align 8
  call void @_ZN1SC2Ev(%struct.S* %3)
  ret void
}

define internal void @__cxx_global_var_init.1() #0 section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @_ZN1QC1Ev(%struct.Q* @_q)
  ret void
}

define linkonce_odr void @_ZN1QC1Ev(%struct.Q*) unnamed_addr  align 2 {
  %2 = alloca %struct.Q*, align 8
  store %struct.Q* %0, %struct.Q** %2, align 8
  %3 = load %struct.Q*, %struct.Q** %2, align 8
  call void @_ZN1QC2Ev(%struct.Q* %3)
  ret void
}

define i32 @main() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  ret i32 0
}

define linkonce_odr void @_ZN1SC2Ev(%struct.S*) unnamed_addr align 2 {
  %2 = alloca %struct.S*, align 8
  %3 = alloca %struct.Foo, align 4
  store %struct.S* %0, %struct.S** %2, align 8
  %4 = load %struct.S*, %struct.S** %2, align 8
  %5 = getelementptr inbounds %struct.S, %struct.S* %4, i32 0, i32 0
  %6 = call i32 bitcast (%struct.Foo* ()* @_ZL3foov to i32 ()*)()
  %7 = getelementptr inbounds %struct.Foo, %struct.Foo* %3, i32 0, i32 0
  store i32 %6, i32* %7, align 4
  %8 = getelementptr inbounds %struct.Foo, %struct.Foo* %3, i32 0, i32 0
  %9 = load i32, i32* %8, align 4
  store i32 %9, i32* %5, align 4
  ret void
}

define internal %struct.Foo* @_ZL3foov() {
  ret %struct.Foo* null
}

define linkonce_odr void @_ZN1QC2Ev(%struct.Q*) unnamed_addr align 2 {
  %2 = alloca %struct.Q*, align 8
  store %struct.Q* %0, %struct.Q** %2, align 8
  %3 = load %struct.Q*, %struct.Q** %2, align 8
  %4 = getelementptr inbounds %struct.Q, %struct.Q* %3, i32 0, i32 0
  %5 = call i32 bitcast (i32 (i32)* @_ZL3baz3Foo to i32 (%struct.Foo*)*)(%struct.Foo* null)
  store i32 %5, i32* %4, align 4
  ret void
}

define internal i32 @_ZL3baz3Foo(i32) {
  %2 = alloca %struct.Foo, align 4
  %3 = getelementptr inbounds %struct.Foo, %struct.Foo* %2, i32 0, i32 0
  store i32 %0, i32* %3, align 4
  %4 = getelementptr inbounds %struct.Foo, %struct.Foo* %2, i32 0, i32 0
  %5 = load i32, i32* %4, align 4
  ret i32 %5
}

; Function Attrs: noinline ssp uwtable
define internal void @_GLOBAL__sub_I_main2.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}

define internal void @_GLOBAL__sub_I_main3.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init.1()
  ret void
}
