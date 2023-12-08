; Checks for few bitcasted call evaluation errors

; REQUIRES: asserts
; RUN: opt -passes=globalopt,instcombine -S -debug-only=evaluator %s -o %t 2>&1 | FileCheck %s

; CHECK: Failed to fold bitcast call expr
; CHECK: Can not convert function argument

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.S = type { i32 }
%struct.Q = type { i32 }
%struct.Foo = type { i32 }

@_s = global %struct.S zeroinitializer, align 4
@_q = global %struct.Q zeroinitializer, align 4
@llvm.global_ctors = appending global [2 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_main2.cpp, ptr null }, { i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_main3.cpp, ptr null }]

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @_ZN1SC1Ev(ptr @_s)
  ret void
}

define linkonce_odr void @_ZN1SC1Ev(ptr) unnamed_addr align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1SC2Ev(ptr %3)
  ret void
}

define internal void @__cxx_global_var_init.1() #0 section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @_ZN1QC1Ev(ptr @_q)
  ret void
}

define linkonce_odr void @_ZN1QC1Ev(ptr) unnamed_addr  align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1QC2Ev(ptr %3)
  ret void
}

define i32 @main() {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 0
}

define linkonce_odr void @_ZN1SC2Ev(ptr) unnamed_addr align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca %struct.Foo, align 4
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = call i32 @_ZL3foov()
  store i32 %5, ptr %3, align 4
  %6 = load i32, ptr %3, align 4
  store i32 %6, ptr %4, align 4
  ret void
}

define internal ptr @_ZL3foov() {
  ret ptr getelementptr (%struct.Foo, ptr null, i32 1)
}

define linkonce_odr void @_ZN1QC2Ev(ptr) unnamed_addr align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call i32 @_ZL3baz3Foo(ptr getelementptr (%struct.Foo, ptr null, i32 1))
  store i32 %4, ptr %3, align 4
  ret void
}

define internal i32 @_ZL3baz3Foo(i32) {
  %2 = alloca %struct.Foo, align 4
  store i32 %0, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  ret i32 %3
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
