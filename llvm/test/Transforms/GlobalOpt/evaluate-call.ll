; Checks if bitcasted call expression can be evaluated
; Given call expresion:
;   ptr @_ZL3fooP3Foo(ptr @gBar)
; We evaluate call to function @_ZL3fooP3Foo casting both parameter and return value
; Given call expression:
;   void @_ZL3bazP3Foo(ptr @gBar) 
; We evaluate call to function _ZL3bazP3Foo casting its parameter and check that evaluated value (nullptr)
; is handled correctly

; RUN: opt -passes=globalopt,instcombine -S %s -o - | FileCheck %s

; CHECK:      @gBar = local_unnamed_addr global %struct.Bar { i32 2 }
; CHECK-NEXT: @_s = local_unnamed_addr global %struct.S { i32 1 }, align 4
; CHECK-NEXT: @llvm.global_ctors = appending global [0 x { i32, ptr, ptr }] zeroinitializer

; CHECK:      define i32 @main()
; CHECK-NEXT:   ret i32 0

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.Bar = type { i32 }
%struct.S = type { i32 }
%struct.Foo = type { i32 }

@gBar = global %struct.Bar zeroinitializer, align 4
@_s = global %struct.S zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_main.cpp, ptr null }]

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @_ZN1SC1Ev_alias(ptr @_s)
  ret void
}

@_ZN1SC1Ev_alias = linkonce_odr unnamed_addr alias void (ptr), ptr @_ZN1SC1Ev

define linkonce_odr void @_ZN1SC1Ev(ptr) unnamed_addr align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  call void @_ZN1SC2Ev(ptr %3)
  ret void
}

define i32 @main()  {
  %1 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  ret i32 0
}

define linkonce_odr void @_ZN1SC2Ev(ptr) unnamed_addr align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = call ptr @_ZL3fooP3Foo(ptr @gBar)
  %5 = load i32, ptr %4, align 4
  store i32 %5, ptr %3, align 4
  call void @_ZL3bazP3Foo(ptr @gBar)
  ret void
}

define internal ptr @_ZL3fooP3Foo(ptr) {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store i32 1, ptr %3, align 4
  %4 = load ptr, ptr %2, align 8
  ret ptr %4
}

define internal void @_ZL3bazP3Foo(ptr) {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  store i32 2, ptr %3, align 4
  ret void
}

; Function Attrs: noinline ssp uwtable
define internal void @_GLOBAL__sub_I_main.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}
