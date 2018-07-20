; Checks if bitcasted call expression can be evaluated
; Given call expresion:
;   %struct.Bar* bitcast (%struct.Foo* (%struct.Foo*)* @_ZL3fooP3Foo to %struct.Bar* (%struct.Bar*)*)(%struct.Bar* @gBar)
; We evaluate call to function @_ZL3fooP3Foo casting both parameter and return value
; Given call expression:
;   void bitcast (void (%struct.Foo*)* @_ZL3bazP3Foo to void (%struct.Bar*)*)(%struct.Bar* @gBar) 
; We evaluate call to function _ZL3bazP3Foo casting its parameter and check that evaluated value (nullptr)
; is handled correctly

; RUN: opt -globalopt -instcombine -S %s -o - | FileCheck %s

; CHECK:      @gBar = local_unnamed_addr global %struct.Bar { i32 2 }
; CHECK-NEXT: @_s = local_unnamed_addr global %struct.S { i32 1 }, align 4
; CHECK-NEXT: @llvm.global_ctors = appending global [0 x { i32, void ()*, i8* }] zeroinitializer

; CHECK:      define i32 @main()
; CHECK-NEXT:   ret i32 0

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

%struct.Bar = type { i32 }
%struct.S = type { i32 }
%struct.Foo = type { i32 }

@gBar = global %struct.Bar zeroinitializer, align 4
@_s = global %struct.S zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main.cpp, i8* null }]

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @_ZN1SC1Ev_alias(%struct.S* @_s)
  ret void
}

@_ZN1SC1Ev_alias = linkonce_odr unnamed_addr alias void (%struct.S*), void (%struct.S*)* @_ZN1SC1Ev

define linkonce_odr void @_ZN1SC1Ev(%struct.S*) unnamed_addr align 2 {
  %2 = alloca %struct.S*, align 8
  store %struct.S* %0, %struct.S** %2, align 8
  %3 = load %struct.S*, %struct.S** %2, align 8
  call void @_ZN1SC2Ev(%struct.S* %3)
  ret void
}

define i32 @main()  {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  ret i32 0
}

define linkonce_odr void @_ZN1SC2Ev(%struct.S*) unnamed_addr align 2 {
  %2 = alloca %struct.S*, align 8
  store %struct.S* %0, %struct.S** %2, align 8
  %3 = load %struct.S*, %struct.S** %2, align 8
  %4 = getelementptr inbounds %struct.S, %struct.S* %3, i32 0, i32 0
  %5 = call %struct.Bar* bitcast (%struct.Foo* (%struct.Foo*)* @_ZL3fooP3Foo to %struct.Bar* (%struct.Bar*)*)(%struct.Bar* @gBar)
  %6 = getelementptr inbounds %struct.Bar, %struct.Bar* %5, i32 0, i32 0
  %7 = load i32, i32* %6, align 4
  store i32 %7, i32* %4, align 4
  call void bitcast (void (%struct.Foo*)* @_ZL3bazP3Foo to void (%struct.Bar*)*)(%struct.Bar* @gBar)
  ret void
}

define internal %struct.Foo* @_ZL3fooP3Foo(%struct.Foo*) {
  %2 = alloca %struct.Foo*, align 8
  store %struct.Foo* %0, %struct.Foo** %2, align 8
  %3 = load %struct.Foo*, %struct.Foo** %2, align 8
  %4 = getelementptr inbounds %struct.Foo, %struct.Foo* %3, i32 0, i32 0
  store i32 1, i32* %4, align 4
  %5 = load %struct.Foo*, %struct.Foo** %2, align 8
  ret %struct.Foo* %5
}

define internal void @_ZL3bazP3Foo(%struct.Foo*) {
  %2 = alloca %struct.Foo*, align 8
  store %struct.Foo* %0, %struct.Foo** %2, align 8
  %3 = load %struct.Foo*, %struct.Foo** %2, align 8
  %4 = getelementptr inbounds %struct.Foo, %struct.Foo* %3, i32 0, i32 0
  store i32 2, i32* %4, align 4
  ret void
}

; Function Attrs: noinline ssp uwtable
define internal void @_GLOBAL__sub_I_main.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
  call void @__cxx_global_var_init()
  ret void
}
