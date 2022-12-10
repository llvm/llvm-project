; Check if we can evaluate a bitcasted call to a function which is constant folded.
; Evaluator folds call to fmodf, replacing it with constant value in case both operands
; are known at compile time.
; RUN: opt -passes=globalopt,instcombine %s -S -o - | FileCheck %s

; CHECK:        @_q = dso_local local_unnamed_addr global %struct.Q { i32 1066527622 }
; CHECK:        define dso_local i32 @main
; CHECK-NEXT:     %[[V:.+]] = load i32, i32* getelementptr inbounds (%struct.Q, %struct.Q* @_q, i64 0, i32 0)
; CHECK-NEXT:     ret i32 %[[V]]

source_filename = "main.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-none-linux-gnu"

%struct.Q = type { i32 }

$_ZN1QC2Ev = comdat any

@_q = dso_local global %struct.Q zeroinitializer, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main.cpp, i8* null }]

define internal void @__cxx_global_var_init() section ".text.startup" {
  call void @_ZN1QC2Ev(%struct.Q* @_q)
  ret void
}

define linkonce_odr dso_local void @_ZN1QC2Ev(%struct.Q*) unnamed_addr #1 comdat align 2 {
  %2 = alloca %struct.Q*, align 8
  store %struct.Q* %0, %struct.Q** %2, align 8
  %3 = load %struct.Q*, %struct.Q** %2, align 8
  %4 = getelementptr inbounds %struct.Q, %struct.Q* %3, i32 0, i32 0
  %5 = call i32 bitcast (float (float, float)* @fmodf to i32 (float, float)*)(float 0x40091EB860000000, float 2.000000e+00)
  store i32 %5, i32* %4, align 4
  ret void
}

define dso_local i32 @main(i32, i8**) {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i8**, align 8
  store i32 0, i32* %3, align 4
  store i32 %0, i32* %4, align 4
  store i8** %1, i8*** %5, align 8
  %6 = load i32, i32* getelementptr inbounds (%struct.Q, %struct.Q* @_q, i32 0, i32 0), align 4
  ret i32 %6
}

; Function Attrs: nounwind
declare dso_local float @fmodf(float, float)

; Function Attrs: noinline uwtable
define internal void @_GLOBAL__sub_I_main.cpp() section ".text.startup" {
  call void @__cxx_global_var_init()
  ret void
}
