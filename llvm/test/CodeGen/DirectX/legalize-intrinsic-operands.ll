; RUN: opt -S -passes=dxil-legalize %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-library"

; Fixed i8 ABI operands must not be promoted with integer SSA values.
define void @input_output() {
; CHECK-LABEL: define void @input_output(
; CHECK-NEXT: [[V:%.*]] = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 2, i32 poison)
; CHECK-NEXT: call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float [[V]])
; CHECK-NEXT: ret void
  %v = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 2, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float %v)
  ret void
}

define i32 @increment(target("dx.RawBuffer", i32, 1, 0) %buffer) {
; CHECK-LABEL: define i32 @increment(
; CHECK-NEXT: [[V:%.*]] = call i32 @llvm.dx.resource.updatecounter.{{.*}}(target("dx.RawBuffer", i32, 1, 0) %buffer, i8 1)
; CHECK-NEXT: ret i32 [[V]]
  %v = call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i32, 1, 0) %buffer, i8 1)
  ret i32 %v
}

define i32 @decrement(target("dx.RawBuffer", i32, 1, 0) %buffer) {
; CHECK-LABEL: define i32 @decrement(
; CHECK-NEXT: [[V:%.*]] = call i32 @llvm.dx.resource.updatecounter.{{.*}}(target("dx.RawBuffer", i32, 1, 0) %buffer, i8 -1)
; CHECK-NEXT: ret i32 [[V]]
  %v = call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i32, 1, 0) %buffer, i8 -1)
  ret i32 %v
}
