; RUN: opt -S -passes=dxil-legalize %s | FileCheck %s
; RUN: opt -S -dxil-legalize -dxil-op-lower %s | FileCheck %s --check-prefix=LOWER

target triple = "dxil-pc-shadermodel6.6-library"

; Fixed i8 ABI operands must not be promoted with integer SSA values.
define void @input_output() {
; LOWER-LABEL: define void @input_output(
; LOWER: [[V:%.*]] = call float @dx.op.loadInput.f32(i32 4, i32 0, i32 0, i8 2, i32 poison)
; LOWER-NEXT: call void @dx.op.storeOutput.f32(i32 5, i32 0, i32 0, i8 2, float [[V]])
; LOWER-NEXT: ret void
; CHECK-LABEL: define void @input_output(
; CHECK-NEXT: [[V:%.*]] = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 2, i32 poison)
; CHECK-NEXT: call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float [[V]])
; CHECK-NEXT: ret void
  %v = call float @llvm.dx.load.input.f32(i32 0, i32 0, i8 2, i32 poison)
  call void @llvm.dx.store.output.f32(i32 0, i32 0, i8 2, float %v)
  ret void
}

define i32 @increment() {
; LOWER-LABEL: define i32 @increment(
; LOWER: [[V:%.*]] = call i32 @dx.op.bufferUpdateCounter(i32 70, %dx.types.Handle {{.*}}, i8 1)
; LOWER-NEXT: ret i32 [[V]]
; CHECK-LABEL: define i32 @increment(
; CHECK-NEXT: [[BUFFER:%.*]] = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr null)
; CHECK-NEXT: [[V:%.*]] = call i32 @llvm.dx.resource.updatecounter.{{.*}}(target("dx.RawBuffer", i32, 1, 0) [[BUFFER]], i8 1)
; CHECK-NEXT: ret i32 [[V]]
  %buffer = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %v = call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i32, 1, 0) %buffer, i8 1)
  ret i32 %v
}

define i32 @decrement() {
; LOWER-LABEL: define i32 @decrement(
; LOWER: [[V:%.*]] = call i32 @dx.op.bufferUpdateCounter(i32 70, %dx.types.Handle {{.*}}, i8 -1)
; LOWER-NEXT: ret i32 [[V]]
; CHECK-LABEL: define i32 @decrement(
; CHECK-NEXT: [[BUFFER:%.*]] = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding.{{.*}}(i32 0, i32 0, i32 1, i32 0, ptr null)
; CHECK-NEXT: [[V:%.*]] = call i32 @llvm.dx.resource.updatecounter.{{.*}}(target("dx.RawBuffer", i32, 1, 0) [[BUFFER]], i8 -1)
; CHECK-NEXT: ret i32 [[V]]
  %buffer = call target("dx.RawBuffer", i32, 1, 0) @llvm.dx.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr null)
  %v = call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i32, 1, 0) %buffer, i8 -1)
  ret i32 %v
}

; CHECK-DAG: declare float @llvm.dx.load.input.f32(i32 immarg, i32, i8 immarg, i32)
; CHECK-DAG: declare void @llvm.dx.store.output.f32(i32 immarg, i32, i8 immarg, float)
; CHECK-DAG: declare i32 @llvm.dx.resource.updatecounter.{{.*}}(target("dx.RawBuffer", i32, 1, 0), i8 immarg)
