; RUN: opt -S -dxil-op-lower %s | FileCheck %s


target triple = "dxil-pc-shadermodel6.6-compute"

 ; CHECK-LABEL: define void @update_counter_decrement_vector() {
define void @update_counter_decrement_vector() {
 ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, 
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)

 ; CHECK-NEXT: [[BUFFANOT:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
 ; CHECK-NEXT: [[REG:%.*]] = call i32 @dx.op.bufferUpdateCounter(i32 70, %dx.types.Handle [[BUFFANOT]], i8 -1){{$}}
  %1 = call i32 @llvm.dx.resource.updatecounter(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i8 -1)
  ret void
}

 ; CHECK-LABEL: define void @update_counter_increment_vector() {
define void @update_counter_increment_vector() {
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, 
  %buffer = call target("dx.TypedBuffer", <4 x float>, 0, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_v4f32_0_0_0(
          i32 0, i32 0, i32 1, i32 0, i1 false)
  ; CHECK-NEXT: [[BUFFANOT:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  ; CHECK-NEXT: [[REG:%.*]] = call i32 @dx.op.bufferUpdateCounter(i32 70, %dx.types.Handle [[BUFFANOT]], i8 1){{$}}
  %1 = call i32 @llvm.dx.resource.updatecounter(target("dx.TypedBuffer", <4 x float>, 0, 0, 0) %buffer, i8 1)
  ret void
}

; CHECK-LABEL: define void @update_counter_decrement_scalar() {
define void @update_counter_decrement_scalar() {
    ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217, 
  %buffer = call target("dx.RawBuffer", i8, 0, 0)
      @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_0_0t(
          i32 1, i32 8, i32 1, i32 0, i1 false)
  ; CHECK-NEXT: [[BUFFANOT:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  ; CHECK-NEXT: [[REG:%.*]] = call i32 @dx.op.bufferUpdateCounter(i32 70, %dx.types.Handle [[BUFFANOT]], i8 -1){{$}}
  %1 = call i32 @llvm.dx.resource.updatecounter(target("dx.RawBuffer", i8, 0, 0) %buffer, i8 -1)
  ret void
}
