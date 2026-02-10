; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define i32 @_Z20byteaddr_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i32, align 4
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i32 0)
  ; CHECK: store i32 [[INTERLOCKED]], ptr [[RETURN]]
  store i32 %hlsl.interlocked.or, ptr %returnVal, align 4
  ; CHECK: [[RETLOAD:]] = load i32, ptr [[RETURN]]
  %0 = load i32, ptr %returnVal, align 4
  ; CHECK; ret i32 [[RETLOAD]]
  ret i32 %0
}

define void @_Z23byteaddr_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i32 0)
  ; CHECK: ret void
  ret void
}

%struct.TestStruct = type { i32, i32 }

define i32 @_Z18struct_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i32, align 4
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", %struct.TestStruct, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.TestStructs_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 4, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_s_struct.TestStructs_1_0t(target("dx.RawBuffer", %struct.TestStruct, 1, 0) %buffer, i32 2, i32 1, i32 4, i32 undef, i32 0)
  ; CHECK: store i32 [[INTERLOCKED]], ptr [[RETURN]]
  store i32 %hlsl.interlocked.or, ptr %returnVal, align 4
  ; CHECK: [[RETLOAD:]] = load i32, ptr [[RETURN]]
  %0 = load i32, ptr %returnVal, align 4
  ; CHECK; ret i32 [[RETLOAD]]
  ret i32 %0
}

define void @_Z21struct_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", %struct.TestStruct, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.TestStructs_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 4, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.RawBuffer_s_struct.TestStructs_1_0t(target("dx.RawBuffer", %struct.TestStruct, 1, 0) %buffer, i32 2, i32 1, i32 4, i32 undef, i32 0)
  ; CHECK: ret void
  ret void
}

define i32 @_Z21typed_int_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i32, align 4
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %buffer, i32 2, i32 1, i32 undef, i32 undef, i32 0)
  ; CHECK: store i32 [[INTERLOCKED]], ptr [[RETURN]]
  store i32 %hlsl.interlocked.or, ptr %returnVal, align 4
  ; CHECK: [[RETLOAD:]] = load i32, ptr [[RETURN]]
  %0 = load i32, ptr %returnVal, align 4
  ; CHECK; ret i32 [[RETLOAD]]
  ret i32 %0
}

define void @_Z24typed_int_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i32, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.TypedBuffer_i32_1_0_1t(target("dx.TypedBuffer", i32, 1, 0, 1) %buffer, i32 2, i32 1, i32 undef, i32 undef, i32 0)
  ; CHECK: ret void
  ret void
}

define i32 @_Z22typed_uint_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i32, align 4
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i32, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.TypedBuffer_i32_1_0_0t(target("dx.TypedBuffer", i32, 1, 0, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i32 0)
  ; CHECK: store i32 [[INTERLOCKED]], ptr [[RETURN]]
  store i32 %hlsl.interlocked.or, ptr %returnVal, align 4
  ; CHECK: [[RETLOAD:]] = load i32, ptr [[RETURN]]
  %0 = load i32, ptr %returnVal, align 4
  ; CHECK; ret i32 [[RETLOAD]]
  ret i32 %0
}

define void @_Z25typed_uint_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i32, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i32_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i32 @dx.op.atomicBinOp.i32(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i32 0)
  %hlsl.interlocked.or = call i32 @llvm.dx.resource.atomicbinop.tdx.TypedBuffer_i32_1_0_0t(target("dx.TypedBuffer", i32, 1, 0, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i32 0)
  ; CHECK: ret void
  ret void
}
