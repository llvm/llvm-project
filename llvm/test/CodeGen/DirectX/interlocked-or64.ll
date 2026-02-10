; RUN: opt -S -dxil-op-lower %s | FileCheck %s

target triple = "dxil-pc-shadermodel6.6-compute"

define i64 @_Z20byteaddr_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i64, align 8
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i64 0)
  ; CHECK: store i64 [[INTERLOCKED]], ptr [[RETURN]]
  store i64 %hlsl.interlocked.or, ptr %returnVal, align 8
  ; CHECK: [[RETLOAD:]] = load i64, ptr [[RETURN]]
  %0 = load i64, ptr %returnVal, align 8
  ; CHECK; ret i64 [[RETLOAD]]
  ret i64 %0
}

define void @_Z23byteaddr_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", i8, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_i8_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_i8_1_0t(target("dx.RawBuffer", i8, 1, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i64 0)
  ; CHECK: ret void
  ret void
}

%struct.TestStruct = type { i64, i64 }

define i64 @_Z18struct_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i64, align 8
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", %struct.TestStruct, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.TestStructs_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 8, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_s_struct.TestStructs_1_0t(target("dx.RawBuffer", %struct.TestStruct, 1, 0) %buffer, i32 2, i32 1, i32 8, i32 undef, i64 0)
  ; CHECK: store i64 [[INTERLOCKED]], ptr [[RETURN]]
  store i64 %hlsl.interlocked.or, ptr %returnVal, align 8
  ; CHECK: [[RETLOAD:]] = load i64, ptr [[RETURN]]
  %0 = load i64, ptr %returnVal, align 8
  ; CHECK; ret i64 [[RETLOAD]]
  ret i64 %0
}

define void @_Z21struct_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.RawBuffer", %struct.TestStruct, 1, 0) @llvm.dx.resource.handlefrombinding.tdx.RawBuffer_s_struct.TestStructs_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 8, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.RawBuffer_s_struct.TestStructs_1_0t(target("dx.RawBuffer", %struct.TestStruct, 1, 0) %buffer, i32 2, i32 1, i32 8, i32 undef, i64 0)
  ; CHECK: ret void
  ret void
}

define i64 @_Z21typed_int_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i64, align 8
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i64, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i64_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.TypedBuffer_i64_1_0_1t(target("dx.TypedBuffer", i64, 1, 0, 1) %buffer, i32 2, i32 1, i32 undef, i32 undef, i64 0)
  ; CHECK: store i64 [[INTERLOCKED]], ptr [[RETURN]]
  store i64 %hlsl.interlocked.or, ptr %returnVal, align 8
  ; CHECK: [[RETLOAD:]] = load i64, ptr [[RETURN]]
  %0 = load i64, ptr %returnVal, align 8
  ; CHECK; ret i64 [[RETLOAD]]
  ret i64 %0
}

define void @_Z24typed_int_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i64, 1, 0, 1) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i64_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.TypedBuffer_i64_1_0_1t(target("dx.TypedBuffer", i64, 1, 0, 1) %buffer, i32 2, i32 1, i32 undef, i32 undef, i64 0)
  ; CHECK: ret void
  ret void
}

define i64 @_Z22typed_uint_test_return() {
entry:
  ; CHECK: [[RETURN:%.*]] = alloca
  %returnVal = alloca i64, align 8
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i64, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i64_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.TypedBuffer_i64_1_0_0t(target("dx.TypedBuffer", i64, 1, 0, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i64 0)
  ; CHECK: store i64 [[INTERLOCKED]], ptr [[RETURN]]
  store i64 %hlsl.interlocked.or, ptr %returnVal, align 8
  ; CHECK: [[RETLOAD:]] = load i64, ptr [[RETURN]]
  %0 = load i64, ptr %returnVal, align 8
  ; CHECK; ret i64 [[RETLOAD]]
  ret i64 %0
}

define void @_Z25typed_uint_test_no_return() {
entry:
  ; CHECK: [[BIND:%.*]] = call %dx.types.Handle @dx.op.createHandleFromBinding(i32 217
  ; CHECK: [[HANDLE:%.*]] = call %dx.types.Handle @dx.op.annotateHandle(i32 216, %dx.types.Handle [[BIND]]
  %buffer = call target("dx.TypedBuffer", i64, 1, 0, 0) @llvm.dx.resource.handlefrombinding.tdx.TypedBuffer_i64_1_0t(i32 0, i32 0, i32 1, i32 0, ptr null)
  ; CHECK: [[INTERLOCKED:%.*]] = call i64 @dx.op.atomicBinOp.i64(i32 78, %dx.types.Handle [[HANDLE]], i32 2, i32 1, i32 undef, i32 undef, i64 0)
  %hlsl.interlocked.or = call i64 @llvm.dx.resource.atomicbinop64.tdx.TypedBuffer_i64_1_0_0t(target("dx.TypedBuffer", i64, 1, 0, 0) %buffer, i32 2, i32 1, i32 undef, i32 undef, i64 0)
  ; CHECK: ret void
  ret void
}
