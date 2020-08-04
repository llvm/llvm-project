// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @primitive
func @primitive() {
  // CHECK: !llvm.void
  "some.op"() : () -> !llvm.void
  // CHECK: !llvm.half
  "some.op"() : () -> !llvm.half
  // CHECK: !llvm.bfloat
  "some.op"() : () -> !llvm.bfloat
  // CHECK: !llvm.float
  "some.op"() : () -> !llvm.float
  // CHECK: !llvm.double
  "some.op"() : () -> !llvm.double
  // CHECK: !llvm.fp128
  "some.op"() : () -> !llvm.fp128
  // CHECK: !llvm.x86_fp80
  "some.op"() : () -> !llvm.x86_fp80
  // CHECK: !llvm.ppc_fp128
  "some.op"() : () -> !llvm.ppc_fp128
  // CHECK: !llvm.x86_mmx
  "some.op"() : () -> !llvm.x86_mmx
  // CHECK: !llvm.token
  "some.op"() : () -> !llvm.token
  // CHECK: !llvm.label
  "some.op"() : () -> !llvm.label
  // CHECK: !llvm.metadata
  "some.op"() : () -> !llvm.metadata
  return
}

// CHECK-LABEL: @func
func @func() {
  // CHECK: !llvm.func<void ()>
  "some.op"() : () -> !llvm.func<void ()>
  // CHECK: !llvm.func<void (i32)>
  "some.op"() : () -> !llvm.func<void (i32)>
  // CHECK: !llvm.func<i32 ()>
  "some.op"() : () -> !llvm.func<i32 ()>
  // CHECK: !llvm.func<i32 (half, bfloat, float, double)>
  "some.op"() : () -> !llvm.func<i32 (half, bfloat, float, double)>
  // CHECK: !llvm.func<i32 (i32, i32)>
  "some.op"() : () -> !llvm.func<i32 (i32, i32)>
  // CHECK: !llvm.func<void (...)>
  "some.op"() : () -> !llvm.func<void (...)>
  // CHECK: !llvm.func<void (i32, i32, ...)>
  "some.op"() : () -> !llvm.func<void (i32, i32, ...)>
  return
}

// CHECK-LABEL: @integer
func @integer() {
  // CHECK: !llvm.i1
  "some.op"() : () -> !llvm.i1
  // CHECK: !llvm.i8
  "some.op"() : () -> !llvm.i8
  // CHECK: !llvm.i16
  "some.op"() : () -> !llvm.i16
  // CHECK: !llvm.i32
  "some.op"() : () -> !llvm.i32
  // CHECK: !llvm.i64
  "some.op"() : () -> !llvm.i64
  // CHECK: !llvm.i57
  "some.op"() : () -> !llvm.i57
  // CHECK: !llvm.i129
  "some.op"() : () -> !llvm.i129
  return
}

// CHECK-LABEL: @ptr
func @ptr() {
  // CHECK: !llvm.ptr<i8>
  "some.op"() : () -> !llvm.ptr<i8>
  // CHECK: !llvm.ptr<float>
  "some.op"() : () -> !llvm.ptr<float>
  // CHECK: !llvm.ptr<ptr<i8>>
  "some.op"() : () -> !llvm.ptr<ptr<i8>>
  // CHECK: !llvm.ptr<ptr<ptr<ptr<ptr<i8>>>>>
  "some.op"() : () -> !llvm.ptr<ptr<ptr<ptr<ptr<i8>>>>>
  // CHECK: !llvm.ptr<i8>
  "some.op"() : () -> !llvm.ptr<i8, 0>
  // CHECK: !llvm.ptr<i8, 1>
  "some.op"() : () -> !llvm.ptr<i8, 1>
  // CHECK: !llvm.ptr<i8, 42>
  "some.op"() : () -> !llvm.ptr<i8, 42>
  // CHECK: !llvm.ptr<ptr<i8, 42>, 9>
  "some.op"() : () -> !llvm.ptr<ptr<i8, 42>, 9>
  return
}

// CHECK-LABEL: @vec
func @vec() {
  // CHECK: !llvm.vec<4 x i32>
  "some.op"() : () -> !llvm.vec<4 x i32>
  // CHECK: !llvm.vec<4 x float>
  "some.op"() : () -> !llvm.vec<4 x float>
  // CHECK: !llvm.vec<? x 4 x i32>
  "some.op"() : () -> !llvm.vec<? x 4 x i32>
  // CHECK: !llvm.vec<? x 8 x half>
  "some.op"() : () -> !llvm.vec<? x 8 x half>
  // CHECK: !llvm.vec<4 x ptr<i8>>
  "some.op"() : () -> !llvm.vec<4 x ptr<i8>>
  return
}

// CHECK-LABEL: @array
func @array() {
  // CHECK: !llvm.array<10 x i32>
  "some.op"() : () -> !llvm.array<10 x i32>
  // CHECK: !llvm.array<8 x float>
  "some.op"() : () -> !llvm.array<8 x float>
  // CHECK: !llvm.array<10 x ptr<i32, 4>>
  "some.op"() : () -> !llvm.array<10 x ptr<i32, 4>>
  // CHECK: !llvm.array<10 x array<4 x float>>
  "some.op"() : () -> !llvm.array<10 x array<4 x float>>
  return
}

// CHECK-LABEL: @literal_struct
func @literal_struct() {
  // CHECK: !llvm.struct<()>
  "some.op"() : () -> !llvm.struct<()>
  // CHECK: !llvm.struct<(i32)>
  "some.op"() : () -> !llvm.struct<(i32)>
  // CHECK: !llvm.struct<(float, i32)>
  "some.op"() : () -> !llvm.struct<(float, i32)>
  // CHECK: !llvm.struct<(struct<(i32)>)>
  "some.op"() : () -> !llvm.struct<(struct<(i32)>)>
  // CHECK: !llvm.struct<(i32, struct<(i32)>, float)>
  "some.op"() : () -> !llvm.struct<(i32, struct<(i32)>, float)>

  // CHECK: !llvm.struct<packed ()>
  "some.op"() : () -> !llvm.struct<packed ()>
  // CHECK: !llvm.struct<packed (i32)>
  "some.op"() : () -> !llvm.struct<packed (i32)>
  // CHECK: !llvm.struct<packed (float, i32)>
  "some.op"() : () -> !llvm.struct<packed (float, i32)>
  // CHECK: !llvm.struct<packed (float, i32)>
  "some.op"() : () -> !llvm.struct<packed (float, i32)>
  // CHECK: !llvm.struct<packed (struct<(i32)>)>
  "some.op"() : () -> !llvm.struct<packed (struct<(i32)>)>
  // CHECK: !llvm.struct<packed (i32, struct<(i32, i1)>, float)>
  "some.op"() : () -> !llvm.struct<packed (i32, struct<(i32, i1)>, float)>

  // CHECK: !llvm.struct<(struct<packed (i32)>)>
  "some.op"() : () -> !llvm.struct<(struct<packed (i32)>)>
  // CHECK: !llvm.struct<packed (struct<(i32)>)>
  "some.op"() : () -> !llvm.struct<packed (struct<(i32)>)>
  return
}

// CHECK-LABEL: @identified_struct
func @identified_struct() {
  // CHECK: !llvm.struct<"empty", ()>
  "some.op"() : () -> !llvm.struct<"empty", ()>
  // CHECK: !llvm.struct<"opaque", opaque>
  "some.op"() : () -> !llvm.struct<"opaque", opaque>
  // CHECK: !llvm.struct<"long", (i32, struct<(i32, i1)>, float, ptr<func<void ()>>)>
  "some.op"() : () -> !llvm.struct<"long", (i32, struct<(i32, i1)>, float, ptr<func<void ()>>)>
  // CHECK: !llvm.struct<"self-recursive", (ptr<struct<"self-recursive">>)>
  "some.op"() : () -> !llvm.struct<"self-recursive", (ptr<struct<"self-recursive">>)>
  // CHECK: !llvm.struct<"unpacked", (i32)>
  "some.op"() : () -> !llvm.struct<"unpacked", (i32)>
  // CHECK: !llvm.struct<"packed", packed (i32)>
  "some.op"() : () -> !llvm.struct<"packed", packed (i32)>
  // CHECK: !llvm.struct<"name with spaces and !^$@$#", packed (i32)>
  "some.op"() : () -> !llvm.struct<"name with spaces and !^$@$#", packed (i32)>

  // CHECK: !llvm.struct<"mutually-a", (ptr<struct<"mutually-b", (ptr<struct<"mutually-a">, 3>)>>)>
  "some.op"() : () -> !llvm.struct<"mutually-a", (ptr<struct<"mutually-b", (ptr<struct<"mutually-a">, 3>)>>)>
  // CHECK: !llvm.struct<"mutually-b", (ptr<struct<"mutually-a", (ptr<struct<"mutually-b">>)>, 3>)>
  "some.op"() : () -> !llvm.struct<"mutually-b", (ptr<struct<"mutually-a", (ptr<struct<"mutually-b">>)>, 3>)>
  // CHECK: !llvm.struct<"referring-another", (ptr<struct<"unpacked", (i32)>>)>
  "some.op"() : () -> !llvm.struct<"referring-another", (ptr<struct<"unpacked", (i32)>>)>

  // CHECK: !llvm.struct<"struct-of-arrays", (array<10 x i32>)>
  "some.op"() : () -> !llvm.struct<"struct-of-arrays", (array<10 x i32>)>
  // CHECK: !llvm.array<10 x struct<"array-of-structs", (i32)>>
  "some.op"() : () -> !llvm.array<10 x struct<"array-of-structs", (i32)>>
  // CHECK: !llvm.ptr<struct<"ptr-to-struct", (i8)>>
  "some.op"() : () -> !llvm.ptr<struct<"ptr-to-struct", (i8)>>
  return
}

