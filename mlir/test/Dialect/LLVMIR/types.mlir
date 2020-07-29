// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file | mlir-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @primitive
func @primitive() {
  // CHECK: !llvm2.void
  "some.op"() : () -> !llvm2.void
  // CHECK: !llvm2.half
  "some.op"() : () -> !llvm2.half
  // CHECK: !llvm2.bfloat
  "some.op"() : () -> !llvm2.bfloat
  // CHECK: !llvm2.float
  "some.op"() : () -> !llvm2.float
  // CHECK: !llvm2.double
  "some.op"() : () -> !llvm2.double
  // CHECK: !llvm2.fp128
  "some.op"() : () -> !llvm2.fp128
  // CHECK: !llvm2.x86_fp80
  "some.op"() : () -> !llvm2.x86_fp80
  // CHECK: !llvm2.ppc_fp128
  "some.op"() : () -> !llvm2.ppc_fp128
  // CHECK: !llvm2.x86_mmx
  "some.op"() : () -> !llvm2.x86_mmx
  // CHECK: !llvm2.token
  "some.op"() : () -> !llvm2.token
  // CHECK: !llvm2.label
  "some.op"() : () -> !llvm2.label
  // CHECK: !llvm2.metadata
  "some.op"() : () -> !llvm2.metadata
  return
}

// CHECK-LABEL: @func
func @func() {
  // CHECK: !llvm2.func<void ()>
  "some.op"() : () -> !llvm2.func<void ()>
  // CHECK: !llvm2.func<void (i32)>
  "some.op"() : () -> !llvm2.func<void (i32)>
  // CHECK: !llvm2.func<i32 ()>
  "some.op"() : () -> !llvm2.func<i32 ()>
  // CHECK: !llvm2.func<i32 (half, bfloat, float, double)>
  "some.op"() : () -> !llvm2.func<i32 (half, bfloat, float, double)>
  // CHECK: !llvm2.func<i32 (i32, i32)>
  "some.op"() : () -> !llvm2.func<i32 (i32, i32)>
  // CHECK: !llvm2.func<void (...)>
  "some.op"() : () -> !llvm2.func<void (...)>
  // CHECK: !llvm2.func<void (i32, i32, ...)>
  "some.op"() : () -> !llvm2.func<void (i32, i32, ...)>
  return
}

// CHECK-LABEL: @integer
func @integer() {
  // CHECK: !llvm2.i1
  "some.op"() : () -> !llvm2.i1
  // CHECK: !llvm2.i8
  "some.op"() : () -> !llvm2.i8
  // CHECK: !llvm2.i16
  "some.op"() : () -> !llvm2.i16
  // CHECK: !llvm2.i32
  "some.op"() : () -> !llvm2.i32
  // CHECK: !llvm2.i64
  "some.op"() : () -> !llvm2.i64
  // CHECK: !llvm2.i57
  "some.op"() : () -> !llvm2.i57
  // CHECK: !llvm2.i129
  "some.op"() : () -> !llvm2.i129
  return
}

// CHECK-LABEL: @ptr
func @ptr() {
  // CHECK: !llvm2.ptr<i8>
  "some.op"() : () -> !llvm2.ptr<i8>
  // CHECK: !llvm2.ptr<float>
  "some.op"() : () -> !llvm2.ptr<float>
  // CHECK: !llvm2.ptr<ptr<i8>>
  "some.op"() : () -> !llvm2.ptr<ptr<i8>>
  // CHECK: !llvm2.ptr<ptr<ptr<ptr<ptr<i8>>>>>
  "some.op"() : () -> !llvm2.ptr<ptr<ptr<ptr<ptr<i8>>>>>
  // CHECK: !llvm2.ptr<i8>
  "some.op"() : () -> !llvm2.ptr<i8, 0>
  // CHECK: !llvm2.ptr<i8, 1>
  "some.op"() : () -> !llvm2.ptr<i8, 1>
  // CHECK: !llvm2.ptr<i8, 42>
  "some.op"() : () -> !llvm2.ptr<i8, 42>
  // CHECK: !llvm2.ptr<ptr<i8, 42>, 9>
  "some.op"() : () -> !llvm2.ptr<ptr<i8, 42>, 9>
  return
}

// CHECK-LABEL: @vec
func @vec() {
  // CHECK: !llvm2.vec<4 x i32>
  "some.op"() : () -> !llvm2.vec<4 x i32>
  // CHECK: !llvm2.vec<4 x float>
  "some.op"() : () -> !llvm2.vec<4 x float>
  // CHECK: !llvm2.vec<? x 4 x i32>
  "some.op"() : () -> !llvm2.vec<? x 4 x i32>
  // CHECK: !llvm2.vec<? x 8 x half>
  "some.op"() : () -> !llvm2.vec<? x 8 x half>
  // CHECK: !llvm2.vec<4 x ptr<i8>>
  "some.op"() : () -> !llvm2.vec<4 x ptr<i8>>
  return
}

// CHECK-LABEL: @array
func @array() {
  // CHECK: !llvm2.array<10 x i32>
  "some.op"() : () -> !llvm2.array<10 x i32>
  // CHECK: !llvm2.array<8 x float>
  "some.op"() : () -> !llvm2.array<8 x float>
  // CHECK: !llvm2.array<10 x ptr<i32, 4>>
  "some.op"() : () -> !llvm2.array<10 x ptr<i32, 4>>
  // CHECK: !llvm2.array<10 x array<4 x float>>
  "some.op"() : () -> !llvm2.array<10 x array<4 x float>>
  return
}

// CHECK-LABEL: @literal_struct
func @literal_struct() {
  // CHECK: !llvm2.struct<()>
  "some.op"() : () -> !llvm2.struct<()>
  // CHECK: !llvm2.struct<(i32)>
  "some.op"() : () -> !llvm2.struct<(i32)>
  // CHECK: !llvm2.struct<(float, i32)>
  "some.op"() : () -> !llvm2.struct<(float, i32)>
  // CHECK: !llvm2.struct<(struct<(i32)>)>
  "some.op"() : () -> !llvm2.struct<(struct<(i32)>)>
  // CHECK: !llvm2.struct<(i32, struct<(i32)>, float)>
  "some.op"() : () -> !llvm2.struct<(i32, struct<(i32)>, float)>

  // CHECK: !llvm2.struct<packed ()>
  "some.op"() : () -> !llvm2.struct<packed ()>
  // CHECK: !llvm2.struct<packed (i32)>
  "some.op"() : () -> !llvm2.struct<packed (i32)>
  // CHECK: !llvm2.struct<packed (float, i32)>
  "some.op"() : () -> !llvm2.struct<packed (float, i32)>
  // CHECK: !llvm2.struct<packed (float, i32)>
  "some.op"() : () -> !llvm2.struct<packed (float, i32)>
  // CHECK: !llvm2.struct<packed (struct<(i32)>)>
  "some.op"() : () -> !llvm2.struct<packed (struct<(i32)>)>
  // CHECK: !llvm2.struct<packed (i32, struct<(i32, i1)>, float)>
  "some.op"() : () -> !llvm2.struct<packed (i32, struct<(i32, i1)>, float)>

  // CHECK: !llvm2.struct<(struct<packed (i32)>)>
  "some.op"() : () -> !llvm2.struct<(struct<packed (i32)>)>
  // CHECK: !llvm2.struct<packed (struct<(i32)>)>
  "some.op"() : () -> !llvm2.struct<packed (struct<(i32)>)>
  return
}

// CHECK-LABEL: @identified_struct
func @identified_struct() {
  // CHECK: !llvm2.struct<"empty", ()>
  "some.op"() : () -> !llvm2.struct<"empty", ()>
  // CHECK: !llvm2.struct<"opaque", opaque>
  "some.op"() : () -> !llvm2.struct<"opaque", opaque>
  // CHECK: !llvm2.struct<"long", (i32, struct<(i32, i1)>, float, ptr<func<void ()>>)>
  "some.op"() : () -> !llvm2.struct<"long", (i32, struct<(i32, i1)>, float, ptr<func<void ()>>)>
  // CHECK: !llvm2.struct<"self-recursive", (ptr<struct<"self-recursive">>)>
  "some.op"() : () -> !llvm2.struct<"self-recursive", (ptr<struct<"self-recursive">>)>
  // CHECK: !llvm2.struct<"unpacked", (i32)>
  "some.op"() : () -> !llvm2.struct<"unpacked", (i32)>
  // CHECK: !llvm2.struct<"packed", packed (i32)>
  "some.op"() : () -> !llvm2.struct<"packed", packed (i32)>
  // CHECK: !llvm2.struct<"name with spaces and !^$@$#", packed (i32)>
  "some.op"() : () -> !llvm2.struct<"name with spaces and !^$@$#", packed (i32)>

  // CHECK: !llvm2.struct<"mutually-a", (ptr<struct<"mutually-b", (ptr<struct<"mutually-a">, 3>)>>)>
  "some.op"() : () -> !llvm2.struct<"mutually-a", (ptr<struct<"mutually-b", (ptr<struct<"mutually-a">, 3>)>>)>
  // CHECK: !llvm2.struct<"mutually-b", (ptr<struct<"mutually-a", (ptr<struct<"mutually-b">>)>, 3>)>
  "some.op"() : () -> !llvm2.struct<"mutually-b", (ptr<struct<"mutually-a", (ptr<struct<"mutually-b">>)>, 3>)>
  // CHECK: !llvm2.struct<"referring-another", (ptr<struct<"unpacked", (i32)>>)>
  "some.op"() : () -> !llvm2.struct<"referring-another", (ptr<struct<"unpacked", (i32)>>)>

  // CHECK: !llvm2.struct<"struct-of-arrays", (array<10 x i32>)>
  "some.op"() : () -> !llvm2.struct<"struct-of-arrays", (array<10 x i32>)>
  // CHECK: !llvm2.array<10 x struct<"array-of-structs", (i32)>>
  "some.op"() : () -> !llvm2.array<10 x struct<"array-of-structs", (i32)>>
  // CHECK: !llvm2.ptr<struct<"ptr-to-struct", (i8)>>
  "some.op"() : () -> !llvm2.ptr<struct<"ptr-to-struct", (i8)>>
  return
}

