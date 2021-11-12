// RUN: mlir-opt -test-convert-call-op %s | FileCheck %s

// CHECK-LABEL: @ptr
// CHECK: !llvm.ptr<i42>
func private @ptr() -> !llvm.ptr<!test.smpla>

// CHECK-LABEL: @ptr_ptr()
// CHECK: !llvm.ptr<ptr<i42>> 
func private @ptr_ptr() -> !llvm.ptr<!llvm.ptr<!test.smpla>>

// CHECK-LABEL: @struct_ptr()
// CHECK: !llvm.struct<(ptr<i42>)> 
func private @struct_ptr() -> !llvm.struct<(ptr<!test.smpla>)>

// CHECK-LABEL: @named_struct_ptr()
// CHECK: !llvm.struct<"named", (ptr<!test.smpla>)> 
func private @named_struct_ptr() -> !llvm.struct<"named", (ptr<!test.smpla>)>

// CHECK-LABEL: @array_ptr()
// CHECK: !llvm.array<10 x ptr<i42>> 
func private @array_ptr() -> !llvm.array<10 x ptr<!test.smpla>>

// CHECK-LABEL: @func()
// CHECK: !llvm.ptr<func<i42 (i42)>>
func private @func() -> !llvm.ptr<!llvm.func<!test.smpla (!test.smpla)>>

// TODO: support conversion of recursive types in the conversion infra.
// CHECK-LABEL: @named_recursive()
// CHECK: !llvm.struct<"recursive", (ptr<!test.smpla>, ptr<struct<"recursive">>)> 
func private @named_recursive() -> !llvm.struct<"recursive", (ptr<!test.smpla>, ptr<struct<"recursive">>)>

