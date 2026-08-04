// RUN: split-file %s %t
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %t/valid.mlir | FileCheck %s
// RUN: mlir-translate -mlir-to-llvmir %t/i386.mlir | FileCheck %s --check-prefix=I386
// RUN: not mlir-translate -mlir-to-llvmir %t/i386-overflow.mlir 2>&1 | FileCheck %s --check-prefix=I386-OVERFLOW
// RUN: not mlir-translate -mlir-to-llvmir %t/device.mlir 2>&1 | FileCheck %s --check-prefix=DEVICE

//--- valid.mlir

omp.private {type = firstprivate} @x.firstprivate : i32 copy {
^bb0(%original: !llvm.ptr, %private: !llvm.ptr):
  %value = llvm.load %original : !llvm.ptr -> i32
  llvm.store %value, %private : i32, !llvm.ptr
  omp.yield(%private : !llvm.ptr)
}

omp.private {type = private} @y.private : i32

llvm.func @allocator_dynamic(%x: !llvm.ptr, %y: !llvm.ptr, %allocator: i64) {
  omp.parallel allocate(%allocator : i64 -> %x : !llvm.ptr)
      private(@x.firstprivate %x -> %x.private,
              @y.private %y -> %y.private : !llvm.ptr, !llvm.ptr) {
    %x.value = llvm.load %x.private : !llvm.ptr -> i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %next = llvm.add %x.value, %one : i32
    llvm.store %next, %x.private : i32, !llvm.ptr
    llvm.store %one, %y.private : i32, !llvm.ptr
    omp.terminator
  } {allocate_private_indices = array<i64: 0>}
  llvm.return
}

// CHECK-LABEL: define void @allocator_dynamic(
// CHECK-COUNT-1: %[[ALLOCATOR:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK: store ptr %[[ALLOCATOR]], ptr
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call
// CHECK-LABEL: define internal void @allocator_dynamic..omp_par
// CHECK: %[[CAPTURED_ALLOCATOR:.*]] = load ptr, ptr %{{.*}}, align 8
// CHECK: %[[ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr %[[CAPTURED_ALLOCATOR]])
// CHECK: %[[Y_ALLOCA:.*]] = alloca i32, align 4
// CHECK: %[[ORIGINAL:.*]] = load i32, ptr %{{.*}}, align 4
// CHECK: store i32 %[[ORIGINAL]], ptr %[[ALLOC]], align 4
// CHECK: %[[PRIVATE:.*]] = load i32, ptr %[[ALLOC]], align 4
// CHECK: store i32 %{{.*}}, ptr %[[ALLOC]], align 4
// CHECK: store i32 1, ptr %[[Y_ALLOCA]], align 4
// CHECK: .fini:
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[ALLOC]], ptr %[[CAPTURED_ALLOCATOR]])

// -----

llvm.func @private_dealloc(!llvm.ptr)

omp.private {type = firstprivate} @x.private : i32 copy {
^bb0(%original: !llvm.ptr, %private: !llvm.ptr):
  %value = llvm.load %original : !llvm.ptr -> i32
  llvm.store %value, %private : i32, !llvm.ptr
  omp.yield(%private : !llvm.ptr)
} dealloc {
^bb0(%private: !llvm.ptr):
  llvm.call @private_dealloc(%private) : (!llvm.ptr) -> ()
  omp.yield
}
omp.private {type = private} @y.private : i32

llvm.func @allocator_reverse_free(%x: !llvm.ptr, %y: !llvm.ptr,
                                  %allocator.x: i64, %allocator.y: i64) {
  omp.parallel allocate(%allocator.y : i64 -> %y : !llvm.ptr,
                        %allocator.x : i64 -> %x : !llvm.ptr)
      private(@x.private %x -> %x.private,
              @y.private %y -> %y.private : !llvm.ptr, !llvm.ptr) {
    omp.terminator
  } {allocate_private_indices = array<i64: 1, 0>}
  llvm.return
}

// CHECK-LABEL: define void @allocator_reverse_free(
// CHECK: %[[ALLOCATOR_Y:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK: %[[ALLOCATOR_X:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK-LABEL: define internal void @allocator_reverse_free..omp_par
// CHECK: %[[CAPTURED_X:.*]] = load ptr, ptr %{{.*}}, align 8
// CHECK: %[[CAPTURED_Y:.*]] = load ptr, ptr %{{.*}}, align 8
// CHECK: %[[X_ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr %[[CAPTURED_X]])
// CHECK: %[[Y_ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr %[[CAPTURED_Y]])
// CHECK: call void @private_dealloc(ptr %[[X_ALLOC]])
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[Y_ALLOC]], ptr %[[CAPTURED_Y]])
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[X_ALLOC]], ptr %[[CAPTURED_X]])
// CHECK-NOT: call void @__kmpc_free

// -----

omp.private {type = private} @x.private : i32

llvm.func @allocator_cancel(%x: !llvm.ptr) {
  %null = llvm.mlir.constant(0 : i64) : i64
  omp.parallel allocate(%null : i64 -> %x : !llvm.ptr)
      private(@x.private %x -> %x.private : !llvm.ptr) {
    omp.cancel cancellation_construct_type(parallel)
    omp.terminator
  } {allocate_private_indices = array<i64: 0>}
  llvm.return
}

// CHECK-LABEL: define internal void @allocator_cancel..omp_par
// CHECK: %[[ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr null)
// CHECK: {{.*}}.cncl:
// CHECK: br label %[[FINI:.*]]
// CHECK: .fini:
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[ALLOC]], ptr null)
// CHECK: omp.par.pre_finalize:
// CHECK: br label %[[FINI]]
// CHECK-NOT: call void @__kmpc_free

// -----

omp.private {type = private} @x.private : i32

llvm.func @allocator_cancellation_point(%x: !llvm.ptr) {
  %null = llvm.mlir.constant(0 : i64) : i64
  omp.parallel allocate(%null : i64 -> %x : !llvm.ptr)
      private(@x.private %x -> %x.private : !llvm.ptr) {
    omp.cancellation_point cancellation_construct_type(parallel)
    omp.terminator
  } {allocate_private_indices = array<i64: 0>}
  llvm.return
}

// CHECK-LABEL: define internal void @allocator_cancellation_point..omp_par
// CHECK: %[[ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr null)
// CHECK: {{.*}}.cncl:
// CHECK: br label %[[FINI:.*]]
// CHECK: .fini:
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[ALLOC]], ptr null)
// CHECK: omp.par.pre_finalize:
// CHECK: br label %[[FINI]]
// CHECK-NOT: call void @__kmpc_free

//--- i386.mlir

module attributes {
  llvm.data_layout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-S128",
  llvm.target_triple = "i386-unknown-linux-gnu"
} {
  omp.private {type = private} @x.private : i32

  llvm.func @allocator_i386(%x: !llvm.ptr) {
    %null = llvm.mlir.constant(0 : i64) : i64
    omp.parallel allocate(%null : i64 -> %x : !llvm.ptr)
        private(@x.private %x -> %x.private : !llvm.ptr) {
      omp.terminator
    } {allocate_private_indices = array<i64: 0>}
    llvm.return
  }
}

// I386: target datalayout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-S128"
// I386-LABEL: define internal void @allocator_i386..omp_par
// I386: %[[ALLOC:.*]] = call ptr @__kmpc_alloc(i32 %{{.*}}, i32 4, ptr null)
// I386: call void @__kmpc_free(i32 %{{.*}}, ptr %[[ALLOC]], ptr null)
// I386: ret void
// I386-LABEL: declare noalias ptr @__kmpc_alloc(i32, i32, ptr)

//--- i386-overflow.mlir

module attributes {
  llvm.data_layout = "e-m:e-p:32:32-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:32-n8:16:32-S128",
  llvm.target_triple = "i386-unknown-linux-gnu"
} {
  omp.private {type = private} @x.private : !llvm.array<4294967296 x i8>

  llvm.func @allocator_i386_overflow(%x: !llvm.ptr) {
    %null = llvm.mlir.constant(0 : i64) : i64
    omp.parallel allocate(%null : i64 -> %x : !llvm.ptr)
        private(@x.private %x -> %x.private : !llvm.ptr) {
      omp.terminator
    } {allocate_private_indices = array<i64: 0>}
    llvm.return
  }
}

// I386-OVERFLOW-NOT: __kmpc_alloc
// I386-OVERFLOW: OpenMP allocation size cannot be represented by the target size type
// I386-OVERFLOW-NOT: __kmpc_alloc
// I386-OVERFLOW: LLVM Translation failed for operation: omp.parallel
// I386-OVERFLOW-NOT: __kmpc_alloc

//--- device.mlir

omp.private {type = private} @device.private : i32

llvm.func @allocator_device() {
  omp.target kernel_type(generic) {
    %allocator = llvm.mlir.constant(0 : i64) : i64
    %x = llvm.alloca %allocator x i32 : (i64) -> !llvm.ptr
    omp.parallel allocate(%allocator : i64 -> %x : !llvm.ptr)
        private(@device.private %x -> %private : !llvm.ptr) {
      omp.terminator
    } {allocate_private_indices = array<i64: 0>}
    omp.terminator
  }
  llvm.return
}

// DEVICE: allocate clause on a device parallel region is not supported
// DEVICE: LLVM Translation failed for operation: omp.parallel
