// RUN: split-file %s %t
// RUN: mlir-translate -mlir-to-llvmir -split-input-file %t/valid.mlir | FileCheck %s
// RUN: not mlir-translate -mlir-to-llvmir %t/device.mlir 2>&1 | FileCheck %s --check-prefix=DEVICE

//--- valid.mlir

omp.private {type = private} @x.private : i32
omp.private {type = private} @y.private : i32

llvm.func @scope_allocator_unaligned(%x: !llvm.ptr, %y: !llvm.ptr) {
  %null = llvm.mlir.constant(0 : i64) : i64
  omp.scope allocate(%null : i64 -> %x : !llvm.ptr) allocate_private_indices([0])
      private(@x.private %x -> %x.private,
              @y.private %y -> %y.private : !llvm.ptr, !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %x.private : i32, !llvm.ptr
    llvm.store %one, %y.private : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @scope_allocator_unaligned
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: %[[Y_ALLOCA:.*]] = alloca i32, align 4
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK-NOT: call void @__kmpc_free
// CHECK: %[[UNALIGNED:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr null)
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK-NOT: call void @__kmpc_free
// CHECK: store i32 1, ptr %[[UNALIGNED]], align 4
// CHECK: store i32 1, ptr %[[Y_ALLOCA]], align 4
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[UNALIGNED]], ptr null)
// CHECK-NOT: call void @__kmpc_free
// CHECK: ret void

// -----

omp.private {type = private} @constant.private : i32

llvm.func @scope_allocator_constant(%x: !llvm.ptr) {
  %allocator = llvm.mlir.constant(3 : i64) : i64
  omp.scope allocate(%allocator : i64 -> %x : !llvm.ptr) allocate_private_indices([0])
      private(@constant.private %x -> %x.private : !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %x.private : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @scope_allocator_constant
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: %[[CONSTANT_ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr inttoptr (i64 3 to ptr))
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: store i32 1, ptr %[[CONSTANT_ALLOC]], align 4
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[CONSTANT_ALLOC]], ptr inttoptr (i64 3 to ptr))
// CHECK-NOT: call void @__kmpc_free
// CHECK: ret void

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

llvm.func @scope_allocator_firstprivate_aligned(%x: !llvm.ptr, %allocator: i64) {
  omp.scope allocate(%allocator : i64 -> %x : !llvm.ptr) allocate_alignments([64]) allocate_private_indices([0])
      private(@x.private %x -> %x.private : !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %x.private : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @scope_allocator_firstprivate_aligned
// CHECK: %[[ALLOCATOR:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: %[[ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %{{.*}}, i64 64, i64 4, ptr %[[ALLOCATOR]])
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: %[[ORIGINAL:.*]] = load i32, ptr %{{.*}}, align 4
// CHECK: store i32 %[[ORIGINAL]], ptr %[[ALLOC]], align 4
// CHECK: store i32 1, ptr %[[ALLOC]], align 4
// CHECK: call void @private_dealloc(ptr %[[ALLOC]])
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[ALLOC]], ptr %[[ALLOCATOR]])
// CHECK-NOT: call void @__kmpc_free
// CHECK: ret void

// -----

omp.private {type = private} @x.private : i32
omp.private {type = private} @y.private : i32

llvm.func @scope_allocator_order(%x: !llvm.ptr, %y: !llvm.ptr,
                                 %allocator.x: i64, %allocator.y: i64) {
  omp.scope allocate(%allocator.y : i64 -> %y : !llvm.ptr,
                     %allocator.x : i64 -> %x : !llvm.ptr) allocate_alignments([128, 64]) allocate_private_indices([1, 0])
      private(@x.private %x -> %x.private,
              @y.private %y -> %y.private : !llvm.ptr, !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %x.private : i32, !llvm.ptr
    llvm.store %one, %y.private : i32, !llvm.ptr
    omp.terminator
  }
  llvm.return
}

// Allocation-list order (y, x) differs from private-list order (x, y); the
// allocate_private_indices mapping must still route each allocation to the
// correct private slot, and cleanup must free exactly once per allocation.
// CHECK-LABEL: define void @scope_allocator_order
// CHECK: %[[ALLOCATOR_Y:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK: %[[ALLOCATOR_X:.*]] = inttoptr i64 %{{.*}} to ptr
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: %[[X_ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %{{.*}}, i64 64, i64 4, ptr %[[ALLOCATOR_X]])
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: %[[Y_ALLOC:.*]] = call ptr @__kmpc_aligned_alloc(i32 %{{.*}}, i64 128, i64 4, ptr %[[ALLOCATOR_Y]])
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: store i32 1, ptr %[[X_ALLOC]], align 4
// CHECK: store i32 1, ptr %[[Y_ALLOC]], align 4
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[Y_ALLOC]], ptr %[[ALLOCATOR_Y]])
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[X_ALLOC]], ptr %[[ALLOCATOR_X]])
// CHECK-NOT: call void @__kmpc_free
// CHECK: ret void

// -----

omp.private {type = private} @conditional.private : i32

llvm.func @scope_allocator_conditional(%x: !llvm.ptr, %allocator: i64,
                                       %condition: i1) {
  llvm.cond_br %condition, ^scope, ^exit
^scope:
  omp.scope allocate(%allocator : i64 -> %x : !llvm.ptr) allocate_private_indices([0])
      private(@conditional.private %x -> %x.private : !llvm.ptr) {
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %x.private : i32, !llvm.ptr
    omp.terminator
  }
  llvm.br ^exit
^exit:
  llvm.return
}

// CHECK-LABEL: define void @scope_allocator_conditional
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: br i1 {{.*}}, label %[[SCOPE:.*]], label %[[EXIT:.*]]
// CHECK: [[SCOPE]]:
// CHECK: %[[CONDITIONAL_ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr %{{.*}})
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: store i32 1, ptr %[[CONDITIONAL_ALLOC]], align 4
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[CONDITIONAL_ALLOC]], ptr %{{.*}})
// CHECK-NOT: call void @__kmpc_free
// CHECK: br label %[[EXIT]]
// CHECK: [[EXIT]]:
// CHECK: ret void

// -----

omp.private {type = private} @loop.private : i32

llvm.func @scope_allocator_loop(%x: !llvm.ptr, %allocator.base: i64, %n: i64) {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %one = llvm.mlir.constant(1 : i64) : i64
  llvm.br ^loop(%zero : i64)
^loop(%i: i64):
  %condition = llvm.icmp "slt" %i, %n : i64
  llvm.cond_br %condition, ^body, ^exit
^body:
  %allocator = llvm.add %allocator.base, %i : i64
  omp.scope allocate(%allocator : i64 -> %x : !llvm.ptr) allocate_private_indices([0])
      private(@loop.private %x -> %x.private : !llvm.ptr) {
    %value = llvm.trunc %i : i64 to i32
    llvm.store %value, %x.private : i32, !llvm.ptr
    omp.terminator
  }
  %next = llvm.add %i, %one : i64
  llvm.br ^loop(%next : i64)
^exit:
  llvm.return
}

// CHECK-LABEL: define void @scope_allocator_loop
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: omp.region.after_alloca:
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: br label %[[LOOP:[0-9]+]]
// CHECK: [[LOOP]]:
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: br i1 {{.*}}, label %[[BODY:[0-9]+]], label %[[LOOP_EXIT:[0-9]+]]
// CHECK: [[BODY]]:
// CHECK: %[[DYNAMIC_ALLOCATOR:.*]] = add i64 %{{.*}}, %{{.*}}
// CHECK: %[[ALLOCATOR_HANDLE:.*]] = inttoptr i64 %[[DYNAMIC_ALLOCATOR]] to ptr
// CHECK: %[[LOOP_ALLOC:.*]] = call ptr @__kmpc_alloc({{.*}}, i64 4, ptr %[[ALLOCATOR_HANDLE]])
// CHECK-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// CHECK: store i32 {{.*}}, ptr %[[LOOP_ALLOC]], align 4
// CHECK-NOT: call void @__kmpc_free
// CHECK: call void @__kmpc_free({{.*}}, ptr %[[LOOP_ALLOC]], ptr %[[ALLOCATOR_HANDLE]])
// CHECK-NOT: call void @__kmpc_free
// CHECK: br label %[[LOOP]]
// CHECK: [[LOOP_EXIT]]:
// CHECK: ret void

//--- device.mlir

omp.private {type = private} @device.private : i32

llvm.func @scope_allocator_device() {
  omp.target kernel_type(generic) {
    %allocator = llvm.mlir.constant(0 : i64) : i64
    %x = llvm.alloca %allocator x i32 : (i64) -> !llvm.ptr
    omp.scope allocate(%allocator : i64 -> %x : !llvm.ptr) allocate_alignments([64]) allocate_private_indices([0])
        private(@device.private %x -> %private : !llvm.ptr) {
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}

// DEVICE-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
// DEVICE: allocate clause in an OpenMP device context is not supported
// DEVICE: LLVM Translation failed for operation: omp.scope
// DEVICE-NOT: call ptr @__kmpc_{{(aligned_)?}}alloc
