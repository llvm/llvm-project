// RUN: mlir-translate --mlir-to-llvmir %s | FileCheck %s

// CHECK-LABEL: define void @test_interop_init(
// CHECK-SAME:    ptr %[[OBJ:.*]])
// CHECK:         %[[GTID:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
// CHECK:         call void @__tgt_interop_init(ptr @{{.*}}, i32 %[[GTID]], ptr %[[OBJ]], i32 1, i32 -1, i32 0, ptr null, i32 0)
// CHECK:         ret void
llvm.func @test_interop_init(%arg0: !llvm.ptr) {
  %0 = llvm.mlir.constant(1 : i64) : i64
  omp.interop.init %arg0 : !llvm.ptr interop_types([#omp<interop_type(target)>])
  llvm.return
}

// CHECK-LABEL: define void @test_interop_init_targetsync(
// CHECK:         call void @__tgt_interop_init(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 2, i32 -1, i32 0, ptr null, i32 0)
llvm.func @test_interop_init_targetsync(%arg0: !llvm.ptr) {
  omp.interop.init %arg0 : !llvm.ptr interop_types([#omp<interop_type(targetsync)>])
  llvm.return
}

// CHECK-LABEL: define void @test_interop_init_both(
// CHECK:         call void @__tgt_interop_init(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 1, i32 -1, i32 0, ptr null, i32 0)
// CHECK-NOT:     call void @__tgt_interop_init
llvm.func @test_interop_init_both(%arg0: !llvm.ptr) {
  omp.interop.init %arg0 : !llvm.ptr interop_types([#omp<interop_type(targetsync)>, #omp<interop_type(target)>])
  llvm.return
}

// CHECK-LABEL: define void @test_interop_use(
// CHECK:         %[[GTID:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
// CHECK:         call void @__tgt_interop_use(ptr @{{.*}}, i32 %[[GTID]], ptr %{{.*}}, i32 -1, i32 0, ptr null, i32 0)
// CHECK:         ret void
llvm.func @test_interop_use(%arg0: !llvm.ptr) {
  omp.interop.use %arg0 : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: define void @test_interop_destroy(
// CHECK:         %[[GTID:.*]] = call i32 @__kmpc_global_thread_num(ptr @{{.*}})
// CHECK:         call void @__tgt_interop_destroy(ptr @{{.*}}, i32 %[[GTID]], ptr %{{.*}}, i32 -1, i32 0, ptr null, i32 0)
// CHECK:         ret void
llvm.func @test_interop_destroy(%arg0: !llvm.ptr) {
  omp.interop.destroy %arg0 : !llvm.ptr
  llvm.return
}

// CHECK-LABEL: define void @test_interop_init_nowait(
// CHECK:         call void @__tgt_interop_init(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 1, i32 -1, i32 0, ptr null, i32 1)
llvm.func @test_interop_init_nowait(%arg0: !llvm.ptr) {
  omp.interop.init %arg0 : !llvm.ptr interop_types([#omp<interop_type(target)>]) nowait
  llvm.return
}

// CHECK-LABEL: define void @test_interop_destroy_nowait(
// CHECK:         call void @__tgt_interop_destroy(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 -1, i32 0, ptr null, i32 1)
llvm.func @test_interop_destroy_nowait(%arg0: !llvm.ptr) {
  omp.interop.destroy %arg0 : !llvm.ptr nowait
  llvm.return
}

// CHECK-LABEL: define void @test_interop_init_device(
// CHECK:         call void @__tgt_interop_init(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 1, i32 %{{.*}}, i32 0, ptr null, i32 0)
llvm.func @test_interop_init_device(%arg0: !llvm.ptr, %arg1: i32) {
  omp.interop.init %arg0 : !llvm.ptr interop_types([#omp<interop_type(target)>]) device(%arg1 : i32)
  llvm.return
}

// CHECK-LABEL: define void @test_interop_init_device_i64(
// CHECK:         %[[DEVICE:.*]] = trunc i64 %{{.*}} to i32
// CHECK:         call void @__tgt_interop_init(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 1, i32 %[[DEVICE]], i32 0, ptr null, i32 0)
llvm.func @test_interop_init_device_i64(%arg0: !llvm.ptr, %arg1: i64) {
  omp.interop.init %arg0 : !llvm.ptr interop_types([#omp<interop_type(target)>]) device(%arg1 : i64)
  llvm.return
}

// CHECK-LABEL: define void @test_interop_destroy_device_i64(
// CHECK:         %[[DEVICE:.*]] = trunc i64 %{{.*}} to i32
// CHECK:         call void @__tgt_interop_destroy(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 %[[DEVICE]], i32 0, ptr null, i32 0)
llvm.func @test_interop_destroy_device_i64(%arg0: !llvm.ptr, %arg1: i64) {
  omp.interop.destroy %arg0 : !llvm.ptr device(%arg1 : i64)
  llvm.return
}

// CHECK-LABEL: define void @test_interop_use_device_i64(
// CHECK:         %[[DEVICE:.*]] = trunc i64 %{{.*}} to i32
// CHECK:         call void @__tgt_interop_use(ptr @{{.*}}, i32 %{{.*}}, ptr %{{.*}}, i32 %[[DEVICE]], i32 0, ptr null, i32 0)
llvm.func @test_interop_use_device_i64(%arg0: !llvm.ptr, %arg1: i64) {
  omp.interop.use %arg0 : !llvm.ptr device(%arg1 : i64)
  llvm.return
}

// CHECK: declare i32 @__kmpc_global_thread_num(ptr)
// CHECK: declare void @__tgt_interop_init(ptr, i32, ptr, i32, i32, i32, ptr, i32)
// CHECK: declare void @__tgt_interop_use(ptr, i32, ptr, i32, i32, ptr, i32)
// CHECK: declare void @__tgt_interop_destroy(ptr, i32, ptr, i32, i32, ptr, i32)
