// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope --implicit-check-not='call void @__kmpc_barrier'

llvm.func @initialize(!llvm.ptr, !llvm.ptr)
llvm.func @destroy(!llvm.ptr)
llvm.func @use_private(!llvm.ptr, !llvm.ptr)
llvm.func @copy_i32(!llvm.ptr, !llvm.ptr)

omp.private {type = private} @private_i32 : i32 init {
^bb0(%mold: !llvm.ptr, %private: !llvm.ptr):
  llvm.call @initialize(%mold, %private) : (!llvm.ptr, !llvm.ptr) -> ()
  omp.yield(%private : !llvm.ptr)
} dealloc {
^bb0(%private: !llvm.ptr):
  llvm.call @destroy(%private) : (!llvm.ptr) -> ()
  omp.yield
}

omp.private {type = firstprivate} @firstprivate_i32 : i32 copy {
^bb0(%mold: !llvm.ptr, %private: !llvm.ptr):
  %value = llvm.load %mold : !llvm.ptr -> i32
  llvm.store %value, %private : i32, !llvm.ptr
  omp.yield(%private : !llvm.ptr)
} dealloc {
^bb0(%private: !llvm.ptr):
  llvm.call @destroy(%private) : (!llvm.ptr) -> ()
  omp.yield
}

// Initialization, firstprivate copying and cleanup must be executed only by
// the thread selected for the single region. Even private_barrier must not
// introduce a team barrier inside this conditional region.
// CHECK-LABEL: define void @single_private(
// CHECK-SAME: ptr %[[X:[^,]+]], ptr %[[Y:[^,)]+]]
// CHECK: %[[P:.*]] = alloca i32
// CHECK: %[[FP:.*]] = alloca i32
// CHECK: %[[SINGLE:.*]] = call i32 @__kmpc_single
// CHECK-NEXT: %[[COND:.*]] = icmp ne i32 %[[SINGLE]], 0
// CHECK-NEXT: br i1 %[[COND]], label %[[BODY:.*]], label %[[END:.*]]
// CHECK: [[BODY]]:
// CHECK-NEXT: br label %[[INIT:.*]]
// CHECK: [[INIT]]:
// CHECK-NEXT: call void @initialize(ptr %[[X]], ptr %[[P]])
// CHECK-NEXT: br label %[[COPY:.*]]
// CHECK: [[COPY]]:
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[Y]]
// CHECK-NEXT: store i32 %[[VALUE]], ptr %[[FP]]
// CHECK: call void @use_private(ptr %[[P]], ptr %[[FP]])
// CHECK: call void @destroy(ptr %[[P]])
// CHECK-NEXT: call void @destroy(ptr %[[FP]])
// CHECK: call void @__kmpc_end_single
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]:
// CHECK: call void @__kmpc_barrier
// CHECK: ret void
llvm.func @single_private(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.single private(@private_i32 %x -> %px, @firstprivate_i32 %y -> %py : !llvm.ptr, !llvm.ptr) private_barrier {
    llvm.call @use_private(%px, %py) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @single_private_nowait(
// CHECK-SAME: ptr %[[X:[^,]+]], ptr %[[Y:[^,)]+]]
// CHECK: %[[P:.*]] = alloca i32
// CHECK: %[[FP:.*]] = alloca i32
// CHECK: %[[SINGLE:.*]] = call i32 @__kmpc_single
// CHECK-NEXT: %[[COND:.*]] = icmp ne i32 %[[SINGLE]], 0
// CHECK-NEXT: br i1 %[[COND]], label %[[BODY:.*]], label %[[END:.*]]
// CHECK: [[BODY]]:
// CHECK-NEXT: br label %[[INIT:.*]]
// CHECK: [[INIT]]:
// CHECK-NEXT: call void @initialize(ptr %[[X]], ptr %[[P]])
// CHECK-NEXT: br label %[[COPY:.*]]
// CHECK: [[COPY]]:
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[Y]]
// CHECK-NEXT: store i32 %[[VALUE]], ptr %[[FP]]
// CHECK: call void @use_private(ptr %[[P]], ptr %[[FP]])
// CHECK: call void @destroy(ptr %[[P]])
// CHECK-NEXT: call void @destroy(ptr %[[FP]])
// CHECK: call void @__kmpc_end_single
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]:
// CHECK: ret void
llvm.func @single_private_nowait(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.single nowait private(@private_i32 %x -> %px, @firstprivate_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    llvm.call @use_private(%px, %py) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// Copyprivate operands refer to storage in the enclosing context and are
// broadcast after the selected thread has finished its private cleanup.
// CHECK-LABEL: define void @single_private_copyprivate(
// CHECK-SAME: ptr %[[X:[^,]+]], ptr %[[Y:[^,)]+]], ptr %[[Z:[^,)]+]]
// CHECK: %[[P:.*]] = alloca i32
// CHECK: %[[FP:.*]] = alloca i32
// CHECK: %[[SINGLE:.*]] = call i32 @__kmpc_single
// CHECK-NEXT: %[[COND:.*]] = icmp ne i32 %[[SINGLE]], 0
// CHECK-NEXT: br i1 %[[COND]], label %[[BODY:.*]], label %[[END:.*]]
// CHECK: [[BODY]]:
// CHECK-NEXT: br label %[[INIT:.*]]
// CHECK: [[INIT]]:
// CHECK-NEXT: call void @initialize(ptr %[[X]], ptr %[[P]])
// CHECK-NEXT: br label %[[COPY:.*]]
// CHECK: [[COPY]]:
// CHECK-NEXT: %[[VALUE:.*]] = load i32, ptr %[[Y]]
// CHECK-NEXT: store i32 %[[VALUE]], ptr %[[FP]]
// CHECK: call void @use_private(ptr %[[P]], ptr %[[FP]])
// CHECK: call void @destroy(ptr %[[P]])
// CHECK-NEXT: call void @destroy(ptr %[[FP]])
// CHECK: call void @__kmpc_end_single
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]:
// CHECK: call void @__kmpc_copyprivate({{.*}}, ptr %[[Z]], ptr @copy_i32,
// CHECK: ret void
llvm.func @single_private_copyprivate(%x: !llvm.ptr, %y: !llvm.ptr, %z: !llvm.ptr) {
  omp.single copyprivate(%z -> @copy_i32 : !llvm.ptr) private(@private_i32 %x -> %px, @firstprivate_i32 %y -> %py : !llvm.ptr, !llvm.ptr) {
    llvm.call @use_private(%px, %py) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}
