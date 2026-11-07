// RUN: mlir-translate -mlir-to-llvmir %s | FileCheck %s --enable-var-scope

llvm.func @destroy(!llvm.ptr)
llvm.func @use_private(!llvm.ptr, !llvm.ptr)
llvm.func @copy_i32(!llvm.ptr, !llvm.ptr)

// Model array cleanup with a conditional deallocation. Both paths must reach
// the next cleanup and eventually end_single, before any barrier or broadcast.
omp.private {type = firstprivate} @firstprivate_ptr : !llvm.ptr copy {
^bb0(%mold: !llvm.ptr, %private: !llvm.ptr):
  %value = llvm.load %mold : !llvm.ptr -> !llvm.ptr
  llvm.store %value, %private : !llvm.ptr, !llvm.ptr
  omp.yield(%private : !llvm.ptr)
} dealloc {
^bb0(%private: !llvm.ptr):
  %value = llvm.load %private : !llvm.ptr -> !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %allocated = llvm.icmp "ne" %value, %null : !llvm.ptr
  llvm.cond_br %allocated, ^destroy, ^done
^destroy:
  llvm.call @destroy(%value) : (!llvm.ptr) -> ()
  llvm.br ^done
^done:
  omp.yield
}

// CHECK-LABEL: define void @single_cleanup(
// CHECK: %[[SINGLE:.*]] = call i32 @__kmpc_single
// CHECK-NEXT: %[[COND:.*]] = icmp ne i32 %[[SINGLE]], 0
// CHECK-NEXT: br i1 %[[COND]], label %[[BODY:.*]], label %[[END:.*]]
// CHECK: [[BODY]]:
// CHECK: call void @use_private
// CHECK-NEXT: br label %[[BODY_CONT:.*]]
// CHECK: [[BODY_CONT]]:
// CHECK-NEXT: br label %[[CLEAN1:.*]]
// CHECK-NOT: call void @__kmpc_end_single
// CHECK: [[DONE1:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: br label %[[CONT1:.*]]
// CHECK: [[DESTROY1:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: call void @destroy
// CHECK-NEXT: br label %[[DONE1]]
// CHECK: [[CLEAN1]]:
// CHECK: br i1 %{{.*}}, label %[[DESTROY1]], label %[[DONE1]]
// CHECK: [[CONT1]]:
// CHECK-NEXT: br label %[[CLEAN2:.*]]
// CHECK: [[DONE2:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: br label %[[CONT2:.*]]
// CHECK: [[DESTROY2:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: call void @destroy
// CHECK-NEXT: br label %[[DONE2]]
// CHECK: [[CLEAN2]]:
// CHECK: br i1 %{{.*}}, label %[[DESTROY2]], label %[[DONE2]]
// CHECK: [[CONT2]]:
// CHECK-NEXT: br label %[[FINALIZE:.*]]
// CHECK: [[FINALIZE]]:
// CHECK-NEXT: call void @__kmpc_end_single
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]:
// CHECK: call void @__kmpc_barrier
// CHECK: ret void
llvm.func @single_cleanup(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.single private(@firstprivate_ptr %x -> %px, @firstprivate_ptr %y -> %py : !llvm.ptr, !llvm.ptr) {
    llvm.call @use_private(%px, %py) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @single_cleanup_nowait(
// CHECK: %[[SINGLE:.*]] = call i32 @__kmpc_single
// CHECK-NEXT: %[[COND:.*]] = icmp ne i32 %[[SINGLE]], 0
// CHECK-NEXT: br i1 %[[COND]], label %[[BODY:.*]], label %[[END:.*]]
// CHECK: [[BODY]]:
// CHECK: call void @use_private
// CHECK-NEXT: br label %[[BODY_CONT:.*]]
// CHECK: [[BODY_CONT]]:
// CHECK-NEXT: br label %[[CLEAN1:.*]]
// CHECK-NOT: call void @__kmpc_end_single
// CHECK: [[DONE1:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: br label %[[CONT1:.*]]
// CHECK: [[DESTROY1:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: call void @destroy
// CHECK-NEXT: br label %[[DONE1]]
// CHECK: [[CLEAN1]]:
// CHECK: br i1 %{{.*}}, label %[[DESTROY1]], label %[[DONE1]]
// CHECK: [[CONT1]]:
// CHECK-NEXT: br label %[[CLEAN2:.*]]
// CHECK: [[DONE2:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: br label %[[CONT2:.*]]
// CHECK: [[DESTROY2:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: call void @destroy
// CHECK-NEXT: br label %[[DONE2]]
// CHECK: [[CLEAN2]]:
// CHECK: br i1 %{{.*}}, label %[[DESTROY2]], label %[[DONE2]]
// CHECK: [[CONT2]]:
// CHECK-NEXT: br label %[[FINALIZE:.*]]
// CHECK: [[FINALIZE]]:
// CHECK-NEXT: call void @__kmpc_end_single
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]:
// CHECK-NOT: call void @__kmpc_barrier
// CHECK: ret void
llvm.func @single_cleanup_nowait(%x: !llvm.ptr, %y: !llvm.ptr) {
  omp.single nowait private(@firstprivate_ptr %x -> %px, @firstprivate_ptr %y -> %py : !llvm.ptr, !llvm.ptr) {
    llvm.call @use_private(%px, %py) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define void @single_cleanup_copyprivate(
// CHECK-SAME: ptr %{{[^,]+}}, ptr %{{[^,]+}}, ptr %[[Z:[^,)]+]]
// CHECK: %[[SINGLE:.*]] = call i32 @__kmpc_single
// CHECK-NEXT: %[[COND:.*]] = icmp ne i32 %[[SINGLE]], 0
// CHECK-NEXT: br i1 %[[COND]], label %[[BODY:.*]], label %[[END:.*]]
// CHECK: [[BODY]]:
// CHECK: call void @use_private
// CHECK-NEXT: br label %[[BODY_CONT:.*]]
// CHECK: [[BODY_CONT]]:
// CHECK-NEXT: br label %[[CLEAN1:.*]]
// CHECK-NOT: call void @__kmpc_end_single
// CHECK: [[DONE1:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: br label %[[CONT1:.*]]
// CHECK: [[DESTROY1:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: call void @destroy
// CHECK-NEXT: br label %[[DONE1]]
// CHECK: [[CLEAN1]]:
// CHECK: br i1 %{{.*}}, label %[[DESTROY1]], label %[[DONE1]]
// CHECK: [[CONT1]]:
// CHECK-NEXT: br label %[[CLEAN2:.*]]
// CHECK: [[DONE2:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: br label %[[CONT2:.*]]
// CHECK: [[DESTROY2:omp.private.dealloc[^:]*]]:
// CHECK-NEXT: call void @destroy
// CHECK-NEXT: br label %[[DONE2]]
// CHECK: [[CLEAN2]]:
// CHECK: br i1 %{{.*}}, label %[[DESTROY2]], label %[[DONE2]]
// CHECK: [[CONT2]]:
// CHECK-NEXT: br label %[[FINALIZE:.*]]
// CHECK: [[FINALIZE]]:
// CHECK-NEXT: store i32 1, ptr %{{.*}}
// CHECK-NEXT: call void @__kmpc_end_single
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]:
// CHECK: call void @__kmpc_copyprivate({{.*}}, ptr %[[Z]], ptr @copy_i32,
// CHECK: ret void
llvm.func @single_cleanup_copyprivate(%x: !llvm.ptr, %y: !llvm.ptr, %z: !llvm.ptr) {
  omp.single copyprivate(%z -> @copy_i32 : !llvm.ptr) private(@firstprivate_ptr %x -> %px, @firstprivate_ptr %y -> %py : !llvm.ptr, !llvm.ptr) {
    llvm.call @use_private(%px, %py) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}
