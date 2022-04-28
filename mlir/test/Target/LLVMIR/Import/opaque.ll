; RUN: mlir-translate -import-llvm -split-input-file -opaque-pointers %s | FileCheck %s

; CHECK-LABEL: @opaque_ptr_load
define i32 @opaque_ptr_load(ptr %0) {
  ; CHECK: = llvm.load %{{.*}} : !llvm.ptr -> i32
  %2 = load i32, ptr %0, align 4
  ret i32 %2
}

; // -----

; CHECK-LABEL: @opaque_ptr_store
define void @opaque_ptr_store(i32 %0, ptr %1) {
  ; CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr
  store i32 %0, ptr %1, align 4
  ret void
}

; // -----

; CHECK-LABEL: @opaque_ptr_ptr_store
define void @opaque_ptr_ptr_store(ptr %0, ptr %1) {
  ; CHECK: llvm.store %{{.*}}, %{{.*}} : !llvm.ptr, !llvm.ptr
  store ptr %0, ptr %1, align 8
  ret void
}

; // -----

; CHECK-LABEL: @opaque_ptr_alloca
define ptr @opaque_ptr_alloca(i32 %0) {
  ; CHECK: = llvm.alloca %{{.*}} x f32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %2 = alloca float, i32 %0, align 4
  ret ptr %2
}

; // -----

; CHECK-LABEL: @opaque_ptr_gep
define ptr @opaque_ptr_gep(ptr %0, i32 %1) {
  ; CHECK: = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  %3 = getelementptr float, ptr %0, i32 %1
  ret ptr %3
}

; // -----

; CHECK-LABEL: @opaque_ptr_gep
define ptr @opaque_ptr_gep_struct(ptr %0, i32 %1){
  ; CHECK: = llvm.getelementptr %{{.*}}[%{{.*}}, 0, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(struct<(f32, f64)>, struct<(i32, i64)>)>
  %3 = getelementptr { { float, double }, { i32, i64 } }, ptr %0, i32 %1, i32 0, i32 1
  ret ptr %3
}
