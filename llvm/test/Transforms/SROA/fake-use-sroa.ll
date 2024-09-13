; RUN: opt -S -passes=sroa %s | FileCheck %s
; With fake use instrinsics generated for small aggregates, check that when
; SROA slices the aggregate, we generate individual fake use intrinsics for
; the individual values.

; Generated from the following source:
; struct s {
;   int i;
;   int j;
; };
;
; void foo(struct s S) {
; }
;
; void bar() {
;   int arr[2] = {5, 6};
; }
;
%struct.s = type { i32, i32 }
@__const.bar.arr = private unnamed_addr constant [2 x i32] [i32 5, i32 6], align 4

; A small struct passed as parameter
; CHECK-LABEL: define{{.*}}foo
; CHECK:       %[[SLICE1:[^ ]+]] = trunc i64
; CHECK:       %[[SLICE2:[^ ]+]] = trunc i64
; CHECK-DAG:   call{{.*}} @llvm.fake.use(i32 %[[SLICE1]])
; CHECK-DAG:   call{{.*}} @llvm.fake.use(i32 %[[SLICE2]])
define dso_local void @foo(i64 %S.coerce) optdebug {
entry:
  %S = alloca %struct.s, align 4
  store i64 %S.coerce, ptr %S, align 4
  %fake.use = load %struct.s, ptr %S, align 4
  notail call void (...) @llvm.fake.use(%struct.s %fake.use)
  ret void
}

; A local variable with a small array type.
; CHECK-LABEL: define{{.*}}bar
; CHECK:       %[[ARRAYSLICE1:[^ ]+]] = load
; CHECK:       %[[ARRAYSLICE2:[^ ]+]] = load
; CHECK-DAG:   call{{.*}} @llvm.fake.use(i32 %[[ARRAYSLICE1]])
; CHECK-DAG:   call{{.*}} @llvm.fake.use(i32 %[[ARRAYSLICE2]])
define dso_local void @bar() optdebug {
entry:
  %arr = alloca [2 x i32], align 4
  call void @llvm.memcpy.p0i8.p0i8.i64(ptr align 4 %arr, ptr align 4 bitcast (ptr @__const.bar.arr to ptr), i64 8, i1 false)
  %fake.use = load [2 x i32], ptr %arr, align 4
  notail call void (...) @llvm.fake.use([2 x i32] %fake.use)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)
