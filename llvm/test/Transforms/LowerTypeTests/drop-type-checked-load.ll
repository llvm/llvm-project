;; Dropping type tests must also drop @llvm.type.checked.load, but only when
;; dropping everything. Only WholeProgramDevirt and FatLTOCleanup remove those
;; otherwise, so a pipeline that drops the tests and then goes on to codegen
;; without them would carry them into ISel, which rejects them with "intrinsic
;; must be lowered by the LowerTypeTests pass before code generation". In the
;; default Assume mode the loads stay: there they may feed a CFI trap check,
;; which must keep diagnosing loudly.

; RUN: opt -S -passes="drop-type-tests<all>" < %s | FileCheck %s
; RUN: opt -S -passes="drop-type-tests" < %s | FileCheck %s --check-prefix=ASSUME
; RUN: opt -S -passes="drop-type-tests<all>,function(instsimplify)" < %s | FileCheck %s --check-prefix=FOLD

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare i1 @llvm.type.test(ptr, metadata)
declare {ptr, i1} @llvm.type.checked.load(ptr, i32, metadata)
declare {ptr, i1} @llvm.type.checked.load.relative(ptr, i32, metadata)
declare void @llvm.assume(i1)
declare void @use(ptr)

@fn = dso_local global i32 0
@vt = constant [1 x i32] [i32 trunc (i64 sub (i64 ptrtoint (ptr @fn to i64), i64 ptrtoint (ptr @vt to i64)) to i32)]

;; The guarded load stays, the check becomes true, and the intrinsic is gone.
; CHECK-LABEL: define void @absolute(
; CHECK-NEXT: %[[SLOT:.*]] = getelementptr i8, ptr %vtable, i32 8
; CHECK-NEXT: %[[FP:.*]] = load ptr, ptr %[[SLOT]], align 8
; CHECK-NEXT: %[[S0:.*]] = insertvalue { ptr, i1 } { ptr poison, i1 true }, ptr %[[FP]], 0
; CHECK-NOT: call {{.*}}@llvm.type.checked.load(
; ASSUME-LABEL: define void @absolute(
; ASSUME: call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid")
define void @absolute(ptr %vtable) {
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  %pred = extractvalue {ptr, i1} %pair, 1
  br i1 %pred, label %call, label %fail

call:
  call void @use(ptr %fptr)
  ret void

fail:
  ret void
}

;; The relative form becomes llvm.load.relative: a 32-bit displacement added
;; to the vtable pointer. The non-zero offset pins the base: it is %vtable,
;; not the slot.
; CHECK-LABEL: define void @relative(
; CHECK-NEXT: %[[REL:.*]] = call ptr @llvm.load.relative.i32(ptr %vtable, i32 4)
; CHECK-NEXT: %[[S0:.*]] = insertvalue { ptr, i1 } { ptr poison, i1 true }, ptr %[[REL]], 0
; CHECK-NOT: call {{.*}}@llvm.type.checked.load.relative(
; ASSUME-LABEL: define void @relative(
; ASSUME: call { ptr, i1 } @llvm.type.checked.load.relative(ptr %vtable, i32 4, metadata !"typeid")
define void @relative(ptr %vtable) {
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr %vtable, i32 4, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void @use(ptr %fptr)
  ret void
}

;; A type test in the same module drops alongside the checked load.
; CHECK-LABEL: define void @with_type_test(
; CHECK-NOT: call {{.*}}@llvm.type.test(
; CHECK: %[[FP:.*]] = load ptr, ptr %{{.*}}, align 8
; CHECK-NOT: call {{.*}}@llvm.type.checked.load(
; ASSUME-LABEL: define void @with_type_test(
; ASSUME-NOT: call {{.*}}@llvm.type.test(
; ASSUME: call { ptr, i1 } @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid")
define void @with_type_test(ptr %vtable) {
  %t = call i1 @llvm.type.test(ptr %vtable, metadata !"typeid")
  call void @llvm.assume(i1 %t)
  %pair = call {ptr, i1} @llvm.type.checked.load(ptr %vtable, i32 8, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void @use(ptr %fptr)
  ret void
}

;; With a constant vtable the relative load folds all the way to the function.
; FOLD-LABEL: define void @fold(
; FOLD-NOT: @llvm.load.relative
; FOLD: call void @use(ptr @fn)
define void @fold() {
  %pair = call {ptr, i1} @llvm.type.checked.load.relative(ptr @vt, i32 0, metadata !"typeid")
  %fptr = extractvalue {ptr, i1} %pair, 0
  call void @use(ptr %fptr)
  ret void
}
