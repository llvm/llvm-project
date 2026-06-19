; RUN: opt -passes=sandbox-vectorizer -sbvec-vec-reg-bits=1024 -disable-output -sbvec-passes="seed-collection<print-region>" -sbvec-collect-seeds=loads %s | FileCheck %s
; REQUIRES: asserts

; Check that the seed collector will form a aux region containing the loads.
define void @load_seeds(ptr %ptrA, ptr %ptrB) {
; CHECK-LABEL: Aux:
; CHECK-NEXT:  %ld0 = load i8, ptr %ptrA0
; CHECK-NEXT:  %ld1 = load i8, ptr %ptrA1

  %ptrA0 = getelementptr i8, ptr %ptrA, i32 0
  %ptrA1 = getelementptr i8, ptr %ptrA, i32 1
  %ld0 = load i8, ptr %ptrA0
  %ld1 = load i8, ptr %ptrA1
  ret void
}
