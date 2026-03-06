; XFAIL: *
; The verifier should xeject this
; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=ifuncs --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck --check-prefixes=CHECK-FINAL --input-file=%t %s

; CHECK-INTERESTINGNESS: @ifunc_getelementptr

; FIXME: Why is this legal?
@ifunc_getelementptr = ifunc void (), ptr getelementptr (i8, ptr @resolver1, i32 4)

define ptr @resolver1() {
  ret ptr inttoptr (i64 123 to ptr)
}

define void @call_ifunc_getelementptr(ptr %ptr) {
  ; CHECK-FINAL-LABEL: define void @call_ifunc_getelementptr(ptr %ptr) {
  ; CHECK-FINAL-NEXT: call void @ifunc_getelementptr()
  ; CHECK-FINAL-NEXT: store ptr @ifunc_getelementptr, ptr %ptr, align 8
  ; CHECK-FINAL-NEXT: store ptr %ptr, ptr @ifunc_getelementptr, align 8
  ; CHECK-FINAL-NEXT: ret void
  call void @ifunc_getelementptr()
  store ptr @ifunc_getelementptr, ptr %ptr
  store ptr %ptr, ptr @ifunc_getelementptr
  ret void
}


