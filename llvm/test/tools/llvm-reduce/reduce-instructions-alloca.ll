; RUN: llvm-reduce --abort-on-invalid-reduction --delta-passes=instructions --test FileCheck --test-arg --check-prefixes=CHECK,INTERESTING --test-arg %s --test-arg --input-file %s -o %t
; RUN: FileCheck -check-prefixes=CHECK,RESULT %s < %t

; CHECK-LABEL: define void @alloca(
; INTERESTING: call void @llvm.lifetime.start.p0(
; INTERESTING: call void @llvm.lifetime.end.p0(

; RESULT: call void @llvm.lifetime.start.p0(ptr poison)
; RESULT-NEXT: call void @llvm.lifetime.end.p0(ptr poison)
; RESULT-NEXT: ret void
define void @alloca(ptr %ptr) {
  %alloca = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr %alloca)
  call void @llvm.lifetime.end.p0(ptr %alloca)
  ret void
}
