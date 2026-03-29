; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-to-args --test FileCheck --test-arg %s --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefix=REDUCED

; INTERESTING: store
; REDUCED: define void @test(ptr %a) {
; REDUCED-NEXT: %a1 = alloca i32
; REDUCED-NEXT: call void @llvm.lifetime.start.p0(ptr %a1)
; REDUCED-NEXT: store i32 0, ptr %a
; REDUCED-NEXT: store i32 1, ptr %a
; REDUCED-NEXT: call void @llvm.lifetime.end.p0(ptr %a1)
define void @test() {
  %a = alloca i32
  call void @llvm.lifetime.start.p0(ptr %a)
  store i32 0, ptr %a
  store i32 1, ptr %a
  call void @llvm.lifetime.end.p0(ptr %a)
  ret void
}
