; RUN: llc < %s -mcpu=mvp -mattr=+reference-types | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: test_function_pointer_signature_void:
; CHECK-NEXT: .functype	test_function_pointer_signature_void (i32) -> ()
; CHECK-NEXT: .local funcref
; CHECK: local.get	0
; CHECK-NEXT: table.get	__indirect_function_table
; CHECK-NEXT: local.tee	1
; CHECK-NEXT: ref.test (f32, f64, i32) -> (f32)
; CHECK-NEXT: call	use
; CHECK-NEXT: local.get	1
; CHECK-NEXT: ref.test (f32, f64, i32) -> (i32)
; CHECK-NEXT: call	use
; CHECK-NEXT: local.get	1
; CHECK-NEXT: ref.test (i32, i32, i32) -> (i32)
; CHECK-NEXT: call	use
; CHECK-NEXT: local.get	1
; CHECK-NEXT: ref.test (i32, i32, i32) -> ()
; CHECK-NEXT: call	use
; CHECK-NEXT: local.get	1
; CHECK-NEXT: ref.test () -> ()
; CHECK-NEXT: call	use

; Function Attrs: nounwind
define void @test_function_pointer_signature_void(ptr noundef %func) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, float 0.000000e+00, float 0.000000e+00, double 0.000000e+00, i32 0)
  tail call void @use(i32 noundef %0) #3
  %1 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, i32 0, float 0.000000e+00, double 0.000000e+00, i32 0)
  tail call void @use(i32 noundef %1) #3
  %2 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, i32 0, i32 0, i32 0, i32 0)
  tail call void @use(i32 noundef %2) #3
  %3 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, i32 0, i32 0, i32 0)
  tail call void @use(i32 noundef %3) #3
  %4 = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison)
  tail call void @use(i32 noundef %4) #3
  ret void
}

declare void @use(i32 noundef) local_unnamed_addr #1
