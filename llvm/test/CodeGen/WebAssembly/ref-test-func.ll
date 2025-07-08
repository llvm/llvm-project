; RUN: llc < %s -mcpu=mvp -mattr=+reference-types | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: test_fpsig_1:
; CHECK: local.get	0
; CHECK-NEXT: table.get	__indirect_function_table
; CHECK-NEXT: ref.test (f32, f64, i32) -> (f32)
; CHECK-NEXT: call	use
; Function Attrs: nounwind
define void @test_fpsig_1(ptr noundef %func) local_unnamed_addr #0 {
entry:
  %res = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, float 0.000000e+00, float 0.000000e+00, double 0.000000e+00, i32 0)
  tail call void @use(i32 noundef %res) #3
  ret void
}

; CHECK-LABEL: test_fpsig_2:
; CHECK: local.get	0
; CHECK-NEXT: table.get	__indirect_function_table
; CHECK-NEXT: ref.test (f32, f64, i32) -> (i32)
; CHECK-NEXT: call	use
define void @test_fpsig_2(ptr noundef %func) local_unnamed_addr #0 {
entry:
  %res = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, i32 0, float 0.000000e+00, double 0.000000e+00, i32 0)
  tail call void @use(i32 noundef %res) #3
  ret void
}

; CHECK-LABEL: test_fpsig_3:
; CHECK: local.get	0
; CHECK-NEXT: table.get	__indirect_function_table
; CHECK-NEXT: ref.test (i32, i32, i32) -> (i32)
; CHECK-NEXT: call	use
define void @test_fpsig_3(ptr noundef %func) local_unnamed_addr #0 {
entry:
  %res = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, i32 0, i32 0, i32 0, i32 0)
  tail call void @use(i32 noundef %res) #3
  ret void
}

; CHECK-LABEL: test_fpsig_4:
; CHECK: local.get	0
; CHECK-NEXT: table.get	__indirect_function_table
; CHECK-NEXT: ref.test (i32, i32, i32) -> ()
; CHECK-NEXT: call	use
define void @test_fpsig_4(ptr noundef %func) local_unnamed_addr #0 {
entry:
  %res = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison, i32 0, i32 0, i32 0)
  tail call void @use(i32 noundef %res) #3
  ret void
}

; CHECK-LABEL: test_fpsig_5:
; CHECK: local.get	0
; CHECK-NEXT: table.get	__indirect_function_table
; CHECK-NEXT: ref.test () -> ()
; CHECK-NEXT: call	use
define void @test_fpsig_5(ptr noundef %func) local_unnamed_addr #0 {
entry:
  %res = tail call i32 (ptr, ...) @llvm.wasm.ref.test.func(ptr %func, token poison)
  tail call void @use(i32 noundef %res) #3
  ret void
}

declare void @use(i32 noundef) local_unnamed_addr #1
