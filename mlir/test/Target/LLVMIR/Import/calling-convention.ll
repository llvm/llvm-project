; RUN: mlir-translate -import-llvm -split-input-file %s | FileCheck %s

; CHECK: llvm.func fastcc @cconv_fastcc()
declare fastcc void @cconv_fastcc()
; CHECK: llvm.func @cconv_ccc()
declare ccc void @cconv_ccc()

; CHECK-LABEL: @call_cconv
define void @call_cconv() {
  ; CHECK: llvm.call fastcc @cconv_fastcc()
  call fastcc void @cconv_fastcc()
  ; CHECK: llvm.call @cconv_ccc()
  call ccc void @cconv_ccc()
  ret void
}

; // -----

; CHECK: llvm.func fastcc @cconv_fastcc()
declare fastcc void @cconv_fastcc()
declare i32 @__gxx_personality_v0(...)

; CHECK-LABEL: @invoke_cconv
define i32 @invoke_cconv() personality ptr @__gxx_personality_v0 {
  ; CHECK: llvm.invoke fastcc @cconv_fastcc() to ^bb2 unwind ^bb1 : () -> ()
  invoke fastcc void @cconv_fastcc() to label %bb2 unwind label %bb1
bb1:
  %1 = landingpad { ptr, i32 } cleanup
  br label %bb2
bb2:
  ret i32 1
}
