; RUN: llvm-dis %s.bc -o - | FileCheck %s
; RUN: verify-uselistorder %s.bc

; musttail-bitcast-upgrade.ll.bc was produced from the IR below using an
; llvm-as that still emitted the optional no-op bitcast between a musttail call
; and its return. The reader must drop that bitcast so the module verifies.

; CHECK-LABEL: define ptr @caller(ptr %a)
; CHECK-NEXT:    %c = musttail call ptr @callee(ptr %a)
; CHECK-NEXT:    ret ptr %c

define ptr @callee(ptr %a) {
  ret ptr %a
}

define ptr @caller(ptr %a) {
  %c = musttail call ptr @callee(ptr %a)
  %b = bitcast ptr %c to ptr
  ret ptr %b
}

; CHECK-LABEL: define i32 @caller_i32(ptr %a)
; CHECK-NEXT:    %c = musttail call i32 @callee_i32(ptr %a)
; CHECK-NEXT:    ret i32 %c

declare i32 @callee_i32(ptr %a)

define i32 @caller_i32(ptr %a) {
  %c = musttail call i32 @callee_i32(ptr %a)
  %b = bitcast i32 %c to i32
  ret i32 %b
}
