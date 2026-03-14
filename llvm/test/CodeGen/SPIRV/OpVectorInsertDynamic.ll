;; uint8 foo(uint8 c, unsigned i) {
;;   c[i] = 42;
;;   return c;
;; }

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: %[[#TypeInt:]] = OpTypeInt 32
; CHECK: %[[#TypeVector:]] = OpTypeVector %[[#TypeInt]] 8
; CHECK: %[[#]] = OpVectorInsertDynamic %[[#TypeVector]]

define spir_func <8 x i32> @foo(<8 x i32> %c, i32 %i) local_unnamed_addr {
entry:
  %vecins = insertelement <8 x i32> %c, i32 42, i32 %i
  ret <8 x i32> %vecins
}
