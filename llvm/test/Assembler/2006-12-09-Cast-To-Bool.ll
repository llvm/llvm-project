; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: bitcast
define i1 @main(i32 %X) {
  %res = bitcast i1 true to i1
  ret i1 %res
}
