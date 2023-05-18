; RUN: llvm-as < %s | llvm-dis | grep bitcast
; RUN: verify-uselistorder %s

define i1 @main(i32 %X) {
  %res = bitcast i1 true to i1
  ret i1 %res
}
