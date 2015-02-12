; RUN: llvm-as -o %t.bc %s
; RUN: ld -plugin %llvmshlibdir/LLVMgold.so -plugin-opt=emit-llvm \
; RUN:    --no-map-whole-files -r -o %t2.bc %t.bc
; RUN: llvm-dis < %t2.bc -o - | FileCheck %s

; CHECK: main
define i32 @main() {
  ret i32 0
}
