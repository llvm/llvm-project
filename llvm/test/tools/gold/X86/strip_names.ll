; RUN: llvm-as %s -o %t.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=save-temps \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o.0.2.internalize.bc -o - | FileCheck %s

; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o -o %t2.o
; RUN: llvm-dis %t2.o -o - | FileCheck ---check-prefix=NONAME %s

; CHECK: @GlobalValueName
; CHECK: @foo(i32 %in)
; CHECK: somelabel:
; CHECK:  %GV = load i32, i32* @GlobalValueName
; CHECK:  %add = add i32 %in, %GV
; CHECK:  ret i32 %add

; NONAME: @GlobalValueName
; NONAME: @foo(i32)
; NONAME-NOT: somelabel:
; NONAME:  %2 = load i32, i32* @GlobalValueName
; NONAME:  %3 = add i32 %0, %2
; NONAME:  ret i32 %3

@GlobalValueName = global i32 0

define i32 @foo(i32 %in) {
somelabel:
  %GV = load i32, i32* @GlobalValueName
  %add = add i32 %in, %GV
  ret i32 %add
}
