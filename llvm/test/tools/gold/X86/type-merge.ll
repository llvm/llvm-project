; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/type-merge.ll -o %t2.o
; RUN: %gold -plugin %llvmshlibdir/LLVMgold.so \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t.o %t2.o -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

define void @foo()  {
  call void @bar(i8* null)
  ret void
}
declare void @bar(i8*)

; CHECK:      define void @foo() {
; CHECK-NEXT:   call void @bar(i8* null)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: declare void @bar(i8*)

; CHECK:      define void @zed() {
; CHECK-NEXT:   call void bitcast (void (i8*)* @bar to void ()*)()
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
