; RUN: not llvm-as < %s 2>&1 | FileCheck %s

; CHECK: Attribute 'align' exceed the max size 2^14
define dso_local void @foo(i8* %p) {
entry:
  %p1 = bitcast i8* %p to <8 x float>*
  call void @bar(<8 x float>* noundef byval(<8 x float>) align 32768 %p1)
  ret void
}

declare dso_local void @bar(<8 x float>* %p)
