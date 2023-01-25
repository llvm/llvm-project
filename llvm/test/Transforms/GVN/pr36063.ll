; RUN: opt < %s -passes=memcpyopt,mldst-motion,gvn -S | FileCheck %s

define void @foo(ptr %ret, i1 %x) {
  %a = alloca i8
  br i1 %x, label %yes, label %no

yes:                                              ; preds = %0
  store i8 5, ptr %a
  br label %out

no:                                               ; preds = %0
  store i8 5, ptr %a
  br label %out

out:                                              ; preds = %no, %yes
  %tmp = load i8, ptr %a
; CHECK-NOT: undef
  store i8 %tmp, ptr %ret
  ret void
}
