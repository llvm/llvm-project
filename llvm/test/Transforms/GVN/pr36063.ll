; RUN: opt < %s -passes=memcpyopt,mldst-motion,gvn -S | FileCheck %s

define void @foo(i8* %ret, i1 %x) {
  %a = alloca i8
  br i1 %x, label %yes, label %no

yes:                                              ; preds = %0
  %gepa = getelementptr i8, i8* %a, i64 0
  store i8 5, i8* %gepa
  br label %out

no:                                               ; preds = %0
  %gepb = getelementptr i8, i8* %a, i64 0
  store i8 5, i8* %gepb
  br label %out

out:                                              ; preds = %no, %yes
  %tmp = load i8, i8* %a
; CHECK-NOT: undef
  store i8 %tmp, i8* %ret
  ret void
}
