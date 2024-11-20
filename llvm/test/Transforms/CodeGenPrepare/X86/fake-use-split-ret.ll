; RUN: opt -mtriple=x86_64-unknown-unknown -S -codegenprepare <%s -o - | FileCheck %s
;
; Ensure return instruction splitting ignores fake uses.
;
; IR Generated with clang -O2 -S -emit-llvm -fextend-lifetimes test.cpp
;
;// test.cpp
;extern int bar(int);
;
;int foo2(int i)
;{
;  --i;
;  if (i <= 0)
;    return -1;
;  return bar(i);
;}

declare i32 @_Z3bari(i32) local_unnamed_addr

define i32 @_Z4foo2i(i32 %i) local_unnamed_addr optdebug {
entry:
  %dec = add nsw i32 %i, -1
  %cmp = icmp slt i32 %i, 2
  br i1 %cmp, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  %call = tail call i32 @_Z3bari(i32 %dec)
; CHECK: ret i32 %call
  br label %cleanup

cleanup:                                          ; preds = %entry, %if.end
; CHECK: cleanup:
  %retval.0 = phi i32 [ %call, %if.end ], [ -1, %entry ]
  tail call void (...) @llvm.fake.use(i32 %dec)
; CHECK: ret i32 -1
  ret i32 %retval.0
}
