; RUN: opt < %s -basic-aa -globals-aa -S -licm | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@b = common global i32 0, align 4
@c = internal global i32 0, align 4
@d = common global i32 0, align 4
@e = common global i32* null, align 4

define void @foo(i32* %P) noinline {
; CHECK: define void @foo
  %loadp = load i32, i32* %P, align 4
  store i32 %loadp, i32* @d, align 4
  ret void
}

define void @bar() noinline {
; CHECK: define void @bar
  %loadp = load i32, i32* @d, align 4
  store i32 %loadp, i32* @c, align 4
  ret void
}

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %c = alloca [1 x i32], align 4
  store i32 0, i32* %retval, align 4
  call void @bar()
  store i32 0, i32* @b, align 4
  br label %for.cond
  ;; Check that @c is LICM'ed out.
; CHECK: load i32, i32* @c
for.cond:                                         ; preds = %for.inc, %entry
; CHECK-LABEL: for.cond:
; CHECK: load i32, i32* @b
  %a1 = load i32, i32* @b, align 4
  %aa2 = load i32, i32* @c, align 4
  %add = add nsw i32 %a1, %aa2
  %p1 = load i32*, i32** @e, align 4
  call void @foo(i32* %p1)
  %cmp = icmp slt i32 %add, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %a2 = load i32, i32* @b, align 4
  %idxprom = sext i32 %a2 to i64
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %a3 = load i32, i32* @b, align 4
  %inc = add nsw i32 %a3, 1
  store i32 %inc, i32* @b, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 0
}

; Function Attrs: nounwind argmemonly
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i1) nounwind argmemonly

; Function Attrs: noreturn nounwind
declare void @abort() noreturn nounwind
