; RUN: opt %loadNPMPolly "-passes=scop(polly-opt-isl,polly-codegen,polly-codegen)" -S < %s | FileCheck %s
; RUN: opt %loadNPMPolly "-passes=scop(polly-opt-isl,polly-codegen),scop(polly-codegen)" -S < %s | FileCheck %s
;
; llvm.org/PR34441
; Properly handle multiple -polly-scops/-polly-codegen in the same
; RegionPassManager. -polly-codegen must not reuse the -polly-ast analysis the
; was created for the first -polly-scops pass.
; The current solution is that only the first -polly-codegen is allowed to
; generate code, the second detects it is re-using an IslAst that belongs to a
; different ScopInfo.
;
; int a, b, c;
;
; int main () {
;  while (a++)
;    while (b) {
;        c = 0;
;        break;
;      }
;  return 0;
; }
;
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

@a = common global i32 0, align 4
@b = common global i32 0, align 4
@c = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval, align 4
  %.pre = load i32, ptr @a, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.end, %entry
  %0 = phi i32 [ %inc, %while.end ], [ %.pre, %entry ]
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr @a, align 4
  %tobool = icmp ne i32 %0, 0
  br i1 %tobool, label %while.body, label %while.end4

while.body:                                       ; preds = %while.cond
  %1 = load i32, ptr @b, align 4
  %tobool2 = icmp ne i32 %1, 0
  br i1 %tobool2, label %while.body3, label %while.end

while.body3:                                      ; preds = %while.body
  store i32 0, ptr @c, align 4
  br label %while.end

while.end:                                        ; preds = %while.body3, %while.body
  br label %while.cond

while.end4:                                       ; preds = %while.cond
  ret i32 0
}


; CHECK: polly.start:
; CHECK-NOT: polly.start:
