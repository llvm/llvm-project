; RUN: opt < %s -aa-pipeline=basic-aa,globals-aa -passes='function(instcombine),require<globals-aa>,function(invalidate<aa>,loop-mssa(licm))' -S | FileCheck %s

; Make sure -globals-aa ignores dead uses of globals.

@a = internal global i32 0, align 4
@c = common global i32 0, align 4

; Function Attrs: nounwind
define i32 @g() {
; Make sure the load of @a is hoisted.
; CHECK-LABEL: define i32 @g()
; CHECK: entry:
; CHECK-NEXT: load i32, i32* @a, align 4
; CHECK-NEXT: br label %for.cond
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %sum.0 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %cmp = icmp slt i32 %i.0, 1000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %0 = load i32, i32* @a, align 4
  %add = add nsw i32 %sum.0, %0
  call void @f()
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret i32 %sum.0
}

; Function Attrs: nounwind
define internal void @f() {
entry:
  %tobool = icmp ne i32 0, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 ptrtoint (i32* @a to i32), i32* @c, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  %0 = load i32, i32* @c, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @c, align 4
  ret void
}

