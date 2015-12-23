; RUN: opt < %s -globals-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

@a = internal global i32 0, align 4
@b = internal global i32 0, align 4

define void @g(i32* %p, void (i32*)* nocapture %ptr) {
entry:
  tail call void %ptr(i32* %p) #1
  ret void
}

; CHECK-LABEL: Function: f
; CHECK: MayAlias: i32* %p, i32* @a
; CHECK: MayAlias: i32* %q, i32* @a
define i32 @f(i32 %n, i32* nocapture readonly %p, i32* nocapture %q, void (i32*)* nocapture %ptr) {
entry:
  tail call void @g(i32* nonnull @a, void (i32*)* %ptr)
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 0
  %z1 = load i32, i32* %arrayidx, align 4
  %z2 = load i32, i32* %q, align 4
  %add = add nsw i32 %z2, %z1
  store i32 %add, i32* %q, align 4
  ret i32 4
}

define void @g2(i32* nocapture %p, void (i32*)* nocapture %ptr) {
entry:
  tail call void %ptr(i32* %p) #1
  ret void
}

; CHECK-LABEL: Function: f2
; CHECK: NoAlias: i32* %p, i32* @b
; CHECK: NoAlias: i32* %q, i32* @b
define i32 @f2(i32 %n, i32* nocapture readonly %p, i32* nocapture %q, void (i32*)* nocapture %ptr) {
entry:
  tail call void @g2(i32* nonnull @b, void (i32*)* %ptr)
  %arrayidx = getelementptr inbounds i32, i32* %p, i64 0
  %z1 = load i32, i32* %arrayidx, align 4
  %z2 = load i32, i32* %q, align 4
  %add = add nsw i32 %z2, %z1
  store i32 %add, i32* %q, align 4
  ret i32 4
}

declare void @g3()

; CHECK-LABEL: Function: f3
; CHECK: NoAlias: i32* %p, i32* @b
define void @f3(i32* nocapture readonly %p) {
entry:
  tail call void @g3() [ "deopt"(i32* @b, i32 *%p) ]
  unreachable
}
