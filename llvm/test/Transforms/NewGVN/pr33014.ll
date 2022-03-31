; Make sure we don't end up in an infinite recursion in singleReachablePHIPath().
; RUN: opt < %s -passes=newgvn -S | FileCheck %s

@c = external global i64, align 8

; CHECK-LABEL: define void @tinkywinky() {
; CHECK: entry:
; CHECK-NEXT:   br i1 undef, label %l2, label %if.then
; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   br label %for.body
; CHECK: ph:                                               ; preds = %back, %ontrue
; CHECK-NEXT:   br label %for.body
; CHECK: for.body:                                         ; preds = %ph, %if.then
; CHECK-NEXT:   br i1 undef, label %ontrue, label %onfalse
; CHECK: onfalse:                                          ; preds = %for.body
; CHECK-NEXT:   %patatino = load i64, i64* @c
; CHECK-NEXT:   ret void
; CHECK: ontrue:                                           ; preds = %for.body
; CHECK-NEXT:   %dipsy = load i64, i64* @c
; CHECK-NEXT:   br label %ph
; CHECK: back:                                             ; preds = %l2
; CHECK-NEXT:   store i8 poison, i8* null
; CHECK-NEXT:   br label %ph
; CHECK: end:                                              ; preds = %l2
; CHECK-NEXT:   ret void
; CHECK: l2:                                               ; preds = %entry
; CHECK-NEXT:   br i1 false, label %back, label %end
; CHECK-NEXT: }

define void @tinkywinky() {
entry:
  br i1 undef, label %l2, label %if.then
if.then:
  br label %for.body
ph:
  br label %for.body
for.body:
  br i1 undef, label %ontrue, label %onfalse
onfalse:
  %patatino = load i64, i64* @c
  store i64 %patatino, i64* @c
  ret void
ontrue:
  %dipsy = load i64, i64* @c
  store i64 %dipsy, i64* @c
  br label %ph
back:
  br label %ph
end:
  ret void
l2:
  br i1 false, label %back, label %end
}
