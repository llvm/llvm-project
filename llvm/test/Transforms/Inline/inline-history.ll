; RUN: opt -passes=inline -S < %s | FileCheck %s

; This will inline @f1 into @a, causing two new calls to @f2, which will get inlined for two calls to @f1.
; The inline history should stop recursive inlining here and put the inline history into metadata.

define internal void @f1(ptr %p) {
  call void %p(ptr @f1)
  ret void
}

define internal void @f2(ptr %p) {
  call void %p(ptr @f2)
  call void %p(ptr @f2)
  ret void
}

define void @b() {
; CHECK-LABEL: define void @b() {
; CHECK-NEXT:    call void @f1(ptr @f2), !inline_history [[HISTORY:![0-9]+]]
; CHECK-NEXT:    call void @f1(ptr @f2), !inline_history [[HISTORY:![0-9]+]]
; CHECK-NEXT:    ret void
;
  call void @a()
  ret void
}

define internal void @a() {
  call void @f1(ptr @f2)
  ret void
}

; CHECK: [[HISTORY]] = distinct !{null, ptr @f2, ptr @f1}
