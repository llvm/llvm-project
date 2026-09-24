; RUN: opt -passes=inline -S < %s | FileCheck %s

declare ptr @get()
declare ptr @foo1()
declare ptr @foo2()

; Don't need to track inline history for non-mutually recursive calls.

define internal void @x() noinline {
  ret void
}

define internal void @y() {
  call void @x()
  ret void
}

define void @z() {
; CHECK-LABEL: define void @z() {
; CHECK-NEXT:    call void @x(){{$}}
; CHECK-NEXT:    ret void
  call void @y()
  ret void
}

; Indirect calls may be devirtualized, they need inline history tracking. In
; this case, @s is the history, but it gets deleted so the inline history
; metadata becomes null. This is fine since we cannot infinitely inline through
; @s anymore as it doesn't exist.

define internal void @s() {
  %p = call ptr @get()
  call void %p()
  ret void
}

define void @t() {
; CHECK-LABEL: define void @t() {
; CHECK-NEXT:    %p.i = call ptr @get()
; CHECK-NEXT:    call void %p.i(), !inline_history [[HISTORY_T:![0-9]+]]
; CHECK-NEXT:    ret void
  call void @s()
  ret void
}

; This will inline @f1 into @a, causing two new calls to @f2, which will get inlined for two calls to @f1.
; The inline history should stop recursive inlining here.

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
; CHECK-NEXT:    call void @f1(ptr @f2), !inline_history [[HISTORY_B:![0-9]+]]
; CHECK-NEXT:    call void @f1(ptr @f2), !inline_history [[HISTORY_B:![0-9]+]]
; CHECK-NEXT:    ret void
;
  call void @a()
  ret void
}

define internal void @a() {
  call void @f1(ptr @f2)
  ret void
}

; Check that the inline history is
; {callee, processed call's inline_history, just-inlined call's inline_history}

define void @m(ptr %p) noinline {
  ret void
}

define void @n() {
  call void @m(ptr @n2), !inline_history !{ptr @foo1}
  ret void
}

define void @n2() {
  call void @m(ptr @n)
  ret void
}

define void @o() {
; CHECK-LABEL: define void @o() {
; CHECK-NEXT:    call void @m(ptr @n2), !inline_history [[HISTORY_O:![0-9]+]]
; CHECK-NEXT:    ret void
  call void @n(), !inline_history !{ptr @foo2}
  ret void
}

; CHECK-DAG: [[HISTORY_T]] = distinct !{null}
; CHECK-DAG: [[HISTORY_B]] = !{ptr @f2, ptr @f1}
; CHECK-DAG: [[HISTORY_O]] = !{ptr @n, ptr @foo2, ptr @foo1}
