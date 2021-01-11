; RUN: opt -function-attrs -S %s | FileCheck %s

; TODO
define void @mustprogress_readnone() mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK:     define void @mustprogress_readnone()
;
entry:
  br label %while.body

while.body:
  br label %while.body
}

; TODO
define i32 @mustprogress_load(i32* %ptr) mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK:     define i32 @mustprogress_load(
;
entry:
  %r = load i32, i32* %ptr
  ret i32 %r
}

define void @mustprogress_store(i32* %ptr) mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK: define void @mustprogress_store(
;
entry:
  store i32 0, i32* %ptr
  ret void
}

declare void @unknown_fn()

define void @mustprogress_call_unknown_fn() mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK: define void @mustprogress_call_unknown_fn(
;
  call void @unknown_fn()
  ret void
}

; TODO
define i32 @mustprogress_call_known_functions(i32* %ptr) mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK:     define i32 @mustprogress_call_known_functions(
;
  call void @mustprogress_readnone()
  %r = call i32 @mustprogress_load(i32* %ptr)
  ret i32 %r
}

declare i32 @__gxx_personality_v0(...)

; TODO
define i64 @mustprogress_mayunwind() mustprogress personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK:     define i64 @mustprogress_mayunwind(
;
  %a = invoke i64 @fn_noread()
          to label %A unwind label %B
A:
  ret i64 10

B:
  %val = landingpad { i8*, i32 }
           catch i8* null
  ret i64 0
}

declare i64 @fn_noread() readnone
