; RUN: opt -function-attrs -S %s | FileCheck %s

define void @mustprogress_readnone() mustprogress {
; CHECK:      Function Attrs: {{.*}} noreturn {{.*}} readnone willreturn
; CHECK-NEXT: define void @mustprogress_readnone()
;
entry:
  br label %while.body

while.body:
  br label %while.body
}

define i32 @mustprogress_load(i32* %ptr) mustprogress {
; CHECK:      Function Attrs: {{.*}} readonly willreturn
; CHECK-NEXT: define i32 @mustprogress_load(
;
entry:
  br label %while.body

while.body:
  %r = load i32, i32* %ptr
  br label %while.body
}

define void @mustprogress_store(i32* %ptr) mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK: define void @mustprogress_store(
;
entry:
  br label %while.body

while.body:
  store i32 0, i32* %ptr
  br label %while.body
}

declare void @unknown_fn()

define void @mustprogress_call_unknown_fn() mustprogress {
; CHECK-NOT: Function Attrs: {{.*}} willreturn
; CHECK:     define void @mustprogress_call_unknown_fn(
;
  call void @unknown_fn()
  ret void
}

define i32 @mustprogress_call_known_functions(i32* %ptr) mustprogress {
; CHECK:      Function Attrs: {{.*}} readonly willreturn
; CHECK-NEXT: define i32 @mustprogress_call_known_functions(
;
  call void @mustprogress_readnone()
  %r = call i32 @mustprogress_load(i32* %ptr)
  ret i32 %r
}

declare i32 @__gxx_personality_v0(...)

define i64 @mustprogress_mayunwind() mustprogress personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK:      Function Attrs: {{.*}} readnone willreturn
; CHECK-NEXT: define i64 @mustprogress_mayunwind(
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

; Function without loops or non-willreturn calls will return.
define void @willreturn_no_loop(i1 %c, i32* %p) {
; CHECK: Function Attrs: willreturn
; CHECK-NEXT: define void @willreturn_no_loop(
;
  br i1 %c, label %if, label %else

if:
  load atomic i32, i32* %p seq_cst, align 4
  call void @fn_willreturn()
  br label %end

else:
  store atomic i32 0, i32* %p seq_cst, align 4
  br label %end

end:
  ret void
}

; Calls a function that is not guaranteed to return, not willreturn.
define void @willreturn_non_returning_function(i1 %c, i32* %p) {
; CHECK-NOT: Function Attrs: {{.*}}willreturn
; CHECK: define void @willreturn_non_returning_function(
;
  call void @unknown_fn()
  ret void
}

; Infinite loop without mustprogress, will not return.
define void @willreturn_loop() {
; CHECK-NOT: Function Attrs: {{.*}}willreturn
; CHECK: define void @willreturn_loop(
;
  br label %loop

loop:
  br label %loop
}

; Finite loop. Could be willreturn but not detected.
; FIXME
define void @willreturn_finite_loop() {
; CHECK-NOT: Function Attrs: {{.*}}willreturn
; CHECK: define void @willreturn_finite_loop(
;
entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry], [ %i.inc, %loop ]
  %i.inc = add nuw i32 %i, 1
  %c = icmp ne i32 %i.inc, 100
  br i1 %c, label %loop, label %end

end:
  ret void
}

; Infinite recursion without mustprogress, will not return.
define void @willreturn_recursion() {
; CHECK-NOT: Function Attrs: {{.*}}willreturn
; CHECK: define void @willreturn_recursion(
;
  tail call void @willreturn_recursion()
  ret void
}

; Irreducible infinite loop, will not return.
define void @willreturn_irreducible(i1 %c) {
; CHECK-NOT: Function Attrs: {{.*}}willreturn
; CHECK: define void @willreturn_irreducible(
;
  br i1 %c, label %bb1, label %bb2

bb1:
  br label %bb2

bb2:
  br label %bb1
}

declare i64 @fn_noread() readnone
declare void @fn_willreturn() willreturn
