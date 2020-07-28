; RUN: opt < %s -function-attrs -S | FileCheck %s

; TEST 1
; CHECK: Function Attrs: norecurse nounwind readnone
; CHECK-NEXT: define i32 @foo1()
define i32 @foo1() {
  ret i32 1
}

; TEST 2
; CHECK: Function Attrs: nounwind readnone
; CHECK-NEXT: define i32 @scc1_foo()
define i32 @scc1_foo() {
  %1 = call i32 @scc1_bar()
  ret i32 1
}


; TEST 3
; CHECK: Function Attrs: nounwind readnone
; CHECK-NEXT: define i32 @scc1_bar()
define i32 @scc1_bar() {
  %1 = call i32 @scc1_foo()
  ret i32 1
}

; CHECK: declare i32 @non_nounwind()
declare i32 @non_nounwind()

; TEST 4
; CHECK: define void @call_non_nounwind() {
define void @call_non_nounwind(){
    tail call i32 @non_nounwind()
    ret void
}

; TEST 5 - throw
; int maybe_throw(bool canThrow) {
;   if (canThrow)
;     throw;
;   else
;     return -1;
; }

; CHECK: define i32 @maybe_throw(i1 zeroext %0)
define i32 @maybe_throw(i1 zeroext %0) {
  br i1 %0, label %2, label %3

2:                                                ; preds = %1
  tail call void @__cxa_rethrow() #1
  unreachable

3:                                                ; preds = %1
  ret i32 -1
}

declare void @__cxa_rethrow()

; TEST 6 - catch
; int catch_thing() {
;   try {
;       int a = doThing(true);
;   }
;   catch(...) { return -1; }
;   return 1;
; }

; CHECK: define i32 @catch_thing()
define i32 @catch_thing() personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
  invoke void @__cxa_rethrow() #1
          to label %1 unwind label %2

1:                                                ; preds = %0
  unreachable

2:                                                ; preds = %0
  %3 = landingpad { i8*, i32 }
          catch i8* null
  %4 = extractvalue { i8*, i32 } %3, 0
  %5 = tail call i8* @__cxa_begin_catch(i8* %4) #2
  tail call void @__cxa_end_catch()
  ret i32 -1
}

define i32 @catch_thing_user() {
  %catch_thing_call = call i32 @catch_thing()
  ret i32 %catch_thing_call
}


declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()
