; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

%Ty = type { i32, i32 }

; Validate that the argument and return value are both dead
; CHECK-LABEL: define internal void @test1()

define internal ptr @test1(ptr %this) {
  ret ptr %this
}

; do not keep alive the return value of a function with a dead 'returned' argument
; CHECK-LABEL: define internal void @test2()

define internal ptr @test2(ptr returned %this) {
  ret ptr %this
}

; dummy to keep 'this' alive
@dummy = global ptr null 

; Validate that return value is dead
; CHECK-LABEL: define internal void @test3(ptr %this)

define internal ptr @test3(ptr %this) {
  store volatile ptr %this, ptr @dummy
  ret ptr %this
}

; keep alive return value of a function if the 'returned' argument is live
; CHECK-LABEL: define internal ptr @test4(ptr returned %this)

define internal ptr @test4(ptr returned %this) {
  store volatile ptr %this, ptr @dummy
  ret ptr %this
}

; don't do this if 'returned' is on the call site...
; CHECK-LABEL: define internal void @test5(ptr %this)

define internal ptr @test5(ptr %this) {
  store volatile ptr %this, ptr @dummy
  ret ptr %this
}

; Drop all these attributes
; CHECK-LABEL: define internal void @test6
define internal align 8 dereferenceable_or_null(2) noundef noalias ptr @test6() {
  ret ptr null
}

define ptr @caller(ptr %this) {
  %1 = call ptr @test1(ptr %this)
  %2 = call ptr @test2(ptr %this)
  %3 = call ptr @test3(ptr %this)
  %4 = call ptr @test4(ptr %this)
; ...instead, drop 'returned' form the call site
; CHECK: call void @test5(ptr %this)
  %5 = call ptr @test5(ptr returned %this)
  %6 = call ptr @test6()
  ret ptr %this
}
