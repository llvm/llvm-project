; RUN: opt < %s -passes=deadargelim -S | FileCheck %s

%Ty = type { i32, i32 }

; sanity check that the argument and return value are both dead
; CHECK-LABEL: define internal void @test1()

define internal %Ty* @test1(%Ty* %this) {
  ret %Ty* %this
}

; do not keep alive the return value of a function with a dead 'returned' argument
; CHECK-LABEL: define internal void @test2()

define internal %Ty* @test2(%Ty* returned %this) {
  ret %Ty* %this
}

; dummy to keep 'this' alive
@dummy = global %Ty* null 

; sanity check that return value is dead
; CHECK-LABEL: define internal void @test3(%Ty* %this)

define internal %Ty* @test3(%Ty* %this) {
  store volatile %Ty* %this, %Ty** @dummy
  ret %Ty* %this
}

; keep alive return value of a function if the 'returned' argument is live
; CHECK-LABEL: define internal %Ty* @test4(%Ty* returned %this)

define internal %Ty* @test4(%Ty* returned %this) {
  store volatile %Ty* %this, %Ty** @dummy
  ret %Ty* %this
}

; don't do this if 'returned' is on the call site...
; CHECK-LABEL: define internal void @test5(%Ty* %this)

define internal %Ty* @test5(%Ty* %this) {
  store volatile %Ty* %this, %Ty** @dummy
  ret %Ty* %this
}

; Drop all these attributes
; CHECK-LABEL: define internal void @test6
define internal align 8 dereferenceable_or_null(2) noundef noalias i8* @test6() {
  ret i8* null
}

define %Ty* @caller(%Ty* %this) {
  %1 = call %Ty* @test1(%Ty* %this)
  %2 = call %Ty* @test2(%Ty* %this)
  %3 = call %Ty* @test3(%Ty* %this)
  %4 = call %Ty* @test4(%Ty* %this)
; ...instead, drop 'returned' form the call site
; CHECK: call void @test5(%Ty* %this)
  %5 = call %Ty* @test5(%Ty* returned %this)
  %6 = call i8* @test6()
  ret %Ty* %this
}
