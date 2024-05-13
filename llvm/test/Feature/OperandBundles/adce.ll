; RUN: opt -S -passes=adce < %s | FileCheck %s

; While it is normally okay to DCE out calls to @readonly_function and
; @readnone_function, we cannot do that if they're carrying operand
; bundles since the presence of unknown operand bundles implies
; arbitrary memory effects.

declare void @readonly_function() readonly nounwind willreturn
declare void @readnone_function() readnone nounwind willreturn

define void @test0() {
; CHECK-LABEL: @test0(
 entry:
  call void @readonly_function() [ "tag"() ]
; CHECK: call void @readonly_function
  ret void
}

define void @test1() {
; CHECK-LABEL: @test1(
 entry:
  call void @readnone_function() [ "tag"() ]
; CHECK: call void @readnone_function
  ret void
}

define void @test2() {
; CHECK-LABEL: @test2(
 entry:
; CHECK-NOT: @readonly_function(
  call void @readonly_function() readonly [ "tag"() ]
  ret void
}

define void @test3() {
; CHECK-LABEL: @test3(
 entry:
; CHECK-NOT: @readnone_function(
  call void @readnone_function() readnone [ "tag"() ]
  ret void
}

define void @test4() {
; CHECK-LABEL: @test4(
 entry:
; CHECK-NOT: @readonly_function()
  call void @readonly_function() [ "deopt"() ]
  ret void
}
