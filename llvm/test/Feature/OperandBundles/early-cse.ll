; RUN: opt -S -early-cse -earlycse-debug-hash < %s | FileCheck %s

; While it is normally okay to do memory optimizations over calls to
; @readonly_function and @readnone_function, we cannot do that if
; they're carrying unknown operand bundles since the presence of
; unknown operand bundles implies arbitrary memory effects.

declare void @readonly_function() readonly nounwind
declare void @readnone_function() readnone nounwind

define i32 @test0(i32* %x) {
; CHECK-LABEL: @test0(
 entry:
  store i32 100, i32* %x
; CHECK: store i32 100, i32* %x
  call void @readonly_function() [ "tag"() ]
; CHECK: call void @readonly_function()

  %v = load i32, i32* %x
; CHECK: %v = load i32, i32* %x
; CHECK: ret i32 %v
  ret i32 %v
}

define i32 @test1(i32* %x) {
; CHECK: @test1(
 entry:
  store i32 100, i32* %x
; CHECK: store i32 100, i32* %x
  call void @readonly_function() readonly [ "tag"() ]
; CHECK-NOT: call void @readonly_function
  %v = load i32, i32* %x
  ret i32 %v
; CHECK: ret i32 100
}

define i32 @test3(i32* %x) {
; CHECK-LABEL: @test3(
 entry:
  store i32 100, i32* %x
; CHECK: store i32 100, i32* %x
  call void @readonly_function()
; CHECK-NOT: call void @readonly_function
  %v = load i32, i32* %x
  ret i32 %v
; CHECK: ret i32 100
}

define void @test4(i32* %x) {
; CHECK-LABEL: @test4(
 entry:
  store i32 100, i32* %x
; CHECK: store i32 100, i32* %x
  call void @readnone_function() [ "tag"() ]
; CHECK: call void @readnone_function
  store i32 200, i32* %x
; CHECK: store i32 200, i32* %x
  ret void
}

define void @test5(i32* %x) {
; CHECK-LABEL: @test5(
 entry:
  store i32 100, i32* %x
; CHECK-NOT: store i32 100, i32* %x
; CHECK-NOT: call void @readnone_function
  call void @readnone_function() readnone [ "tag"() ]
  store i32 200, i32* %x
; CHECK: store i32 200, i32* %x
  ret void
}

define void @test6(i32* %x) {
; The "deopt" operand bundle does not make the call to
; @readonly_function read-write; and so the nounwind readonly call can
; be deleted.

; CHECK-LABEL: @test6(
 entry:

; CHECK-NEXT: entry:
; CHECK-NEXT:  store i32 200, i32* %x
; CHECK-NEXT:  ret void

  store i32 100, i32* %x
  call void @readonly_function() [ "deopt"() ]
  store i32 200, i32* %x
  ret void
}
