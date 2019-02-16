; RUN: opt < %s -inline -inline-remark-attribute --inline-threshold=-2 -S | FileCheck %s

; Test that the inliner adds inline remark attributes to non-inlined callsites.

define void @foo() {
  call void @bar(i1 true)
  ret void
}

define void @bar(i1 %p) {
  br i1 %p, label %bb1, label %bb2

bb1:
  call void @foo()
  ret void

bb2:
  call void @bar(i1 true)
  ret void
}

;; Test 1 - Add different inline remarks to similar callsites.
define void @test1() {
; CHECK-LABEL: @test1
; CHECK-NEXT: call void @bar(i1 true) [[ATTR1:#[0-9]+]]
; CHECK-NEXT: call void @bar(i1 false) [[ATTR2:#[0-9]+]]
  call void @bar(i1 true)
  call void @bar(i1 false)
  ret void
}

define void @noop() {
  ret void
}

;; Test 2 - Printed InlineResult messages are followed by InlineCost.
define void @test2(i8*) {
; CHECK-LABEL: @test2
; CHECK-NEXT: call void @noop() [[ATTR3:#[0-9]+]] [ "CUSTOM_OPERAND_BUNDLE"() ]
; CHECK-NEXT: ret void
  call void @noop() ; extepected to be inlined
  call void @noop() [ "CUSTOM_OPERAND_BUNDLE"() ] ; cannot be inlined because of unsupported operand bundle
  ret void
}

; CHECK: attributes [[ATTR1]] = { "inline-remark"="(cost=-5, threshold=-6)" }
; CHECK: attributes [[ATTR2]] = { "inline-remark"="(cost=never): recursive" }
; CHECK: attributes [[ATTR3]] = { "inline-remark"="unsupported operand bundle; (cost={{.*}}, threshold={{.*}})" }
