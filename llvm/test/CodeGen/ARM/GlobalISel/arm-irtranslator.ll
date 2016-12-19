; RUN: llc -mtriple arm-unknown -global-isel -stop-after=irtranslator %s -o - | FileCheck %s

define void @test_void_return() {
; CHECK-LABEL: name: test_void_return
; CHECK: BX_RET 14, _
entry:
  ret void
}

define i32 @test_add(i32 %x, i32 %y) {
; CHECK-LABEL: name: test_add
; CHECK: liveins: %r0, %r1
; CHECK: [[VREGX:%[0-9]+]]{{.*}} = COPY %r0
; CHECK: [[VREGY:%[0-9]+]]{{.*}} = COPY %r1
; CHECK: [[SUM:%[0-9]+]]{{.*}} = G_ADD [[VREGX]], [[VREGY]]
; CHECK: %r0 = COPY [[SUM]]
; CHECK: BX_RET 14, _, implicit %r0
entry:
  %sum = add i32 %x, %y
  ret i32 %sum
}
