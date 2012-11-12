; Test CFG simplify removal of branch instructions.
;
; RUN: opt < %s -simplifycfg -S | FileCheck %s

define void @test1() {
        br label %1
        ret void
; CHECK: @test1
; CHECK-NEXT: ret void
}

define void @test2() {
        ret void
        ret void
; CHECK: @test2
; CHECK-NEXT: ret void
; CHECK-NEXT: }
}

define void @test3(i1 %T) {
        br i1 %T, label %1, label %1
        ret void
; CHECK: @test3
; CHECK-NEXT: ret void
}


; PR5795
define void @test5(i32 %A) {
  switch i32 %A, label %return [
    i32 2, label %1
    i32 10, label %2
  ]

  ret void

  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: @test5
; CHECK-NEXT: ret void
}
