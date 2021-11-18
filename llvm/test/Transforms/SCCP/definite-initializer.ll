; RUN: opt -S -passes=ipsccp < %s | FileCheck %s
@d = internal externally_initialized global i32 0, section ".openbsd.randomdata", align 4

; CHECK-LABEL: @test1(
define i32 @test1() {
entry:
  %load = load i32, i32* @d, align 4
  ret i32 %load
; CHECK: %[[load:.*]] = load i32, i32* @d, align 4
; CHECK: ret i32 %[[load]]
}
