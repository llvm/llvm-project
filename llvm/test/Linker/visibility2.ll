; This file is used by visibility1.ll, so it doesn't actually do anything itself
;
; RUN: true

; Variables
@v1 = weak hidden global i32 0
@v2 = weak protected global i32 0
@v3 = weak hidden global i32 0

; Aliases
@a1 = hidden alias weak i32* @v1
@a2 = protected alias weak i32* @v2
@a3 = hidden alias weak i32* @v3

; Functions
define weak hidden void @f1() {
entry:
  ret void
}
define weak protected void @f2() {
entry:
  ret void
}
define weak hidden void @f3() {
entry:
  ret void
}
