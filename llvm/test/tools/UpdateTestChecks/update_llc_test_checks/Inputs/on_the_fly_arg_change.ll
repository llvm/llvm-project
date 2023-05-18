; RUN: llc < %s -mtriple=i686-unknown-unknown | FileCheck %s

declare void @foo()

define dso_local i64 @check_lines_1() {
  ret i64 1
}

; UTC_ARGS: --disable

define dso_local i64 @no_check_lines() {
; A check line that would not be auto-generated (should not be removed!).
; CHECK: manual check line
  ret i64 2
}

; UTC_ARGS: --enable --no_x86_scrub_rip

define dso_local i64 @check_lines_2() {
  %result = call i64 @no_check_lines()
  ret i64 %result
}
