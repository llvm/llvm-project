; RUN: not llc -mtriple=aarch64 -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not llc -mtriple=aarch64 -global-isel -global-isel-abort=1 \
; RUN:   -filetype=null < %s 2>&1 | FileCheck %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash, with either
; instruction selector.

; CHECK: error: cannot handle direct memory outputs yet for constraint 'm'
define i64 @rm_output() {
  %r = call i64 asm "# $0", "=rm"()
  ret i64 %r
}
