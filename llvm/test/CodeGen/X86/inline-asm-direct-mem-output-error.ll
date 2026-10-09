; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -O0 -filetype=null < %s 2>&1 \
; RUN:   | FileCheck --check-prefix=O0 --implicit-check-not=error: %s
; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=null < %s 2>&1 \
; RUN:   | FileCheck --check-prefix=O2 --implicit-check-not=error: %s

; A direct output is the asm's result, so there is no memory to write it to.
; Picking a memory constraint for one is an error, not a crash.

; "rm" picks memory, the most general constraint.
; O0: error: cannot handle direct memory outputs yet for constraint 'm'
; O2: error: cannot handle direct memory outputs yet for constraint 'm'
define i32 @rm_i32() {
  %r = call i32 asm "# $0", "=rm"()
  ret i32 %r
}

; The same for a value no 'r' register can hold.
; O0: error: cannot handle direct memory outputs yet for constraint 'm'
; O2: error: cannot handle direct memory outputs yet for constraint 'm'
define x86_fp80 @rm_x86_fp80() {
  %r = call x86_fp80 asm "# $0", "=rm"()
  ret x86_fp80 %r
}

; "=m" has no other choice.
; O0: error: cannot handle direct memory outputs yet for constraint 'm'
; O2: error: cannot handle direct memory outputs yet for constraint 'm'
define i32 @m_i32() {
  %r = call i32 asm "# $0", "=m"()
  ret i32 %r
}
