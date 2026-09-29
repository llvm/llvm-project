; RUN: not llc < %s -mtriple=i386-apple-darwin 2>&1 %t

; Previously, this would assert in an assert build, but crash in a release build.
; No FileCheck, just make sure we handle this gracefully.
define i64 @t1(ptr %p, i64 %val) #0 {
entry:
  %0 = tail call i64 asm sideeffect "xaddq $0, $1", "=q,*m,0,~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"(ptr %p, i64 %val)
  ret i64 %0
}
