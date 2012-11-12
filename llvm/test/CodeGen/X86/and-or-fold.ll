; RUN: llc < %s -mtriple=i686-apple-darwin | FileCheck -check-prefix=DARWIN %s
; RUN: opt < %s -O2 | llc -mtriple=x86_64-apple-darwin | FileCheck -check-prefix=DARWIN-OPT %s

; The dag combiner should fold together (x&127)|(y&16711680) -> (x|y)&c1
; in this case.

define i32 @test1(i32 %x, i16 %y) {
  %tmp1 = zext i16 %y to i32
  %tmp2 = and i32 %tmp1, 127
  %tmp4 = shl i32 %x, 16
  %tmp5 = and i32 %tmp4, 16711680
  %tmp6 = or i32 %tmp2, %tmp5
  ret i32 %tmp6
; DARWIN: andl $16711807, %eax
}

; <rdar://problem/7529774> The optimizer shouldn't fold this into (and (or, C),  D)
; if (C & D) == 0
define i64 @test2(i64 %x) nounwind readnone ssp {
entry:
  %tmp1 = and i64 %x, 123127
  %tmp2 = or i64 %tmp1, 3
  ret i64 %tmp2
; DARWIN-OPT:       andq $123124
; DARWIN-OPT-NEXT:  leaq 3
}
