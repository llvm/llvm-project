; RUN:  llc < %s -mtriple=x86_64-unknown-linux-gnu
; PR9601
; Previously we'd crash trying to put a 32-bit float into a constraint
; for a normal 'r' register.
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
entry:
  %0 = call float asm sideeffect "xchg $0, $1", "=r,*m,0,~{memory},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) undef, float 2.000000e+00) nounwind
  unreachable
}
