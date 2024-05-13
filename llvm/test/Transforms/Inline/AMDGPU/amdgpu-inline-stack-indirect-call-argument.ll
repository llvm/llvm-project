; REQUIRES: asserts
; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes=inline -inline-cost-full=true -inline-threshold=0 -inline-instr-cost=5 -inline-call-penalty=0 -debug-only=inline < %s 2>&1 | FileCheck %s

; CHECK:      Inlining (cost={{-+[0-9]+}}, threshold=330), Call:   call void @Dummy

define void @Wrapper(ptr nocapture nofree noundef readonly %func, i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0) {
entry:
  call void %func(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0)
  ret void
}

define internal void @Dummy(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0) {
entry:
  ret void
}

define void @Caller(i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0) minsize {
entry:
  call void @Wrapper(ptr noundef @Dummy, i64 noundef %a0, i64 noundef %b0, i64 noundef %c0, i64 noundef %d0, i64 noundef %e0, i64 noundef %f0, i64 noundef %g0, i64 noundef %h0, i64 noundef %i0, i64 noundef %j0, i64 noundef %k0, i64 noundef %l0, i64 noundef %m0, i64 noundef %n0, i64 noundef %o0, i64 noundef %p0, i64 noundef %q0)
  ret void
}
