; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK-LABEL: test
define i64 @test(i64 %a, i256 %b, i1 %c) {
  %u = zext i64 %a to i256
  %s = add i256 %u, 1
  %o = trunc i256 %s to i1
  %j = add i256 %s, 1
  %i = icmp ule i64 %a, 1
  %f = select i1 %o, i256 undef, i256 %j
  %d = select i1 %i, i256 %f, i256 1
  %e = add i256 %b, 1
  %n = select i1 %c, i256 %e, i256 %b
  %m = trunc i256 %n to i64
  %h = add i64 %m, 1
  %r = zext i64 %h to i256
  %v = lshr i256 %d, %r
  %t = trunc i256 %v to i1
  %q = shl i256 1, %r
  %p = and i256 %d, %q
  %w = icmp ule i256 %n, 1
  %y = select i1 %t, i256 undef, i256 %p
  %x = select i1 %w, i256 %y, i256 %d
  %z = trunc i256 %x to i64
  ret i64 %z
}
