; RUN: llc < %s -mtriple=arm64-eabi -verify-machineinstrs | FileCheck %s

; We used to not mark NZCV as being used in the continuation basic-block
; when lowering a 128-bit "select" to branches. This meant a subsequent use
; of the same flags gave an internal fault here.

declare void @foo(fp128)

define double @test_f128csel_flags(i32 %lhs, fp128 %a, fp128 %b, double %l, double %r) nounwind {
; CHECK: test_f128csel_flags

    %tst = icmp ne i32 %lhs, 42
    %val = select i1 %tst, fp128 %a, fp128 %b
; CHECK: cmp w0, #42
; CHECK: b.eq {{.?LBB0}}

    call void @foo(fp128 %val)
    %retval = select i1 %tst, double %l, double %r

    ; It's also reasonably important that the actual fcsel comes before the
    ; function call since bl may corrupt NZCV. We were doing the right thing anyway,
    ; but just as well test it while we're here.
; CHECK: fcsel {{d[0-9]+}}, {{d[0-9]+}}, {{d[0-9]+}}, ne
; CHECK: bl {{_?foo}}

    ret double %retval
}
