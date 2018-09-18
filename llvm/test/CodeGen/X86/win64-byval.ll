; RUN: llc -mtriple x86_64-w64-mingw32 %s -o - | FileCheck %s

declare void @foo({ float, double }* byval)
@G = external constant { float, double }

define void @bar()
{
; Make sure we're creating a temporary stack slot, rather than just passing
; the pointer through unmodified.
; CHECK-LABEL: @bar
; CHECK: movq    .refptr.G(%rip), %rax
; CHECK: movq    (%rax), %rcx
; CHECK: movq    8(%rax), %rax
; CHECK: movq    %rax, 40(%rsp)
; CHECK: movq    %rcx, 32(%rsp)
; CHECK: leaq    32(%rsp), %rcx
    call void @foo({ float, double }* byval @G)
    ret void
}

define void @baz({ float, double }* byval %arg)
{
; On Win64 the byval is effectively ignored on declarations, since we do
; pass a real pointer in registers. However, by our semantics if we pass
; the pointer on to another byval function, we do need to make a copy.
; CHECK-LABEL: @baz
; CHECK: movq	(%rcx), %rax
; CHECK: movq	8(%rcx), %rcx
; CHECK: movq	%rcx, 40(%rsp)
; CHECK: movq	%rax, 32(%rsp)
; CHECK: leaq	32(%rsp), %rcx
    call void @foo({ float, double }* byval %arg)
    ret void
}
