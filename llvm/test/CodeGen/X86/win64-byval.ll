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

declare void @foo2({ float, double }* byval, { float, double }* byval, { float, double }* byval, { float, double }* byval, { float, double }* byval, i64 %f)
@data = external constant { float, double }

define void @test() {
; CHECK-LABEL: @test
; CHECK:      movq    (%rax), %rcx
; CHECK-NEXT: movq    8(%rax), %rax
; CHECK-NEXT: movq    %rax, 120(%rsp)
; CHECK-NEXT: movq    %rcx, 112(%rsp)
; CHECK-NEXT: movq    %rcx, 96(%rsp)
; CHECK-NEXT: movq    %rax, 104(%rsp)
; CHECK-NEXT: movq    %rcx, 80(%rsp)
; CHECK-NEXT: movq    %rax, 88(%rsp)
; CHECK-NEXT: movq    %rcx, 64(%rsp)
; CHECK-NEXT: movq    %rax, 72(%rsp)
; CHECK-NEXT: movq    %rax, 56(%rsp)
; CHECK-NEXT: movq    %rcx, 48(%rsp)
; CHECK-NEXT: leaq    48(%rsp), %rax
; CHECK-NEXT: movq    %rax, 32(%rsp)
; CHECK-NEXT: movq    $10, 40(%rsp)
; CHECK-NEXT: leaq    112(%rsp), %rcx
; CHECK-NEXT: leaq    96(%rsp), %rdx
; CHECK-NEXT: leaq    80(%rsp), %r8
; CHECK-NEXT: leaq    64(%rsp), %r9
  call void @foo2({ float, double }* byval @G, { float, double }* byval @G, { float, double }* byval @G, { float, double }* byval @G, { float, double }* byval @G, i64 10)
  ret void
}
