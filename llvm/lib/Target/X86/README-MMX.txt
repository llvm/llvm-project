//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend: MMX-specific stuff.
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//

This:

#include <mmintrin.h>

__v2si qux(int A) {
  return (__v2si){ 0, A };
}

is compiled into:

_qux:
        subl $28, %esp
        movl 32(%esp), %eax
        movd %eax, %mm0
        movq %mm0, (%esp)
        movl (%esp), %eax
        movl %eax, 20(%esp)
        movq %mm0, 8(%esp)
        movl 12(%esp), %eax
        movl %eax, 16(%esp)
        movq 16(%esp), %mm0
        addl $28, %esp
        ret

Yuck!

GCC gives us:

_qux:
        subl    $12, %esp
        movl    16(%esp), %eax
        movl    20(%esp), %edx
        movl    $0, (%eax)
        movl    %edx, 4(%eax)
        addl    $12, %esp
        ret     $4

//===---------------------------------------------------------------------===//

We generate crappy code for this:

__m64 t() {
  return _mm_cvtsi32_si64(1);
}

_t:
	subl	$12, %esp
	movl	$1, %eax
	movd	%eax, %mm0
	movq	%mm0, (%esp)
	movl	(%esp), %eax
	movl	4(%esp), %edx
	addl	$12, %esp
	ret

The extra stack traffic is covered in the previous entry. But the other reason
is we are not smart about materializing constants in MMX registers. With -m64

	movl	$1, %eax
	movd	%eax, %mm0
	movd	%mm0, %rax
	ret

We should be using a constantpool load instead:
	movq	LC0(%rip), %rax
