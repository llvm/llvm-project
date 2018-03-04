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
