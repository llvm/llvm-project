# See PR48742.
    .text
    .p2align 4
foo:
    movq    %rdi, %rax
    .p2align 4,,10
    .p2align 3
L1:
    movzbl  (%rax), %edx
    cmpb    $10, %dl
    je L4
L2:
    cmpb    $100, %dl
    je  L5
    cmpb    $200, %dl
    je  L5
    cmpb    $300, %dl
    jne L5
    .p2align 4,,10
    .p2align 3
L3:
    movq    %rax, %rdx
    incq    %rax
    cmpb    $30, (%rax)
    jne L3
    leaq    2(%rdx), %rax
    movzbl  (%rax), %edx
    cmpb    $90, %dl
    jne L2
    .p2align 4,,10
    .p2align 3
L4:
    cmpb    $99, 4(%rax)
    je L7
L5:
    incq    %rax
    jmp L1
    .p2align 4,,10
    .p2align 3
L6:
    ret
L7:
    ret
