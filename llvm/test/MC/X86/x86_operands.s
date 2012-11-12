// RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

# Immediates
# CHECK: addl $1, %eax
        addl $1, %eax
# CHECK: addl $3, %eax
        addl $(1+2), %eax
# CHECK: addl $a, %eax
        addl $a, %eax
# CHECK: addl $3, %eax
        addl $1 + 2, %eax
        
# Disambiguation

# CHECK: addl $1, 8
        addl $1, 4+4
# CHECK: addl $1, 8
        addl $1, (4+4)
# CHECK: addl $1, 8(%eax)
        addl $1, 4+4(%eax)
# CHECK: addl $1, 8(%eax)
        addl $1, (4+4)(%eax)
# CHECK: addl $1, 8(%eax)
        addl $1, 8(%eax)
# CHECK: addl $1, (%eax)
        addl $1, (%eax)
# CHECK: addl $1, 8(,%eax)
        addl $1, (4+4)(,%eax)
        
# Indirect Memory Operands
# CHECK: addl $1, 1(%eax)
        addl $1, 1(%eax)
# CHECK: addl $1, 1(%eax,%ebx)
        addl $1, 1(%eax,%ebx)
# CHECK: addl $1, 1(%eax,%ebx)
        addl $1, 1(%eax,%ebx,)
# CHECK: addl $1, 1(%eax,%ebx,4)
        addl $1, 1(%eax,%ebx,4)
# CHECK: addl $1, 1(,%ebx)
        addl $1, 1(,%ebx)
# CHECK: addl $1, 1(,%ebx)
        addl $1, 1(,%ebx,)
# CHECK: addl $1, 1(,%ebx,4)
        addl $1, 1(,%ebx,4)
# CHECK: addl $1, 1(,%ebx,4)
        addl $1, 1(,%ebx,(2+2))

# '*'
# CHECK: calll a
        call a
# CHECK: calll *%eax
        call *%eax
# CHECK: calll *4(%eax)
        call *4(%eax)

# CHECK: movl	%gs:8, %eax
movl %gs:8, %eax

