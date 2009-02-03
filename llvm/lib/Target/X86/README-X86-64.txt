//===- README_X86_64.txt - Notes for X86-64 code gen ----------------------===//

Implement different PIC models? Right now we only support Mac OS X with small
PIC code model.

//===---------------------------------------------------------------------===//

Make use of "Red Zone".

//===---------------------------------------------------------------------===//

For this:

extern void xx(void);
void bar(void) {
  xx();
}

gcc compiles to:

.globl _bar
_bar:
	jmp	_xx

We need to do the tailcall optimization as well.

//===---------------------------------------------------------------------===//

AMD64 Optimization Manual 8.2 has some nice information about optimizing integer
multiplication by a constant. How much of it applies to Intel's X86-64
implementation? There are definite trade-offs to consider: latency vs. register
pressure vs. code size.

//===---------------------------------------------------------------------===//

Are we better off using branches instead of cmove to implement FP to
unsigned i64?

_conv:
	ucomiss	LC0(%rip), %xmm0
	cvttss2siq	%xmm0, %rdx
	jb	L3
	subss	LC0(%rip), %xmm0
	movabsq	$-9223372036854775808, %rax
	cvttss2siq	%xmm0, %rdx
	xorq	%rax, %rdx
L3:
	movq	%rdx, %rax
	ret

instead of

_conv:
	movss LCPI1_0(%rip), %xmm1
	cvttss2siq %xmm0, %rcx
	movaps %xmm0, %xmm2
	subss %xmm1, %xmm2
	cvttss2siq %xmm2, %rax
	movabsq $-9223372036854775808, %rdx
	xorq %rdx, %rax
	ucomiss %xmm1, %xmm0
	cmovb %rcx, %rax
	ret

Seems like the jb branch has high likelyhood of being taken. It would have
saved a few instructions.

//===---------------------------------------------------------------------===//

Poor codegen:

int X[2];
int b;
void test(void) {
  memset(X, b, 2*sizeof(X[0]));
}

llc:
	movq _b@GOTPCREL(%rip), %rax
	movzbq (%rax), %rax
	movq %rax, %rcx
	shlq $8, %rcx
	orq %rax, %rcx
	movq %rcx, %rax
	shlq $16, %rax
	orq %rcx, %rax
	movq %rax, %rcx
	shlq $32, %rcx
	movq _X@GOTPCREL(%rip), %rdx
	orq %rax, %rcx
	movq %rcx, (%rdx)
	ret

gcc:
	movq	_b@GOTPCREL(%rip), %rax
	movabsq	$72340172838076673, %rdx
	movzbq	(%rax), %rax
	imulq	%rdx, %rax
	movq	_X@GOTPCREL(%rip), %rdx
	movq	%rax, (%rdx)
	ret

//===---------------------------------------------------------------------===//

Vararg function prologue can be further optimized. Currently all XMM registers
are stored into register save area. Most of them can be eliminated since the
upper bound of the number of XMM registers used are passed in %al. gcc produces
something like the following:

	movzbl	%al, %edx
	leaq	0(,%rdx,4), %rax
	leaq	4+L2(%rip), %rdx
	leaq	239(%rsp), %rax
       	jmp	*%rdx
	movaps	%xmm7, -15(%rax)
	movaps	%xmm6, -31(%rax)
	movaps	%xmm5, -47(%rax)
	movaps	%xmm4, -63(%rax)
	movaps	%xmm3, -79(%rax)
	movaps	%xmm2, -95(%rax)
	movaps	%xmm1, -111(%rax)
	movaps	%xmm0, -127(%rax)
L2:

It jumps over the movaps that do not need to be stored. Hard to see this being
significant as it added 5 instruciton (including a indirect branch) to avoid
executing 0 to 8 stores in the function prologue.

Perhaps we can optimize for the common case where no XMM registers are used for
parameter passing. i.e. is %al == 0 jump over all stores. Or in the case of a
leaf function where we can determine that no XMM input parameter is need, avoid
emitting the stores at all.

//===---------------------------------------------------------------------===//

AMD64 has a complex calling convention for aggregate passing by value:

1. If the size of an object is larger than two eightbytes, or in C++, is a non- 
   POD structure or union type, or contains unaligned fields, it has class 
   MEMORY.
2. Both eightbytes get initialized to class NO_CLASS. 
3. Each field of an object is classified recursively so that always two fields
   are considered. The resulting class is calculated according to the classes
   of the fields in the eightbyte: 
   (a) If both classes are equal, this is the resulting class. 
   (b) If one of the classes is NO_CLASS, the resulting class is the other 
       class. 
   (c) If one of the classes is MEMORY, the result is the MEMORY class. 
   (d) If one of the classes is INTEGER, the result is the INTEGER. 
   (e) If one of the classes is X87, X87UP, COMPLEX_X87 class, MEMORY is used as
      class. 
   (f) Otherwise class SSE is used. 
4. Then a post merger cleanup is done: 
   (a) If one of the classes is MEMORY, the whole argument is passed in memory. 
   (b) If SSEUP is not preceeded by SSE, it is converted to SSE.

Currently llvm frontend does not handle this correctly.

Problem 1:
    typedef struct { int i; double d; } QuadWordS;
It is currently passed in two i64 integer registers. However, gcc compiled
callee expects the second element 'd' to be passed in XMM0.

Problem 2:
    typedef struct { int32_t i; float j; double d; } QuadWordS;
The size of the first two fields == i64 so they will be combined and passed in
a integer register RDI. The third field is still passed in XMM0.

Problem 3:
    typedef struct { int64_t i; int8_t j; int64_t d; } S;
    void test(S s)
The size of this aggregate is greater than two i64 so it should be passed in 
memory. Currently llvm breaks this down and passed it in three integer
registers.

Problem 4:
Taking problem 3 one step ahead where a function expects a aggregate value
in memory followed by more parameter(s) passed in register(s).
    void test(S s, int b)

LLVM IR does not allow parameter passing by aggregates, therefore it must break
the aggregates value (in problem 3 and 4) into a number of scalar values:
    void %test(long %s.i, byte %s.j, long %s.d);

However, if the backend were to lower this code literally it would pass the 3
values in integer registers. To force it be passed in memory, the frontend
should change the function signiture to:
    void %test(long %undef1, long %undef2, long %undef3, long %undef4, 
               long %undef5, long %undef6,
               long %s.i, byte %s.j, long %s.d);
And the callee would look something like this:
    call void %test( undef, undef, undef, undef, undef, undef,
                     %tmp.s.i, %tmp.s.j, %tmp.s.d );
The first 6 undef parameters would exhaust the 6 integer registers used for
parameter passing. The following three integer values would then be forced into
memory.

For problem 4, the parameter 'd' would be moved to the front of the parameter
list so it will be passed in register:
    void %test(int %d,
               long %undef1, long %undef2, long %undef3, long %undef4, 
               long %undef5, long %undef6,
               long %s.i, byte %s.j, long %s.d);

//===---------------------------------------------------------------------===//

Right now the asm printer assumes GlobalAddress are accessed via RIP relative
addressing. Therefore, it is not possible to generate this:
        movabsq $__ZTV10polynomialIdE+16, %rax

That is ok for now since we currently only support small model. So the above
is selected as
        leaq __ZTV10polynomialIdE+16(%rip), %rax

This is probably slightly slower but is much shorter than movabsq. However, if
we were to support medium or larger code models, we need to use the movabs
instruction. We should probably introduce something like AbsoluteAddress to
distinguish it from GlobalAddress so the asm printer and JIT code emitter can
do the right thing.

//===---------------------------------------------------------------------===//

It's not possible to reference AH, BH, CH, and DH registers in an instruction
requiring REX prefix. However, divb and mulb both produce results in AH. If isel
emits a CopyFromReg which gets turned into a movb and that can be allocated a
r8b - r15b.

To get around this, isel emits a CopyFromReg from AX and then right shift it
down by 8 and truncate it. It's not pretty but it works. We need some register
allocation magic to make the hack go away (e.g. putting additional constraints
on the result of the movb).

//===---------------------------------------------------------------------===//

The x86-64 ABI for hidden-argument struct returns requires that the
incoming value of %rdi be copied into %rax by the callee upon return.

The idea is that it saves callers from having to remember this value,
which would often require a callee-saved register. Callees usually
need to keep this value live for most of their body anyway, so it
doesn't add a significant burden on them.

We currently implement this in codegen, however this is suboptimal
because it means that it would be quite awkward to implement the
optimization for callers.

A better implementation would be to relax the LLVM IR rules for sret
arguments to allow a function with an sret argument to have a non-void
return type, and to have the front-end to set up the sret argument value
as the return value of the function. The front-end could more easily
emit uses of the returned struct value to be in terms of the function's
lowered return value, and it would free non-C frontends from a
complication only required by a C-based ABI.

//===---------------------------------------------------------------------===//
