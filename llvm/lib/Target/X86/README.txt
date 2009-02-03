//===---------------------------------------------------------------------===//
// Random ideas for the X86 backend.
//===---------------------------------------------------------------------===//


//===---------------------------------------------------------------------===//

CodeGen/X86/lea-3.ll:test3 should be a single LEA, not a shift/move.  The X86
backend knows how to three-addressify this shift, but it appears the register
allocator isn't even asking it to do so in this case.  We should investigate
why this isn't happening, it could have significant impact on other important
cases for X86 as well.

//===---------------------------------------------------------------------===//

This should be one DIV/IDIV instruction, not a libcall:

unsigned test(unsigned long long X, unsigned Y) {
        return X/Y;
}

This can be done trivially with a custom legalizer.  What about overflow 
though?  http://gcc.gnu.org/bugzilla/show_bug.cgi?id=14224

//===---------------------------------------------------------------------===//

Improvements to the multiply -> shift/add algorithm:
http://gcc.gnu.org/ml/gcc-patches/2004-08/msg01590.html

//===---------------------------------------------------------------------===//

Improve code like this (occurs fairly frequently, e.g. in LLVM):
long long foo(int x) { return 1LL << x; }

http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01109.html
http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01128.html
http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01136.html

Another useful one would be  ~0ULL >> X and ~0ULL << X.

One better solution for 1LL << x is:
        xorl    %eax, %eax
        xorl    %edx, %edx
        testb   $32, %cl
        sete    %al
        setne   %dl
        sall    %cl, %eax
        sall    %cl, %edx

But that requires good 8-bit subreg support.

Also, this might be better.  It's an extra shift, but it's one instruction
shorter, and doesn't stress 8-bit subreg support.
(From http://gcc.gnu.org/ml/gcc-patches/2004-09/msg01148.html,
but without the unnecessary and.)
        movl %ecx, %eax
        shrl $5, %eax
        movl %eax, %edx
        xorl $1, %edx
        sall %cl, %eax
        sall %cl. %edx

64-bit shifts (in general) expand to really bad code.  Instead of using
cmovs, we should expand to a conditional branch like GCC produces.

//===---------------------------------------------------------------------===//

Compile this:
_Bool f(_Bool a) { return a!=1; }

into:
        movzbl  %dil, %eax
        xorl    $1, %eax
        ret

(Although note that this isn't a legal way to express the code that llvm-gcc
currently generates for that function.)

//===---------------------------------------------------------------------===//

Some isel ideas:

1. Dynamic programming based approach when compile time if not an
   issue.
2. Code duplication (addressing mode) during isel.
3. Other ideas from "Register-Sensitive Selection, Duplication, and
   Sequencing of Instructions".
4. Scheduling for reduced register pressure.  E.g. "Minimum Register 
   Instruction Sequence Problem: Revisiting Optimal Code Generation for DAGs" 
   and other related papers.
   http://citeseer.ist.psu.edu/govindarajan01minimum.html

//===---------------------------------------------------------------------===//

Should we promote i16 to i32 to avoid partial register update stalls?

//===---------------------------------------------------------------------===//

Leave any_extend as pseudo instruction and hint to register
allocator. Delay codegen until post register allocation.
Note. any_extend is now turned into an INSERT_SUBREG. We still need to teach
the coalescer how to deal with it though.

//===---------------------------------------------------------------------===//

It appears icc use push for parameter passing. Need to investigate.

//===---------------------------------------------------------------------===//

Only use inc/neg/not instructions on processors where they are faster than
add/sub/xor.  They are slower on the P4 due to only updating some processor
flags.

//===---------------------------------------------------------------------===//

The instruction selector sometimes misses folding a load into a compare.  The
pattern is written as (cmp reg, (load p)).  Because the compare isn't 
commutative, it is not matched with the load on both sides.  The dag combiner
should be made smart enough to cannonicalize the load into the RHS of a compare
when it can invert the result of the compare for free.

//===---------------------------------------------------------------------===//

How about intrinsics? An example is:
  *res = _mm_mulhi_epu16(*A, _mm_mul_epu32(*B, *C));

compiles to
	pmuludq (%eax), %xmm0
	movl 8(%esp), %eax
	movdqa (%eax), %xmm1
	pmulhuw %xmm0, %xmm1

The transformation probably requires a X86 specific pass or a DAG combiner
target specific hook.

//===---------------------------------------------------------------------===//

In many cases, LLVM generates code like this:

_test:
        movl 8(%esp), %eax
        cmpl %eax, 4(%esp)
        setl %al
        movzbl %al, %eax
        ret

on some processors (which ones?), it is more efficient to do this:

_test:
        movl 8(%esp), %ebx
        xor  %eax, %eax
        cmpl %ebx, 4(%esp)
        setl %al
        ret

Doing this correctly is tricky though, as the xor clobbers the flags.

//===---------------------------------------------------------------------===//

We should generate bts/btr/etc instructions on targets where they are cheap or
when codesize is important.  e.g., for:

void setbit(int *target, int bit) {
    *target |= (1 << bit);
}
void clearbit(int *target, int bit) {
    *target &= ~(1 << bit);
}

//===---------------------------------------------------------------------===//

Instead of the following for memset char*, 1, 10:

	movl $16843009, 4(%edx)
	movl $16843009, (%edx)
	movw $257, 8(%edx)

It might be better to generate

	movl $16843009, %eax
	movl %eax, 4(%edx)
	movl %eax, (%edx)
	movw al, 8(%edx)
	
when we can spare a register. It reduces code size.

//===---------------------------------------------------------------------===//

Evaluate what the best way to codegen sdiv X, (2^C) is.  For X/8, we currently
get this:

define i32 @test1(i32 %X) {
    %Y = sdiv i32 %X, 8
    ret i32 %Y
}

_test1:
        movl 4(%esp), %eax
        movl %eax, %ecx
        sarl $31, %ecx
        shrl $29, %ecx
        addl %ecx, %eax
        sarl $3, %eax
        ret

GCC knows several different ways to codegen it, one of which is this:

_test1:
        movl    4(%esp), %eax
        cmpl    $-1, %eax
        leal    7(%eax), %ecx
        cmovle  %ecx, %eax
        sarl    $3, %eax
        ret

which is probably slower, but it's interesting at least :)

//===---------------------------------------------------------------------===//

We are currently lowering large (1MB+) memmove/memcpy to rep/stosl and rep/movsl
We should leave these as libcalls for everything over a much lower threshold,
since libc is hand tuned for medium and large mem ops (avoiding RFO for large
stores, TLB preheating, etc)

//===---------------------------------------------------------------------===//

Optimize this into something reasonable:
 x * copysign(1.0, y) * copysign(1.0, z)

//===---------------------------------------------------------------------===//

Optimize copysign(x, *y) to use an integer load from y.

//===---------------------------------------------------------------------===//

%X = weak global int 0

void %foo(int %N) {
	%N = cast int %N to uint
	%tmp.24 = setgt int %N, 0
	br bool %tmp.24, label %no_exit, label %return

no_exit:
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %no_exit ]
	%i.0.0 = cast uint %indvar to int
	volatile store int %i.0.0, int* %X
	%indvar.next = add uint %indvar, 1
	%exitcond = seteq uint %indvar.next, %N
	br bool %exitcond, label %return, label %no_exit

return:
	ret void
}

compiles into:

	.text
	.align	4
	.globl	_foo
_foo:
	movl 4(%esp), %eax
	cmpl $1, %eax
	jl LBB_foo_4	# return
LBB_foo_1:	# no_exit.preheader
	xorl %ecx, %ecx
LBB_foo_2:	# no_exit
	movl L_X$non_lazy_ptr, %edx
	movl %ecx, (%edx)
	incl %ecx
	cmpl %eax, %ecx
	jne LBB_foo_2	# no_exit
LBB_foo_3:	# return.loopexit
LBB_foo_4:	# return
	ret

We should hoist "movl L_X$non_lazy_ptr, %edx" out of the loop after
remateralization is implemented. This can be accomplished with 1) a target
dependent LICM pass or 2) makeing SelectDAG represent the whole function. 

//===---------------------------------------------------------------------===//

The following tests perform worse with LSR:

lambda, siod, optimizer-eval, ackermann, hash2, nestedloop, strcat, and Treesor.

//===---------------------------------------------------------------------===//

We are generating far worse code than gcc:

volatile short X, Y;

void foo(int N) {
  int i;
  for (i = 0; i < N; i++) { X = i; Y = i*4; }
}

LBB1_1:	# entry.bb_crit_edge
	xorl	%ecx, %ecx
	xorw	%dx, %dx
LBB1_2:	# bb
	movl	L_X$non_lazy_ptr, %esi
	movw	%cx, (%esi)
	movl	L_Y$non_lazy_ptr, %esi
	movw	%dx, (%esi)
	addw	$4, %dx
	incl	%ecx
	cmpl	%eax, %ecx
	jne	LBB1_2	# bb

vs.

	xorl	%edx, %edx
	movl	L_X$non_lazy_ptr-"L00000000001$pb"(%ebx), %esi
	movl	L_Y$non_lazy_ptr-"L00000000001$pb"(%ebx), %ecx
L4:
	movw	%dx, (%esi)
	leal	0(,%edx,4), %eax
	movw	%ax, (%ecx)
	addl	$1, %edx
	cmpl	%edx, %edi
	jne	L4

This is due to the lack of post regalloc LICM.

//===---------------------------------------------------------------------===//

Teach the coalescer to coalesce vregs of different register classes. e.g. FR32 /
FR64 to VR128.

//===---------------------------------------------------------------------===//

Adding to the list of cmp / test poor codegen issues:

int test(__m128 *A, __m128 *B) {
  if (_mm_comige_ss(*A, *B))
    return 3;
  else
    return 4;
}

_test:
	movl 8(%esp), %eax
	movaps (%eax), %xmm0
	movl 4(%esp), %eax
	movaps (%eax), %xmm1
	comiss %xmm0, %xmm1
	setae %al
	movzbl %al, %ecx
	movl $3, %eax
	movl $4, %edx
	cmpl $0, %ecx
	cmove %edx, %eax
	ret

Note the setae, movzbl, cmpl, cmove can be replaced with a single cmovae. There
are a number of issues. 1) We are introducing a setcc between the result of the
intrisic call and select. 2) The intrinsic is expected to produce a i32 value
so a any extend (which becomes a zero extend) is added.

We probably need some kind of target DAG combine hook to fix this.

//===---------------------------------------------------------------------===//

We generate significantly worse code for this than GCC:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=21150
http://gcc.gnu.org/bugzilla/attachment.cgi?id=8701

There is also one case we do worse on PPC.

//===---------------------------------------------------------------------===//

For this:

int test(int a)
{
  return a * 3;
}

We currently emits
	imull $3, 4(%esp), %eax

Perhaps this is what we really should generate is? Is imull three or four
cycles? Note: ICC generates this:
	movl	4(%esp), %eax
	leal	(%eax,%eax,2), %eax

The current instruction priority is based on pattern complexity. The former is
more "complex" because it folds a load so the latter will not be emitted.

Perhaps we should use AddedComplexity to give LEA32r a higher priority? We
should always try to match LEA first since the LEA matching code does some
estimate to determine whether the match is profitable.

However, if we care more about code size, then imull is better. It's two bytes
shorter than movl + leal.

On a Pentium M, both variants have the same characteristics with regard
to throughput; however, the multiplication has a latency of four cycles, as
opposed to two cycles for the movl+lea variant.

//===---------------------------------------------------------------------===//

__builtin_ffs codegen is messy.

int ffs_(unsigned X) { return __builtin_ffs(X); }

llvm produces:
ffs_:
        movl    4(%esp), %ecx
        bsfl    %ecx, %eax
        movl    $32, %edx
        cmove   %edx, %eax
        incl    %eax
        xorl    %edx, %edx
        testl   %ecx, %ecx
        cmove   %edx, %eax
        ret

vs gcc:

_ffs_:
        movl    $-1, %edx
        bsfl    4(%esp), %eax
        cmove   %edx, %eax
        addl    $1, %eax
        ret

Another example of __builtin_ffs (use predsimplify to eliminate a select):

int foo (unsigned long j) {
  if (j)
    return __builtin_ffs (j) - 1;
  else
    return 0;
}

//===---------------------------------------------------------------------===//

It appears gcc place string data with linkonce linkage in
.section __TEXT,__const_coal,coalesced instead of
.section __DATA,__const_coal,coalesced.
Take a look at darwin.h, there are other Darwin assembler directives that we
do not make use of.

//===---------------------------------------------------------------------===//

define i32 @foo(i32* %a, i32 %t) {
entry:
	br label %cond_true

cond_true:		; preds = %cond_true, %entry
	%x.0.0 = phi i32 [ 0, %entry ], [ %tmp9, %cond_true ]		; <i32> [#uses=3]
	%t_addr.0.0 = phi i32 [ %t, %entry ], [ %tmp7, %cond_true ]		; <i32> [#uses=1]
	%tmp2 = getelementptr i32* %a, i32 %x.0.0		; <i32*> [#uses=1]
	%tmp3 = load i32* %tmp2		; <i32> [#uses=1]
	%tmp5 = add i32 %t_addr.0.0, %x.0.0		; <i32> [#uses=1]
	%tmp7 = add i32 %tmp5, %tmp3		; <i32> [#uses=2]
	%tmp9 = add i32 %x.0.0, 1		; <i32> [#uses=2]
	%tmp = icmp sgt i32 %tmp9, 39		; <i1> [#uses=1]
	br i1 %tmp, label %bb12, label %cond_true

bb12:		; preds = %cond_true
	ret i32 %tmp7
}
is pessimized by -loop-reduce and -indvars

//===---------------------------------------------------------------------===//

u32 to float conversion improvement:

float uint32_2_float( unsigned u ) {
  float fl = (int) (u & 0xffff);
  float fh = (int) (u >> 16);
  fh *= 0x1.0p16f;
  return fh + fl;
}

00000000        subl    $0x04,%esp
00000003        movl    0x08(%esp,1),%eax
00000007        movl    %eax,%ecx
00000009        shrl    $0x10,%ecx
0000000c        cvtsi2ss        %ecx,%xmm0
00000010        andl    $0x0000ffff,%eax
00000015        cvtsi2ss        %eax,%xmm1
00000019        mulss   0x00000078,%xmm0
00000021        addss   %xmm1,%xmm0
00000025        movss   %xmm0,(%esp,1)
0000002a        flds    (%esp,1)
0000002d        addl    $0x04,%esp
00000030        ret

//===---------------------------------------------------------------------===//

When using fastcc abi, align stack slot of argument of type double on 8 byte
boundary to improve performance.

//===---------------------------------------------------------------------===//

Codegen:

int f(int a, int b) {
  if (a == 4 || a == 6)
    b++;
  return b;
}


as:

or eax, 2
cmp eax, 6
jz label

//===---------------------------------------------------------------------===//

GCC's ix86_expand_int_movcc function (in i386.c) has a ton of interesting
simplifications for integer "x cmp y ? a : b".  For example, instead of:

int G;
void f(int X, int Y) {
  G = X < 0 ? 14 : 13;
}

compiling to:

_f:
        movl $14, %eax
        movl $13, %ecx
        movl 4(%esp), %edx
        testl %edx, %edx
        cmovl %eax, %ecx
        movl %ecx, _G
        ret

it could be:
_f:
        movl    4(%esp), %eax
        sarl    $31, %eax
        notl    %eax
        addl    $14, %eax
        movl    %eax, _G
        ret

etc.

Another is:
int usesbb(unsigned int a, unsigned int b) {
       return (a < b ? -1 : 0);
}
to:
_usesbb:
	movl	8(%esp), %eax
	cmpl	%eax, 4(%esp)
	sbbl	%eax, %eax
	ret

instead of:
_usesbb:
	xorl	%eax, %eax
	movl	8(%esp), %ecx
	cmpl	%ecx, 4(%esp)
	movl	$4294967295, %ecx
	cmovb	%ecx, %eax
	ret

//===---------------------------------------------------------------------===//

Currently we don't have elimination of redundant stack manipulations. Consider
the code:

int %main() {
entry:
	call fastcc void %test1( )
	call fastcc void %test2( sbyte* cast (void ()* %test1 to sbyte*) )
	ret int 0
}

declare fastcc void %test1()

declare fastcc void %test2(sbyte*)


This currently compiles to:

	subl $16, %esp
	call _test5
	addl $12, %esp
	subl $16, %esp
	movl $_test5, (%esp)
	call _test6
	addl $12, %esp

The add\sub pair is really unneeded here.

//===---------------------------------------------------------------------===//

Consider the expansion of:

define i32 @test3(i32 %X) {
        %tmp1 = urem i32 %X, 255
        ret i32 %tmp1
}

Currently it compiles to:

...
        movl $2155905153, %ecx
        movl 8(%esp), %esi
        movl %esi, %eax
        mull %ecx
...

This could be "reassociated" into:

        movl $2155905153, %eax
        movl 8(%esp), %ecx
        mull %ecx

to avoid the copy.  In fact, the existing two-address stuff would do this
except that mul isn't a commutative 2-addr instruction.  I guess this has
to be done at isel time based on the #uses to mul?

//===---------------------------------------------------------------------===//

Make sure the instruction which starts a loop does not cross a cacheline
boundary. This requires knowning the exact length of each machine instruction.
That is somewhat complicated, but doable. Example 256.bzip2:

In the new trace, the hot loop has an instruction which crosses a cacheline
boundary.  In addition to potential cache misses, this can't help decoding as I
imagine there has to be some kind of complicated decoder reset and realignment
to grab the bytes from the next cacheline.

532  532 0x3cfc movb     (1809(%esp, %esi), %bl   <<<--- spans 2 64 byte lines
942  942 0x3d03 movl     %dh, (1809(%esp, %esi)
937  937 0x3d0a incl     %esi
3    3   0x3d0b cmpb     %bl, %dl
27   27  0x3d0d jnz      0x000062db <main+11707>

//===---------------------------------------------------------------------===//

In c99 mode, the preprocessor doesn't like assembly comments like #TRUNCATE.

//===---------------------------------------------------------------------===//

This could be a single 16-bit load.

int f(char *p) {
    if ((p[0] == 1) & (p[1] == 2)) return 1;
    return 0;
}

//===---------------------------------------------------------------------===//

We should inline lrintf and probably other libc functions.

//===---------------------------------------------------------------------===//

Start using the flags more.  For example, compile:

int add_zf(int *x, int y, int a, int b) {
     if ((*x += y) == 0)
          return a;
     else
          return b;
}

to:
       addl    %esi, (%rdi)
       movl    %edx, %eax
       cmovne  %ecx, %eax
       ret
instead of:

_add_zf:
        addl (%rdi), %esi
        movl %esi, (%rdi)
        testl %esi, %esi
        cmove %edx, %ecx
        movl %ecx, %eax
        ret

and:

int add_zf(int *x, int y, int a, int b) {
     if ((*x + y) < 0)
          return a;
     else
          return b;
}

to:

add_zf:
        addl    (%rdi), %esi
        movl    %edx, %eax
        cmovns  %ecx, %eax
        ret

instead of:

_add_zf:
        addl (%rdi), %esi
        testl %esi, %esi
        cmovs %edx, %ecx
        movl %ecx, %eax
        ret

//===---------------------------------------------------------------------===//

These two functions have identical effects:

unsigned int f(unsigned int i, unsigned int n) {++i; if (i == n) ++i; return i;}
unsigned int f2(unsigned int i, unsigned int n) {++i; i += i == n; return i;}

We currently compile them to:

_f:
        movl 4(%esp), %eax
        movl %eax, %ecx
        incl %ecx
        movl 8(%esp), %edx
        cmpl %edx, %ecx
        jne LBB1_2      #UnifiedReturnBlock
LBB1_1: #cond_true
        addl $2, %eax
        ret
LBB1_2: #UnifiedReturnBlock
        movl %ecx, %eax
        ret
_f2:
        movl 4(%esp), %eax
        movl %eax, %ecx
        incl %ecx
        cmpl 8(%esp), %ecx
        sete %cl
        movzbl %cl, %ecx
        leal 1(%ecx,%eax), %eax
        ret

both of which are inferior to GCC's:

_f:
        movl    4(%esp), %edx
        leal    1(%edx), %eax
        addl    $2, %edx
        cmpl    8(%esp), %eax
        cmove   %edx, %eax
        ret
_f2:
        movl    4(%esp), %eax
        addl    $1, %eax
        xorl    %edx, %edx
        cmpl    8(%esp), %eax
        sete    %dl
        addl    %edx, %eax
        ret

//===---------------------------------------------------------------------===//

This code:

void test(int X) {
  if (X) abort();
}

is currently compiled to:

_test:
        subl $12, %esp
        cmpl $0, 16(%esp)
        jne LBB1_1
        addl $12, %esp
        ret
LBB1_1:
        call L_abort$stub

It would be better to produce:

_test:
        subl $12, %esp
        cmpl $0, 16(%esp)
        jne L_abort$stub
        addl $12, %esp
        ret

This can be applied to any no-return function call that takes no arguments etc.
Alternatively, the stack save/restore logic could be shrink-wrapped, producing
something like this:

_test:
        cmpl $0, 4(%esp)
        jne LBB1_1
        ret
LBB1_1:
        subl $12, %esp
        call L_abort$stub

Both are useful in different situations.  Finally, it could be shrink-wrapped
and tail called, like this:

_test:
        cmpl $0, 4(%esp)
        jne LBB1_1
        ret
LBB1_1:
        pop %eax   # realign stack.
        call L_abort$stub

Though this probably isn't worth it.

//===---------------------------------------------------------------------===//

We need to teach the codegen to convert two-address INC instructions to LEA
when the flags are dead (likewise dec).  For example, on X86-64, compile:

int foo(int A, int B) {
  return A+1;
}

to:

_foo:
        leal    1(%edi), %eax
        ret

instead of:

_foo:
        incl %edi
        movl %edi, %eax
        ret

Another example is:

;; X's live range extends beyond the shift, so the register allocator
;; cannot coalesce it with Y.  Because of this, a copy needs to be
;; emitted before the shift to save the register value before it is
;; clobbered.  However, this copy is not needed if the register
;; allocator turns the shift into an LEA.  This also occurs for ADD.

; Check that the shift gets turned into an LEA.
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | \
; RUN:   not grep {mov E.X, E.X}

@G = external global i32		; <i32*> [#uses=3]

define i32 @test1(i32 %X, i32 %Y) {
	%Z = add i32 %X, %Y		; <i32> [#uses=1]
	volatile store i32 %Y, i32* @G
	volatile store i32 %Z, i32* @G
	ret i32 %X
}

define i32 @test2(i32 %X) {
	%Z = add i32 %X, 1		; <i32> [#uses=1]
	volatile store i32 %Z, i32* @G
	ret i32 %X
}

//===---------------------------------------------------------------------===//

Sometimes it is better to codegen subtractions from a constant (e.g. 7-x) with
a neg instead of a sub instruction.  Consider:

int test(char X) { return 7-X; }

we currently produce:
_test:
        movl $7, %eax
        movsbl 4(%esp), %ecx
        subl %ecx, %eax
        ret

We would use one fewer register if codegen'd as:

        movsbl 4(%esp), %eax
	neg %eax
        add $7, %eax
        ret

Note that this isn't beneficial if the load can be folded into the sub.  In
this case, we want a sub:

int test(int X) { return 7-X; }
_test:
        movl $7, %eax
        subl 4(%esp), %eax
        ret

//===---------------------------------------------------------------------===//

Leaf functions that require one 4-byte spill slot have a prolog like this:

_foo:
        pushl   %esi
        subl    $4, %esp
...
and an epilog like this:
        addl    $4, %esp
        popl    %esi
        ret

It would be smaller, and potentially faster, to push eax on entry and to
pop into a dummy register instead of using addl/subl of esp.  Just don't pop 
into any return registers :)

//===---------------------------------------------------------------------===//

The X86 backend should fold (branch (or (setcc, setcc))) into multiple 
branches.  We generate really poor code for:

double testf(double a) {
       return a == 0.0 ? 0.0 : (a > 0.0 ? 1.0 : -1.0);
}

For example, the entry BB is:

_testf:
        subl    $20, %esp
        pxor    %xmm0, %xmm0
        movsd   24(%esp), %xmm1
        ucomisd %xmm0, %xmm1
        setnp   %al
        sete    %cl
        testb   %cl, %al
        jne     LBB1_5  # UnifiedReturnBlock
LBB1_1: # cond_true


it would be better to replace the last four instructions with:

	jp LBB1_1
	je LBB1_5
LBB1_1:

We also codegen the inner ?: into a diamond:

       cvtss2sd        LCPI1_0(%rip), %xmm2
        cvtss2sd        LCPI1_1(%rip), %xmm3
        ucomisd %xmm1, %xmm0
        ja      LBB1_3  # cond_true
LBB1_2: # cond_true
        movapd  %xmm3, %xmm2
LBB1_3: # cond_true
        movapd  %xmm2, %xmm0
        ret

We should sink the load into xmm3 into the LBB1_2 block.  This should
be pretty easy, and will nuke all the copies.

//===---------------------------------------------------------------------===//

This:
        #include <algorithm>
        inline std::pair<unsigned, bool> full_add(unsigned a, unsigned b)
        { return std::make_pair(a + b, a + b < a); }
        bool no_overflow(unsigned a, unsigned b)
        { return !full_add(a, b).second; }

Should compile to:


        _Z11no_overflowjj:
                addl    %edi, %esi
                setae   %al
                ret

FIXME: That code looks wrong; bool return is normally defined as zext.

on x86-64, not:

__Z11no_overflowjj:
        addl    %edi, %esi
        cmpl    %edi, %esi
        setae   %al
        movzbl  %al, %eax
        ret


//===---------------------------------------------------------------------===//

Re-materialize MOV32r0 etc. with xor instead of changing them to moves if the
condition register is dead. xor reg reg is shorter than mov reg, #0.

//===---------------------------------------------------------------------===//

We aren't matching RMW instructions aggressively
enough.  Here's a reduced testcase (more in PR1160):

define void @test(i32* %huge_ptr, i32* %target_ptr) {
        %A = load i32* %huge_ptr                ; <i32> [#uses=1]
        %B = load i32* %target_ptr              ; <i32> [#uses=1]
        %C = or i32 %A, %B              ; <i32> [#uses=1]
        store i32 %C, i32* %target_ptr
        ret void
}

$ llvm-as < t.ll | llc -march=x86-64

_test:
        movl (%rdi), %eax
        orl (%rsi), %eax
        movl %eax, (%rsi)
        ret

That should be something like:

_test:
        movl (%rdi), %eax
        orl %eax, (%rsi)
        ret

//===---------------------------------------------------------------------===//

The following code:

bb114.preheader:		; preds = %cond_next94
	%tmp231232 = sext i16 %tmp62 to i32		; <i32> [#uses=1]
	%tmp233 = sub i32 32, %tmp231232		; <i32> [#uses=1]
	%tmp245246 = sext i16 %tmp65 to i32		; <i32> [#uses=1]
	%tmp252253 = sext i16 %tmp68 to i32		; <i32> [#uses=1]
	%tmp254 = sub i32 32, %tmp252253		; <i32> [#uses=1]
	%tmp553554 = bitcast i16* %tmp37 to i8*		; <i8*> [#uses=2]
	%tmp583584 = sext i16 %tmp98 to i32		; <i32> [#uses=1]
	%tmp585 = sub i32 32, %tmp583584		; <i32> [#uses=1]
	%tmp614615 = sext i16 %tmp101 to i32		; <i32> [#uses=1]
	%tmp621622 = sext i16 %tmp104 to i32		; <i32> [#uses=1]
	%tmp623 = sub i32 32, %tmp621622		; <i32> [#uses=1]
	br label %bb114

produces:

LBB3_5:	# bb114.preheader
	movswl	-68(%ebp), %eax
	movl	$32, %ecx
	movl	%ecx, -80(%ebp)
	subl	%eax, -80(%ebp)
	movswl	-52(%ebp), %eax
	movl	%ecx, -84(%ebp)
	subl	%eax, -84(%ebp)
	movswl	-70(%ebp), %eax
	movl	%ecx, -88(%ebp)
	subl	%eax, -88(%ebp)
	movswl	-50(%ebp), %eax
	subl	%eax, %ecx
	movl	%ecx, -76(%ebp)
	movswl	-42(%ebp), %eax
	movl	%eax, -92(%ebp)
	movswl	-66(%ebp), %eax
	movl	%eax, -96(%ebp)
	movw	$0, -98(%ebp)

This appears to be bad because the RA is not folding the store to the stack 
slot into the movl.  The above instructions could be:
	movl    $32, -80(%ebp)
...
	movl    $32, -84(%ebp)
...
This seems like a cross between remat and spill folding.

This has redundant subtractions of %eax from a stack slot. However, %ecx doesn't
change, so we could simply subtract %eax from %ecx first and then use %ecx (or
vice-versa).

//===---------------------------------------------------------------------===//

This code:

	%tmp659 = icmp slt i16 %tmp654, 0		; <i1> [#uses=1]
	br i1 %tmp659, label %cond_true662, label %cond_next715

produces this:

	testw	%cx, %cx
	movswl	%cx, %esi
	jns	LBB4_109	# cond_next715

Shark tells us that using %cx in the testw instruction is sub-optimal. It
suggests using the 32-bit register (which is what ICC uses).

//===---------------------------------------------------------------------===//

We compile this:

void compare (long long foo) {
  if (foo < 4294967297LL)
    abort();
}

to:

compare:
        subl    $4, %esp
        cmpl    $0, 8(%esp)
        setne   %al
        movzbw  %al, %ax
        cmpl    $1, 12(%esp)
        setg    %cl
        movzbw  %cl, %cx
        cmove   %ax, %cx
        testb   $1, %cl
        jne     .LBB1_2 # UnifiedReturnBlock
.LBB1_1:        # ifthen
        call    abort
.LBB1_2:        # UnifiedReturnBlock
        addl    $4, %esp
        ret

(also really horrible code on ppc).  This is due to the expand code for 64-bit
compares.  GCC produces multiple branches, which is much nicer:

compare:
        subl    $12, %esp
        movl    20(%esp), %edx
        movl    16(%esp), %eax
        decl    %edx
        jle     .L7
.L5:
        addl    $12, %esp
        ret
        .p2align 4,,7
.L7:
        jl      .L4
        cmpl    $0, %eax
        .p2align 4,,8
        ja      .L5
.L4:
        .p2align 4,,9
        call    abort

//===---------------------------------------------------------------------===//

Tail call optimization improvements: Tail call optimization currently
pushes all arguments on the top of the stack (their normal place for
non-tail call optimized calls) that source from the callers arguments
or  that source from a virtual register (also possibly sourcing from
callers arguments).
This is done to prevent overwriting of parameters (see example
below) that might be used later.

example:  

int callee(int32, int64); 
int caller(int32 arg1, int32 arg2) { 
  int64 local = arg2 * 2; 
  return callee(arg2, (int64)local); 
}

[arg1]          [!arg2 no longer valid since we moved local onto it]
[arg2]      ->  [(int64)
[RETADDR]        local  ]

Moving arg1 onto the stack slot of callee function would overwrite
arg2 of the caller.

Possible optimizations:


 - Analyse the actual parameters of the callee to see which would
   overwrite a caller parameter which is used by the callee and only
   push them onto the top of the stack.

   int callee (int32 arg1, int32 arg2);
   int caller (int32 arg1, int32 arg2) {
       return callee(arg1,arg2);
   }

   Here we don't need to write any variables to the top of the stack
   since they don't overwrite each other.

   int callee (int32 arg1, int32 arg2);
   int caller (int32 arg1, int32 arg2) {
       return callee(arg2,arg1);
   }

   Here we need to push the arguments because they overwrite each
   other.

//===---------------------------------------------------------------------===//

main ()
{
  int i = 0;
  unsigned long int z = 0;

  do {
    z -= 0x00004000;
    i++;
    if (i > 0x00040000)
      abort ();
  } while (z > 0);
  exit (0);
}

gcc compiles this to:

_main:
	subl	$28, %esp
	xorl	%eax, %eax
	jmp	L2
L3:
	cmpl	$262144, %eax
	je	L10
L2:
	addl	$1, %eax
	cmpl	$262145, %eax
	jne	L3
	call	L_abort$stub
L10:
	movl	$0, (%esp)
	call	L_exit$stub

llvm:

_main:
	subl	$12, %esp
	movl	$1, %eax
	movl	$16384, %ecx
LBB1_1:	# bb
	cmpl	$262145, %eax
	jge	LBB1_4	# cond_true
LBB1_2:	# cond_next
	incl	%eax
	addl	$4294950912, %ecx
	cmpl	$16384, %ecx
	jne	LBB1_1	# bb
LBB1_3:	# bb11
	xorl	%eax, %eax
	addl	$12, %esp
	ret
LBB1_4:	# cond_true
	call	L_abort$stub

1. LSR should rewrite the first cmp with induction variable %ecx.
2. DAG combiner should fold
        leal    1(%eax), %edx
        cmpl    $262145, %edx
   =>
        cmpl    $262144, %eax

//===---------------------------------------------------------------------===//

define i64 @test(double %X) {
	%Y = fptosi double %X to i64
	ret i64 %Y
}

compiles to:

_test:
	subl	$20, %esp
	movsd	24(%esp), %xmm0
	movsd	%xmm0, 8(%esp)
	fldl	8(%esp)
	fisttpll	(%esp)
	movl	4(%esp), %edx
	movl	(%esp), %eax
	addl	$20, %esp
	#FP_REG_KILL
	ret

This should just fldl directly from the input stack slot.

//===---------------------------------------------------------------------===//

This code:
int foo (int x) { return (x & 65535) | 255; }

Should compile into:

_foo:
        movzwl  4(%esp), %eax
        orl     $255, %eax
        ret

instead of:
_foo:
        movl    $255, %eax
        orl     4(%esp), %eax
        andl    $65535, %eax
        ret

//===---------------------------------------------------------------------===//

We're codegen'ing multiply of long longs inefficiently:

unsigned long long LLM(unsigned long long arg1, unsigned long long arg2) {
  return arg1 *  arg2;
}

We compile to (fomit-frame-pointer):

_LLM:
	pushl	%esi
	movl	8(%esp), %ecx
	movl	16(%esp), %esi
	movl	%esi, %eax
	mull	%ecx
	imull	12(%esp), %esi
	addl	%edx, %esi
	imull	20(%esp), %ecx
	movl	%esi, %edx
	addl	%ecx, %edx
	popl	%esi
	ret

This looks like a scheduling deficiency and lack of remat of the load from
the argument area.  ICC apparently produces:

        movl      8(%esp), %ecx
        imull     12(%esp), %ecx
        movl      16(%esp), %eax
        imull     4(%esp), %eax 
        addl      %eax, %ecx  
        movl      4(%esp), %eax
        mull      12(%esp) 
        addl      %ecx, %edx
        ret

Note that it remat'd loads from 4(esp) and 12(esp).  See this GCC PR:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=17236

//===---------------------------------------------------------------------===//

We can fold a store into "zeroing a reg".  Instead of:

xorl    %eax, %eax
movl    %eax, 124(%esp)

we should get:

movl    $0, 124(%esp)

if the flags of the xor are dead.

Likewise, we isel "x<<1" into "add reg,reg".  If reg is spilled, this should
be folded into: shl [mem], 1

//===---------------------------------------------------------------------===//

This testcase misses a read/modify/write opportunity (from PR1425):

void vertical_decompose97iH1(int *b0, int *b1, int *b2, int width){
    int i;
    for(i=0; i<width; i++)
        b1[i] += (1*(b0[i] + b2[i])+0)>>0;
}

We compile it down to:

LBB1_2:	# bb
	movl	(%esi,%edi,4), %ebx
	addl	(%ecx,%edi,4), %ebx
	addl	(%edx,%edi,4), %ebx
	movl	%ebx, (%ecx,%edi,4)
	incl	%edi
	cmpl	%eax, %edi
	jne	LBB1_2	# bb

the inner loop should add to the memory location (%ecx,%edi,4), saving
a mov.  Something like:

        movl    (%esi,%edi,4), %ebx
        addl    (%edx,%edi,4), %ebx
        addl    %ebx, (%ecx,%edi,4)

Here is another interesting example:

void vertical_compose97iH1(int *b0, int *b1, int *b2, int width){
    int i;
    for(i=0; i<width; i++)
        b1[i] -= (1*(b0[i] + b2[i])+0)>>0;
}

We miss the r/m/w opportunity here by using 2 subs instead of an add+sub[mem]:

LBB9_2:	# bb
	movl	(%ecx,%edi,4), %ebx
	subl	(%esi,%edi,4), %ebx
	subl	(%edx,%edi,4), %ebx
	movl	%ebx, (%ecx,%edi,4)
	incl	%edi
	cmpl	%eax, %edi
	jne	LBB9_2	# bb

Additionally, LSR should rewrite the exit condition of these loops to use
a stride-4 IV, would would allow all the scales in the loop to go away.
This would result in smaller code and more efficient microops.

//===---------------------------------------------------------------------===//

In SSE mode, we turn abs and neg into a load from the constant pool plus a xor
or and instruction, for example:

	xorpd	LCPI1_0, %xmm2

However, if xmm2 gets spilled, we end up with really ugly code like this:

	movsd	(%esp), %xmm0
	xorpd	LCPI1_0, %xmm0
	movsd	%xmm0, (%esp)

Since we 'know' that this is a 'neg', we can actually "fold" the spill into
the neg/abs instruction, turning it into an *integer* operation, like this:

	xorl 2147483648, [mem+4]     ## 2147483648 = (1 << 31)

you could also use xorb, but xorl is less likely to lead to a partial register
stall.  Here is a contrived testcase:

double a, b, c;
void test(double *P) {
  double X = *P;
  a = X;
  bar();
  X = -X;
  b = X;
  bar();
  c = X;
}

//===---------------------------------------------------------------------===//

handling llvm.memory.barrier on pre SSE2 cpus

should generate:
lock ; mov %esp, %esp

//===---------------------------------------------------------------------===//

The generated code on x86 for checking for signed overflow on a multiply the
obvious way is much longer than it needs to be.

int x(int a, int b) {
  long long prod = (long long)a*b;
  return  prod > 0x7FFFFFFF || prod < (-0x7FFFFFFF-1);
}

See PR2053 for more details.

//===---------------------------------------------------------------------===//

We should investigate using cdq/ctld (effect: edx = sar eax, 31)
more aggressively; it should cost the same as a move+shift on any modern
processor, but it's a lot shorter. Downside is that it puts more
pressure on register allocation because it has fixed operands.

Example:
int abs(int x) {return x < 0 ? -x : x;}

gcc compiles this to the following when using march/mtune=pentium2/3/4/m/etc.:
abs:
        movl    4(%esp), %eax
        cltd
        xorl    %edx, %eax
        subl    %edx, %eax
        ret

//===---------------------------------------------------------------------===//

Consider:
int test(unsigned long a, unsigned long b) { return -(a < b); }

We currently compile this to:

define i32 @test(i32 %a, i32 %b) nounwind  {
	%tmp3 = icmp ult i32 %a, %b		; <i1> [#uses=1]
	%tmp34 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	%tmp5 = sub i32 0, %tmp34		; <i32> [#uses=1]
	ret i32 %tmp5
}

and

_test:
	movl	8(%esp), %eax
	cmpl	%eax, 4(%esp)
	setb	%al
	movzbl	%al, %eax
	negl	%eax
	ret

Several deficiencies here.  First, we should instcombine zext+neg into sext:

define i32 @test2(i32 %a, i32 %b) nounwind  {
	%tmp3 = icmp ult i32 %a, %b		; <i1> [#uses=1]
	%tmp34 = sext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp34
}

However, before we can do that, we have to fix the bad codegen that we get for
sext from bool:

_test2:
	movl	8(%esp), %eax
	cmpl	%eax, 4(%esp)
	setb	%al
	movzbl	%al, %eax
	shll	$31, %eax
	sarl	$31, %eax
	ret

This code should be at least as good as the code above.  Once this is fixed, we
can optimize this specific case even more to:

	movl	8(%esp), %eax
	xorl	%ecx, %ecx
        cmpl    %eax, 4(%esp)
        sbbl    %ecx, %ecx

//===---------------------------------------------------------------------===//

Take the following code (from 
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=16541):

extern unsigned char first_one[65536];
int FirstOnet(unsigned long long arg1)
{
  if (arg1 >> 48)
    return (first_one[arg1 >> 48]);
  return 0;
}


The following code is currently generated:
FirstOnet:
        movl    8(%esp), %eax
        cmpl    $65536, %eax
        movl    4(%esp), %ecx
        jb      .LBB1_2 # UnifiedReturnBlock
.LBB1_1:        # ifthen
        shrl    $16, %eax
        movzbl  first_one(%eax), %eax
        ret
.LBB1_2:        # UnifiedReturnBlock
        xorl    %eax, %eax
        ret

There are a few possible improvements here:
1. We should be able to eliminate the dead load into %ecx
2. We could change the "movl 8(%esp), %eax" into
   "movzwl 10(%esp), %eax"; this lets us change the cmpl
   into a testl, which is shorter, and eliminate the shift.

We could also in theory eliminate the branch by using a conditional
for the address of the load, but that seems unlikely to be worthwhile
in general.

//===---------------------------------------------------------------------===//

We compile this function:

define i32 @foo(i32 %a, i32 %b, i32 %c, i8 zeroext  %d) nounwind  {
entry:
	%tmp2 = icmp eq i8 %d, 0		; <i1> [#uses=1]
	br i1 %tmp2, label %bb7, label %bb

bb:		; preds = %entry
	%tmp6 = add i32 %b, %a		; <i32> [#uses=1]
	ret i32 %tmp6

bb7:		; preds = %entry
	%tmp10 = sub i32 %a, %c		; <i32> [#uses=1]
	ret i32 %tmp10
}

to:

_foo:
	cmpb	$0, 16(%esp)
	movl	12(%esp), %ecx
	movl	8(%esp), %eax
	movl	4(%esp), %edx
	je	LBB1_2	# bb7
LBB1_1:	# bb
	addl	%edx, %eax
	ret
LBB1_2:	# bb7
	movl	%edx, %eax
	subl	%ecx, %eax
	ret

The coalescer could coalesce "edx" with "eax" to avoid the movl in LBB1_2
if it commuted the addl in LBB1_1.

//===---------------------------------------------------------------------===//

See rdar://4653682.

From flops:

LBB1_15:        # bb310
        cvtss2sd        LCPI1_0, %xmm1
        addsd   %xmm1, %xmm0
        movsd   176(%esp), %xmm2
        mulsd   %xmm0, %xmm2
        movapd  %xmm2, %xmm3
        mulsd   %xmm3, %xmm3
        movapd  %xmm3, %xmm4
        mulsd   LCPI1_23, %xmm4
        addsd   LCPI1_24, %xmm4
        mulsd   %xmm3, %xmm4
        addsd   LCPI1_25, %xmm4
        mulsd   %xmm3, %xmm4
        addsd   LCPI1_26, %xmm4
        mulsd   %xmm3, %xmm4
        addsd   LCPI1_27, %xmm4
        mulsd   %xmm3, %xmm4
        addsd   LCPI1_28, %xmm4
        mulsd   %xmm3, %xmm4
        addsd   %xmm1, %xmm4
        mulsd   %xmm2, %xmm4
        movsd   152(%esp), %xmm1
        addsd   %xmm4, %xmm1
        movsd   %xmm1, 152(%esp)
        incl    %eax
        cmpl    %eax, %esi
        jge     LBB1_15 # bb310
LBB1_16:        # bb358.loopexit
        movsd   152(%esp), %xmm0
        addsd   %xmm0, %xmm0
        addsd   LCPI1_22, %xmm0
        movsd   %xmm0, 152(%esp)

Rather than spilling the result of the last addsd in the loop, we should have
insert a copy to split the interval (one for the duration of the loop, one
extending to the fall through). The register pressure in the loop isn't high
enough to warrant the spill.

Also check why xmm7 is not used at all in the function.

//===---------------------------------------------------------------------===//

Legalize loses track of the fact that bools are always zero extended when in
memory.  This causes us to compile abort_gzip (from 164.gzip) from:

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@in_exit.4870.b = internal global i1 false		; <i1*> [#uses=2]
define fastcc void @abort_gzip() noreturn nounwind  {
entry:
	%tmp.b.i = load i1* @in_exit.4870.b		; <i1> [#uses=1]
	br i1 %tmp.b.i, label %bb.i, label %bb4.i
bb.i:		; preds = %entry
	tail call void @exit( i32 1 ) noreturn nounwind 
	unreachable
bb4.i:		; preds = %entry
	store i1 true, i1* @in_exit.4870.b
	tail call void @exit( i32 1 ) noreturn nounwind 
	unreachable
}
declare void @exit(i32) noreturn nounwind 

into:

_abort_gzip:
	subl	$12, %esp
	movb	_in_exit.4870.b, %al
	notb	%al
	testb	$1, %al
	jne	LBB1_2	## bb4.i
LBB1_1:	## bb.i
  ...

//===---------------------------------------------------------------------===//

We compile:

int test(int x, int y) {
  return x-y-1;
}

into (-m64):

_test:
	decl	%edi
	movl	%edi, %eax
	subl	%esi, %eax
	ret

it would be better to codegen as: x+~y  (notl+addl)

//===---------------------------------------------------------------------===//

This code:

int foo(const char *str,...)
{
 __builtin_va_list a; int x;
 __builtin_va_start(a,str); x = __builtin_va_arg(a,int); __builtin_va_end(a);
 return x;
}

gets compiled into this on x86-64:
	subq    $200, %rsp
        movaps  %xmm7, 160(%rsp)
        movaps  %xmm6, 144(%rsp)
        movaps  %xmm5, 128(%rsp)
        movaps  %xmm4, 112(%rsp)
        movaps  %xmm3, 96(%rsp)
        movaps  %xmm2, 80(%rsp)
        movaps  %xmm1, 64(%rsp)
        movaps  %xmm0, 48(%rsp)
        movq    %r9, 40(%rsp)
        movq    %r8, 32(%rsp)
        movq    %rcx, 24(%rsp)
        movq    %rdx, 16(%rsp)
        movq    %rsi, 8(%rsp)
        leaq    (%rsp), %rax
        movq    %rax, 192(%rsp)
        leaq    208(%rsp), %rax
        movq    %rax, 184(%rsp)
        movl    $48, 180(%rsp)
        movl    $8, 176(%rsp)
        movl    176(%rsp), %eax
        cmpl    $47, %eax
        jbe     .LBB1_3 # bb
.LBB1_1:        # bb3
        movq    184(%rsp), %rcx
        leaq    8(%rcx), %rax
        movq    %rax, 184(%rsp)
.LBB1_2:        # bb4
        movl    (%rcx), %eax
        addq    $200, %rsp
        ret
.LBB1_3:        # bb
        movl    %eax, %ecx
        addl    $8, %eax
        addq    192(%rsp), %rcx
        movl    %eax, 176(%rsp)
        jmp     .LBB1_2 # bb4

gcc 4.3 generates:
	subq    $96, %rsp
.LCFI0:
        leaq    104(%rsp), %rax
        movq    %rsi, -80(%rsp)
        movl    $8, -120(%rsp)
        movq    %rax, -112(%rsp)
        leaq    -88(%rsp), %rax
        movq    %rax, -104(%rsp)
        movl    $8, %eax
        cmpl    $48, %eax
        jb      .L6
        movq    -112(%rsp), %rdx
        movl    (%rdx), %eax
        addq    $96, %rsp
        ret
        .p2align 4,,10
        .p2align 3
.L6:
        mov     %eax, %edx
        addq    -104(%rsp), %rdx
        addl    $8, %eax
        movl    %eax, -120(%rsp)
        movl    (%rdx), %eax
        addq    $96, %rsp
        ret

and it gets compiled into this on x86:
	pushl   %ebp
        movl    %esp, %ebp
        subl    $4, %esp
        leal    12(%ebp), %eax
        movl    %eax, -4(%ebp)
        leal    16(%ebp), %eax
        movl    %eax, -4(%ebp)
        movl    12(%ebp), %eax
        addl    $4, %esp
        popl    %ebp
        ret

gcc 4.3 generates:
	pushl   %ebp
        movl    %esp, %ebp
        movl    12(%ebp), %eax
        popl    %ebp
        ret

//===---------------------------------------------------------------------===//

Teach tblgen not to check bitconvert source type in some cases. This allows us
to consolidate the following patterns in X86InstrMMX.td:

def : Pat<(v2i32 (bitconvert (i64 (vector_extract (v2i64 VR128:$src),
                                                  (iPTR 0))))),
          (v2i32 (MMX_MOVDQ2Qrr VR128:$src))>;
def : Pat<(v4i16 (bitconvert (i64 (vector_extract (v2i64 VR128:$src),
                                                  (iPTR 0))))),
          (v4i16 (MMX_MOVDQ2Qrr VR128:$src))>;
def : Pat<(v8i8 (bitconvert (i64 (vector_extract (v2i64 VR128:$src),
                                                  (iPTR 0))))),
          (v8i8 (MMX_MOVDQ2Qrr VR128:$src))>;

There are other cases in various td files.

//===---------------------------------------------------------------------===//

Take something like the following on x86-32:
unsigned a(unsigned long long x, unsigned y) {return x % y;}

We currently generate a libcall, but we really shouldn't: the expansion is
shorter and likely faster than the libcall.  The expected code is something
like the following:

	movl	12(%ebp), %eax
	movl	16(%ebp), %ecx
	xorl	%edx, %edx
	divl	%ecx
	movl	8(%ebp), %eax
	divl	%ecx
	movl	%edx, %eax
	ret

A similar code sequence works for division.

//===---------------------------------------------------------------------===//

These should compile to the same code, but the later codegen's to useless
instructions on X86. This may be a trivial dag combine (GCC PR7061):

struct s1 { unsigned char a, b; };
unsigned long f1(struct s1 x) {
    return x.a + x.b;
}
struct s2 { unsigned a: 8, b: 8; };
unsigned long f2(struct s2 x) {
    return x.a + x.b;
}

//===---------------------------------------------------------------------===//

