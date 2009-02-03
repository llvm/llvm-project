Target Independent Opportunities:

//===---------------------------------------------------------------------===//

With the recent changes to make the implicit def/use set explicit in
machineinstrs, we should change the target descriptions for 'call' instructions
so that the .td files don't list all the call-clobbered registers as implicit
defs.  Instead, these should be added by the code generator (e.g. on the dag).

This has a number of uses:

1. PPC32/64 and X86 32/64 can avoid having multiple copies of call instructions
   for their different impdef sets.
2. Targets with multiple calling convs (e.g. x86) which have different clobber
   sets don't need copies of call instructions.
3. 'Interprocedural register allocation' can be done to reduce the clobber sets
   of calls.

//===---------------------------------------------------------------------===//

Make the PPC branch selector target independant

//===---------------------------------------------------------------------===//

Get the C front-end to expand hypot(x,y) -> llvm.sqrt(x*x+y*y) when errno and
precision don't matter (ffastmath).  Misc/mandel will like this. :)  This isn't
safe in general, even on darwin.  See the libm implementation of hypot for
examples (which special case when x/y are exactly zero to get signed zeros etc
right).

//===---------------------------------------------------------------------===//

Solve this DAG isel folding deficiency:

int X, Y;

void fn1(void)
{
  X = X | (Y << 3);
}

compiles to

fn1:
	movl Y, %eax
	shll $3, %eax
	orl X, %eax
	movl %eax, X
	ret

The problem is the store's chain operand is not the load X but rather
a TokenFactor of the load X and load Y, which prevents the folding.

There are two ways to fix this:

1. The dag combiner can start using alias analysis to realize that y/x
   don't alias, making the store to X not dependent on the load from Y.
2. The generated isel could be made smarter in the case it can't
   disambiguate the pointers.

Number 1 is the preferred solution.

This has been "fixed" by a TableGen hack. But that is a short term workaround
which will be removed once the proper fix is made.

//===---------------------------------------------------------------------===//

On targets with expensive 64-bit multiply, we could LSR this:

for (i = ...; ++i) {
   x = 1ULL << i;

into:
 long long tmp = 1;
 for (i = ...; ++i, tmp+=tmp)
   x = tmp;

This would be a win on ppc32, but not x86 or ppc64.

//===---------------------------------------------------------------------===//

Shrink: (setlt (loadi32 P), 0) -> (setlt (loadi8 Phi), 0)

//===---------------------------------------------------------------------===//

Reassociate should turn: X*X*X*X -> t=(X*X) (t*t) to eliminate a multiply.

//===---------------------------------------------------------------------===//

Interesting? testcase for add/shift/mul reassoc:

int bar(int x, int y) {
  return x*x*x+y+x*x*x*x*x*y*y*y*y;
}
int foo(int z, int n) {
  return bar(z, n) + bar(2*z, 2*n);
}

Reassociate should handle the example in GCC PR16157.

//===---------------------------------------------------------------------===//

These two functions should generate the same code on big-endian systems:

int g(int *j,int *l)  {  return memcmp(j,l,4);  }
int h(int *j, int *l) {  return *j - *l; }

this could be done in SelectionDAGISel.cpp, along with other special cases,
for 1,2,4,8 bytes.

//===---------------------------------------------------------------------===//

It would be nice to revert this patch:
http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20060213/031986.html

And teach the dag combiner enough to simplify the code expanded before 
legalize.  It seems plausible that this knowledge would let it simplify other
stuff too.

//===---------------------------------------------------------------------===//

For vector types, TargetData.cpp::getTypeInfo() returns alignment that is equal
to the type size. It works but can be overly conservative as the alignment of
specific vector types are target dependent.

//===---------------------------------------------------------------------===//

We should add 'unaligned load/store' nodes, and produce them from code like
this:

v4sf example(float *P) {
  return (v4sf){P[0], P[1], P[2], P[3] };
}

//===---------------------------------------------------------------------===//

Add support for conditional increments, and other related patterns.  Instead
of:

	movl 136(%esp), %eax
	cmpl $0, %eax
	je LBB16_2	#cond_next
LBB16_1:	#cond_true
	incl _foo
LBB16_2:	#cond_next

emit:
	movl	_foo, %eax
	cmpl	$1, %edi
	sbbl	$-1, %eax
	movl	%eax, _foo

//===---------------------------------------------------------------------===//

Combine: a = sin(x), b = cos(x) into a,b = sincos(x).

Expand these to calls of sin/cos and stores:
      double sincos(double x, double *sin, double *cos);
      float sincosf(float x, float *sin, float *cos);
      long double sincosl(long double x, long double *sin, long double *cos);

Doing so could allow SROA of the destination pointers.  See also:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=17687

This is now easily doable with MRVs.  We could even make an intrinsic for this
if anyone cared enough about sincos.

//===---------------------------------------------------------------------===//

Scalar Repl cannot currently promote this testcase to 'ret long cst':

        %struct.X = type { i32, i32 }
        %struct.Y = type { %struct.X }

define i64 @bar() {
        %retval = alloca %struct.Y, align 8
        %tmp12 = getelementptr %struct.Y* %retval, i32 0, i32 0, i32 0
        store i32 0, i32* %tmp12
        %tmp15 = getelementptr %struct.Y* %retval, i32 0, i32 0, i32 1
        store i32 1, i32* %tmp15
        %retval.upgrd.1 = bitcast %struct.Y* %retval to i64*
        %retval.upgrd.2 = load i64* %retval.upgrd.1
        ret i64 %retval.upgrd.2
}

it should be extended to do so.

//===---------------------------------------------------------------------===//

-scalarrepl should promote this to be a vector scalar.

        %struct..0anon = type { <4 x float> }

define void @test1(<4 x float> %V, float* %P) {
        %u = alloca %struct..0anon, align 16
        %tmp = getelementptr %struct..0anon* %u, i32 0, i32 0
        store <4 x float> %V, <4 x float>* %tmp
        %tmp1 = bitcast %struct..0anon* %u to [4 x float]*
        %tmp.upgrd.1 = getelementptr [4 x float]* %tmp1, i32 0, i32 1
        %tmp.upgrd.2 = load float* %tmp.upgrd.1
        %tmp3 = mul float %tmp.upgrd.2, 2.000000e+00
        store float %tmp3, float* %P
        ret void
}

//===---------------------------------------------------------------------===//

Turn this into a single byte store with no load (the other 3 bytes are
unmodified):

void %test(uint* %P) {
	%tmp = load uint* %P
        %tmp14 = or uint %tmp, 3305111552
        %tmp15 = and uint %tmp14, 3321888767
        store uint %tmp15, uint* %P
        ret void
}

//===---------------------------------------------------------------------===//

dag/inst combine "clz(x)>>5 -> x==0" for 32-bit x.

Compile:

int bar(int x)
{
  int t = __builtin_clz(x);
  return -(t>>5);
}

to:

_bar:   addic r3,r3,-1
        subfe r3,r3,r3
        blr

//===---------------------------------------------------------------------===//

Legalize should lower ctlz like this:
  ctlz(x) = popcnt((x-1) & ~x)

on targets that have popcnt but not ctlz.  itanium, what else?

//===---------------------------------------------------------------------===//

quantum_sigma_x in 462.libquantum contains the following loop:

      for(i=0; i<reg->size; i++)
	{
	  /* Flip the target bit of each basis state */
	  reg->node[i].state ^= ((MAX_UNSIGNED) 1 << target);
	} 

Where MAX_UNSIGNED/state is a 64-bit int.  On a 32-bit platform it would be just
so cool to turn it into something like:

   long long Res = ((MAX_UNSIGNED) 1 << target);
   if (target < 32) {
     for(i=0; i<reg->size; i++)
       reg->node[i].state ^= Res & 0xFFFFFFFFULL;
   } else {
     for(i=0; i<reg->size; i++)
       reg->node[i].state ^= Res & 0xFFFFFFFF00000000ULL
   }
   
... which would only do one 32-bit XOR per loop iteration instead of two.

It would also be nice to recognize the reg->size doesn't alias reg->node[i], but
alas...

//===---------------------------------------------------------------------===//

This isn't recognized as bswap by instcombine (yes, it really is bswap):

unsigned long reverse(unsigned v) {
    unsigned t;
    t = v ^ ((v << 16) | (v >> 16));
    t &= ~0xff0000;
    v = (v << 24) | (v >> 8);
    return v ^ (t >> 8);
}

//===---------------------------------------------------------------------===//

These idioms should be recognized as popcount (see PR1488):

unsigned countbits_slow(unsigned v) {
  unsigned c;
  for (c = 0; v; v >>= 1)
    c += v & 1;
  return c;
}
unsigned countbits_fast(unsigned v){
  unsigned c;
  for (c = 0; v; c++)
    v &= v - 1; // clear the least significant bit set
  return c;
}

BITBOARD = unsigned long long
int PopCnt(register BITBOARD a) {
  register int c=0;
  while(a) {
    c++;
    a &= a - 1;
  }
  return c;
}
unsigned int popcount(unsigned int input) {
  unsigned int count = 0;
  for (unsigned int i =  0; i < 4 * 8; i++)
    count += (input >> i) & i;
  return count;
}

//===---------------------------------------------------------------------===//

These should turn into single 16-bit (unaligned?) loads on little/big endian
processors.

unsigned short read_16_le(const unsigned char *adr) {
  return adr[0] | (adr[1] << 8);
}
unsigned short read_16_be(const unsigned char *adr) {
  return (adr[0] << 8) | adr[1];
}

//===---------------------------------------------------------------------===//

-instcombine should handle this transform:
   icmp pred (sdiv X / C1 ), C2
when X, C1, and C2 are unsigned.  Similarly for udiv and signed operands. 

Currently InstCombine avoids this transform but will do it when the signs of
the operands and the sign of the divide match. See the FIXME in 
InstructionCombining.cpp in the visitSetCondInst method after the switch case 
for Instruction::UDiv (around line 4447) for more details.

The SingleSource/Benchmarks/Shootout-C++/hash and hash2 tests have examples of
this construct. 

//===---------------------------------------------------------------------===//

viterbi speeds up *significantly* if the various "history" related copy loops
are turned into memcpy calls at the source level.  We need a "loops to memcpy"
pass.

//===---------------------------------------------------------------------===//

Consider:

typedef unsigned U32;
typedef unsigned long long U64;
int test (U32 *inst, U64 *regs) {
    U64 effective_addr2;
    U32 temp = *inst;
    int r1 = (temp >> 20) & 0xf;
    int b2 = (temp >> 16) & 0xf;
    effective_addr2 = temp & 0xfff;
    if (b2) effective_addr2 += regs[b2];
    b2 = (temp >> 12) & 0xf;
    if (b2) effective_addr2 += regs[b2];
    effective_addr2 &= regs[4];
     if ((effective_addr2 & 3) == 0)
        return 1;
    return 0;
}

Note that only the low 2 bits of effective_addr2 are used.  On 32-bit systems,
we don't eliminate the computation of the top half of effective_addr2 because
we don't have whole-function selection dags.  On x86, this means we use one
extra register for the function when effective_addr2 is declared as U64 than
when it is declared U32.

//===---------------------------------------------------------------------===//

Promote for i32 bswap can use i64 bswap + shr.  Useful on targets with 64-bit
regs and bswap, like itanium.

//===---------------------------------------------------------------------===//

LSR should know what GPR types a target has.  This code:

volatile short X, Y; // globals

void foo(int N) {
  int i;
  for (i = 0; i < N; i++) { X = i; Y = i*4; }
}

produces two identical IV's (after promotion) on PPC/ARM:

LBB1_1: @bb.preheader
        mov r3, #0
        mov r2, r3
        mov r1, r3
LBB1_2: @bb
        ldr r12, LCPI1_0
        ldr r12, [r12]
        strh r2, [r12]
        ldr r12, LCPI1_1
        ldr r12, [r12]
        strh r3, [r12]
        add r1, r1, #1    <- [0,+,1]
        add r3, r3, #4
        add r2, r2, #1    <- [0,+,1]
        cmp r1, r0
        bne LBB1_2      @bb


//===---------------------------------------------------------------------===//

Tail call elim should be more aggressive, checking to see if the call is
followed by an uncond branch to an exit block.

; This testcase is due to tail-duplication not wanting to copy the return
; instruction into the terminating blocks because there was other code
; optimized out of the function after the taildup happened.
; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

define i32 @t4(i32 %a) {
entry:
	%tmp.1 = and i32 %a, 1		; <i32> [#uses=1]
	%tmp.2 = icmp ne i32 %tmp.1, 0		; <i1> [#uses=1]
	br i1 %tmp.2, label %then.0, label %else.0

then.0:		; preds = %entry
	%tmp.5 = add i32 %a, -1		; <i32> [#uses=1]
	%tmp.3 = call i32 @t4( i32 %tmp.5 )		; <i32> [#uses=1]
	br label %return

else.0:		; preds = %entry
	%tmp.7 = icmp ne i32 %a, 0		; <i1> [#uses=1]
	br i1 %tmp.7, label %then.1, label %return

then.1:		; preds = %else.0
	%tmp.11 = add i32 %a, -2		; <i32> [#uses=1]
	%tmp.9 = call i32 @t4( i32 %tmp.11 )		; <i32> [#uses=1]
	br label %return

return:		; preds = %then.1, %else.0, %then.0
	%result.0 = phi i32 [ 0, %else.0 ], [ %tmp.3, %then.0 ],
                            [ %tmp.9, %then.1 ]
	ret i32 %result.0
}

//===---------------------------------------------------------------------===//

Tail recursion elimination is not transforming this function, because it is
returning n, which fails the isDynamicConstant check in the accumulator 
recursion checks.

long long fib(const long long n) {
  switch(n) {
    case 0:
    case 1:
      return n;
    default:
      return fib(n-1) + fib(n-2);
  }
}

//===---------------------------------------------------------------------===//

Tail recursion elimination should handle:

int pow2m1(int n) {
 if (n == 0)
   return 0;
 return 2 * pow2m1 (n - 1) + 1;
}

Also, multiplies can be turned into SHL's, so they should be handled as if
they were associative.  "return foo() << 1" can be tail recursion eliminated.

//===---------------------------------------------------------------------===//

Argument promotion should promote arguments for recursive functions, like 
this:

; RUN: llvm-as < %s | opt -argpromotion | llvm-dis | grep x.val

define internal i32 @foo(i32* %x) {
entry:
	%tmp = load i32* %x		; <i32> [#uses=0]
	%tmp.foo = call i32 @foo( i32* %x )		; <i32> [#uses=1]
	ret i32 %tmp.foo
}

define i32 @bar(i32* %x) {
entry:
	%tmp3 = call i32 @foo( i32* %x )		; <i32> [#uses=1]
	ret i32 %tmp3
}

//===---------------------------------------------------------------------===//

"basicaa" should know how to look through "or" instructions that act like add
instructions.  For example in this code, the x*4+1 is turned into x*4 | 1, and
basicaa can't analyze the array subscript, leading to duplicated loads in the
generated code:

void test(int X, int Y, int a[]) {
int i;
  for (i=2; i<1000; i+=4) {
  a[i+0] = a[i-1+0]*a[i-2+0];
  a[i+1] = a[i-1+1]*a[i-2+1];
  a[i+2] = a[i-1+2]*a[i-2+2];
  a[i+3] = a[i-1+3]*a[i-2+3];
  }
}

BasicAA also doesn't do this for add.  It needs to know that &A[i+1] != &A[i].

//===---------------------------------------------------------------------===//

We should investigate an instruction sinking pass.  Consider this silly
example in pic mode:

#include <assert.h>
void foo(int x) {
  assert(x);
  //...
}

we compile this to:
_foo:
	subl	$28, %esp
	call	"L1$pb"
"L1$pb":
	popl	%eax
	cmpl	$0, 32(%esp)
	je	LBB1_2	# cond_true
LBB1_1:	# return
	# ...
	addl	$28, %esp
	ret
LBB1_2:	# cond_true
...

The PIC base computation (call+popl) is only used on one path through the 
code, but is currently always computed in the entry block.  It would be 
better to sink the picbase computation down into the block for the 
assertion, as it is the only one that uses it.  This happens for a lot of 
code with early outs.

Another example is loads of arguments, which are usually emitted into the 
entry block on targets like x86.  If not used in all paths through a 
function, they should be sunk into the ones that do.

In this case, whole-function-isel would also handle this.

//===---------------------------------------------------------------------===//

Investigate lowering of sparse switch statements into perfect hash tables:
http://burtleburtle.net/bob/hash/perfect.html

//===---------------------------------------------------------------------===//

We should turn things like "load+fabs+store" and "load+fneg+store" into the
corresponding integer operations.  On a yonah, this loop:

double a[256];
void foo() {
  int i, b;
  for (b = 0; b < 10000000; b++)
  for (i = 0; i < 256; i++)
    a[i] = -a[i];
}

is twice as slow as this loop:

long long a[256];
void foo() {
  int i, b;
  for (b = 0; b < 10000000; b++)
  for (i = 0; i < 256; i++)
    a[i] ^= (1ULL << 63);
}

and I suspect other processors are similar.  On X86 in particular this is a
big win because doing this with integers allows the use of read/modify/write
instructions.

//===---------------------------------------------------------------------===//

DAG Combiner should try to combine small loads into larger loads when 
profitable.  For example, we compile this C++ example:

struct THotKey { short Key; bool Control; bool Shift; bool Alt; };
extern THotKey m_HotKey;
THotKey GetHotKey () { return m_HotKey; }

into (-O3 -fno-exceptions -static -fomit-frame-pointer):

__Z9GetHotKeyv:
	pushl	%esi
	movl	8(%esp), %eax
	movb	_m_HotKey+3, %cl
	movb	_m_HotKey+4, %dl
	movb	_m_HotKey+2, %ch
	movw	_m_HotKey, %si
	movw	%si, (%eax)
	movb	%ch, 2(%eax)
	movb	%cl, 3(%eax)
	movb	%dl, 4(%eax)
	popl	%esi
	ret	$4

GCC produces:

__Z9GetHotKeyv:
	movl	_m_HotKey, %edx
	movl	4(%esp), %eax
	movl	%edx, (%eax)
	movzwl	_m_HotKey+4, %edx
	movw	%dx, 4(%eax)
	ret	$4

The LLVM IR contains the needed alignment info, so we should be able to 
merge the loads and stores into 4-byte loads:

	%struct.THotKey = type { i16, i8, i8, i8 }
define void @_Z9GetHotKeyv(%struct.THotKey* sret  %agg.result) nounwind  {
...
	%tmp2 = load i16* getelementptr (@m_HotKey, i32 0, i32 0), align 8
	%tmp5 = load i8* getelementptr (@m_HotKey, i32 0, i32 1), align 2
	%tmp8 = load i8* getelementptr (@m_HotKey, i32 0, i32 2), align 1
	%tmp11 = load i8* getelementptr (@m_HotKey, i32 0, i32 3), align 2

Alternatively, we should use a small amount of base-offset alias analysis
to make it so the scheduler doesn't need to hold all the loads in regs at
once.

//===---------------------------------------------------------------------===//

We should extend parameter attributes to capture more information about
pointer parameters for alias analysis.  Some ideas:

1. Add a "nocapture" attribute, which indicates that the callee does not store
   the address of the parameter into a global or any other memory location
   visible to the callee.  This can be used to make basicaa and other analyses
   more powerful.  It is true for things like memcpy, strcat, and many other
   things, including structs passed by value, most C++ references, etc.
2. Generalize readonly to be set on parameters.  This is important mod/ref 
   info for the function, which is important for basicaa and others.  It can
   also be used by the inliner to avoid inserting a memcpy for byval 
   arguments when the function is inlined.

These functions can be inferred by various analysis passes such as the 
globalsmodrefaa pass.  Note that getting #2 right is actually really tricky.
Consider this code:

struct S;  S G;
void caller(S byvalarg) { G.field = 1; ... }
void callee() { caller(G); }

The fact that the caller does not modify byval arg is not enough, we need
to know that it doesn't modify G either.  This is very tricky.

//===---------------------------------------------------------------------===//

We should add an FRINT node to the DAG to model targets that have legal
implementations of ceil/floor/rint.

//===---------------------------------------------------------------------===//

This GCC bug: http://gcc.gnu.org/bugzilla/show_bug.cgi?id=34043
contains a testcase that compiles down to:

	%struct.XMM128 = type { <4 x float> }
..
	%src = alloca %struct.XMM128
..
	%tmp6263 = bitcast %struct.XMM128* %src to <2 x i64>*
	%tmp65 = getelementptr %struct.XMM128* %src, i32 0, i32 0
	store <2 x i64> %tmp5899, <2 x i64>* %tmp6263, align 16
	%tmp66 = load <4 x float>* %tmp65, align 16		
	%tmp71 = add <4 x float> %tmp66, %tmp66		

If the mid-level optimizer turned the bitcast of pointer + store of tmp5899
into a bitcast of the vector value and a store to the pointer, then the 
store->load could be easily removed.

//===---------------------------------------------------------------------===//

Consider:

int test() {
  long long input[8] = {1,1,1,1,1,1,1,1};
  foo(input);
}

We currently compile this into a memcpy from a global array since the 
initializer is fairly large and not memset'able.  This is good, but the memcpy
gets lowered to load/stores in the code generator.  This is also ok, except
that the codegen lowering for memcpy doesn't handle the case when the source
is a constant global.  This gives us atrocious code like this:

	call	"L1$pb"
"L1$pb":
	popl	%eax
	movl	_C.0.1444-"L1$pb"+32(%eax), %ecx
	movl	%ecx, 40(%esp)
	movl	_C.0.1444-"L1$pb"+20(%eax), %ecx
	movl	%ecx, 28(%esp)
	movl	_C.0.1444-"L1$pb"+36(%eax), %ecx
	movl	%ecx, 44(%esp)
	movl	_C.0.1444-"L1$pb"+44(%eax), %ecx
	movl	%ecx, 52(%esp)
	movl	_C.0.1444-"L1$pb"+40(%eax), %ecx
	movl	%ecx, 48(%esp)
	movl	_C.0.1444-"L1$pb"+12(%eax), %ecx
	movl	%ecx, 20(%esp)
	movl	_C.0.1444-"L1$pb"+4(%eax), %ecx
...

instead of:
	movl	$1, 16(%esp)
	movl	$0, 20(%esp)
	movl	$1, 24(%esp)
	movl	$0, 28(%esp)
	movl	$1, 32(%esp)
	movl	$0, 36(%esp)
	...

//===---------------------------------------------------------------------===//

http://llvm.org/PR717:

The following code should compile into "ret int undef". Instead, LLVM
produces "ret int 0":

int f() {
  int x = 4;
  int y;
  if (x == 3) y = 0;
  return y;
}

//===---------------------------------------------------------------------===//

The loop unroller should partially unroll loops (instead of peeling them)
when code growth isn't too bad and when an unroll count allows simplification
of some code within the loop.  One trivial example is:

#include <stdio.h>
int main() {
    int nRet = 17;
    int nLoop;
    for ( nLoop = 0; nLoop < 1000; nLoop++ ) {
        if ( nLoop & 1 )
            nRet += 2;
        else
            nRet -= 1;
    }
    return nRet;
}

Unrolling by 2 would eliminate the '&1' in both copies, leading to a net
reduction in code size.  The resultant code would then also be suitable for
exit value computation.

//===---------------------------------------------------------------------===//

We miss a bunch of rotate opportunities on various targets, including ppc, x86,
etc.  On X86, we miss a bunch of 'rotate by variable' cases because the rotate
matching code in dag combine doesn't look through truncates aggressively 
enough.  Here are some testcases reduces from GCC PR17886:

unsigned long long f(unsigned long long x, int y) {
  return (x << y) | (x >> 64-y); 
} 
unsigned f2(unsigned x, int y){
  return (x << y) | (x >> 32-y); 
} 
unsigned long long f3(unsigned long long x){
  int y = 9;
  return (x << y) | (x >> 64-y); 
} 
unsigned f4(unsigned x){
  int y = 10;
  return (x << y) | (x >> 32-y); 
}
unsigned long long f5(unsigned long long x, unsigned long long y) {
  return (x << 8) | ((y >> 48) & 0xffull);
}
unsigned long long f6(unsigned long long x, unsigned long long y, int z) {
  switch(z) {
  case 1:
    return (x << 8) | ((y >> 48) & 0xffull);
  case 2:
    return (x << 16) | ((y >> 40) & 0xffffull);
  case 3:
    return (x << 24) | ((y >> 32) & 0xffffffull);
  case 4:
    return (x << 32) | ((y >> 24) & 0xffffffffull);
  default:
    return (x << 40) | ((y >> 16) & 0xffffffffffull);
  }
}

On X86-64, we only handle f2/f3/f4 right.  On x86-32, a few of these 
generate truly horrible code, instead of using shld and friends.  On
ARM, we end up with calls to L___lshrdi3/L___ashldi3 in f, which is
badness.  PPC64 misses f, f5 and f6.  CellSPU aborts in isel.

//===---------------------------------------------------------------------===//

We do a number of simplifications in simplify libcalls to strength reduce
standard library functions, but we don't currently merge them together.  For
example, it is useful to merge memcpy(a,b,strlen(b)) -> strcpy.  This can only
be done safely if "b" isn't modified between the strlen and memcpy of course.

//===---------------------------------------------------------------------===//

We should be able to evaluate this loop:

int test(int x_offs) {
  while (x_offs > 4)
     x_offs -= 4;
  return x_offs;
}

//===---------------------------------------------------------------------===//

Reassociate should turn things like:

int factorial(int X) {
 return X*X*X*X*X*X*X*X;
}

into llvm.powi calls, allowing the code generator to produce balanced
multiplication trees.

//===---------------------------------------------------------------------===//

We generate a horrible  libcall for llvm.powi.  For example, we compile:

#include <cmath>
double f(double a) { return std::pow(a, 4); }

into:

__Z1fd:
	subl	$12, %esp
	movsd	16(%esp), %xmm0
	movsd	%xmm0, (%esp)
	movl	$4, 8(%esp)
	call	L___powidf2$stub
	addl	$12, %esp
	ret

GCC produces:

__Z1fd:
	subl	$12, %esp
	movsd	16(%esp), %xmm0
	mulsd	%xmm0, %xmm0
	mulsd	%xmm0, %xmm0
	movsd	%xmm0, (%esp)
	fldl	(%esp)
	addl	$12, %esp
	ret

//===---------------------------------------------------------------------===//

We compile this program: (from GCC PR11680)
http://gcc.gnu.org/bugzilla/attachment.cgi?id=4487

Into code that runs the same speed in fast/slow modes, but both modes run 2x
slower than when compile with GCC (either 4.0 or 4.2):

$ llvm-g++ perf.cpp -O3 -fno-exceptions
$ time ./a.out fast
1.821u 0.003s 0:01.82 100.0%	0+0k 0+0io 0pf+0w

$ g++ perf.cpp -O3 -fno-exceptions
$ time ./a.out fast
0.821u 0.001s 0:00.82 100.0%	0+0k 0+0io 0pf+0w

It looks like we are making the same inlining decisions, so this may be raw
codegen badness or something else (haven't investigated).

//===---------------------------------------------------------------------===//

We miss some instcombines for stuff like this:
void bar (void);
void foo (unsigned int a) {
  /* This one is equivalent to a >= (3 << 2).  */
  if ((a >> 2) >= 3)
    bar ();
}

A few other related ones are in GCC PR14753.

//===---------------------------------------------------------------------===//

Divisibility by constant can be simplified (according to GCC PR12849) from
being a mulhi to being a mul lo (cheaper).  Testcase:

void bar(unsigned n) {
  if (n % 3 == 0)
    true();
}

I think this basically amounts to a dag combine to simplify comparisons against
multiply hi's into a comparison against the mullo.

//===---------------------------------------------------------------------===//

SROA is not promoting the union on the stack in this example, we should end
up with no allocas.

union vec2d {
    double e[2];
    double v __attribute__((vector_size(16)));
};
typedef union vec2d vec2d;

static vec2d a={{1,2}}, b={{3,4}};
    
vec2d foo () {
    return (vec2d){ .v = a.v + b.v * (vec2d){{5,5}}.v };
}

//===---------------------------------------------------------------------===//

Better mod/ref analysis for scanf would allow us to eliminate the vtable and a
bunch of other stuff from this example (see PR1604): 

#include <cstdio>
struct test {
    int val;
    virtual ~test() {}
};

int main() {
    test t;
    std::scanf("%d", &t.val);
    std::printf("%d\n", t.val);
}

//===---------------------------------------------------------------------===//

Instcombine will merge comparisons like (x >= 10) && (x < 20) by producing (x -
10) u< 10, but only when the comparisons have matching sign.

This could be converted with a similiar technique. (PR1941)

define i1 @test(i8 %x) {
  %A = icmp uge i8 %x, 5
  %B = icmp slt i8 %x, 20
  %C = and i1 %A, %B
  ret i1 %C
}

//===---------------------------------------------------------------------===//

These functions perform the same computation, but produce different assembly.

define i8 @select(i8 %x) readnone nounwind {
  %A = icmp ult i8 %x, 250
  %B = select i1 %A, i8 0, i8 1
  ret i8 %B 
}

define i8 @addshr(i8 %x) readnone nounwind {
  %A = zext i8 %x to i9
  %B = add i9 %A, 6       ;; 256 - 250 == 6
  %C = lshr i9 %B, 8
  %D = trunc i9 %C to i8
  ret i8 %D
}

//===---------------------------------------------------------------------===//

From gcc bug 24696:
int
f (unsigned long a, unsigned long b, unsigned long c)
{
  return ((a & (c - 1)) != 0) || ((b & (c - 1)) != 0);
}
int
f (unsigned long a, unsigned long b, unsigned long c)
{
  return ((a & (c - 1)) != 0) | ((b & (c - 1)) != 0);
}
Both should combine to ((a|b) & (c-1)) != 0.  Currently not optimized with
"clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

From GCC Bug 20192:
#define PMD_MASK    (~((1UL << 23) - 1))
void clear_pmd_range(unsigned long start, unsigned long end)
{
   if (!(start & ~PMD_MASK) && !(end & ~PMD_MASK))
       f();
}
The expression should optimize to something like
"!((start|end)&~PMD_MASK). Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

From GCC Bug 15241:
unsigned int
foo (unsigned int a, unsigned int b)
{
 if (a <= 7 && b <= 7)
   baz ();
}
Should combine to "(a|b) <= 7".  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

From GCC Bug 3756:
int
pn (int n)
{
 return (n >= 0 ? 1 : -1);
}
Should combine to (n >> 31) | 1.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts | llc".

//===---------------------------------------------------------------------===//

From GCC Bug 28685:
int test(int a, int b)
{
 int lt = a < b;
 int eq = a == b;

 return (lt || eq);
}
Should combine to "a <= b".  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts | llc".

//===---------------------------------------------------------------------===//

void a(int variable)
{
 if (variable == 4 || variable == 6)
   bar();
}
This should optimize to "if ((variable | 2) == 6)".  Currently not
optimized with "clang -emit-llvm-bc | opt -std-compile-opts | llc".

//===---------------------------------------------------------------------===//

unsigned int f(unsigned int i, unsigned int n) {++i; if (i == n) ++i; return
i;}
unsigned int f2(unsigned int i, unsigned int n) {++i; i += i == n; return i;}
These should combine to the same thing.  Currently, the first function
produces better code on X86.

//===---------------------------------------------------------------------===//

From GCC Bug 15784:
#define abs(x) x>0?x:-x
int f(int x, int y)
{
 return (abs(x)) >= 0;
}
This should optimize to x == INT_MIN. (With -fwrapv.)  Currently not
optimized with "clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

From GCC Bug 14753:
void
rotate_cst (unsigned int a)
{
 a = (a << 10) | (a >> 22);
 if (a == 123)
   bar ();
}
void
minus_cst (unsigned int a)
{
 unsigned int tem;

 tem = 20 - a;
 if (tem == 5)
   bar ();
}
void
mask_gt (unsigned int a)
{
 /* This is equivalent to a > 15.  */
 if ((a & ~7) > 8)
   bar ();
}
void
rshift_gt (unsigned int a)
{
 /* This is equivalent to a > 23.  */
 if ((a >> 2) > 5)
   bar ();
}
All should simplify to a single comparison.  All of these are
currently not optimized with "clang -emit-llvm-bc | opt
-std-compile-opts".

//===---------------------------------------------------------------------===//

From GCC Bug 32605:
int c(int* x) {return (char*)x+2 == (char*)x;}
Should combine to 0.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts" (although llc can optimize it).

//===---------------------------------------------------------------------===//

int a(unsigned char* b) {return *b > 99;}
There's an unnecessary zext in the generated code with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(unsigned b) {return ((b << 31) | (b << 30)) >> 31;}
Should be combined to  "((b >> 1) | b) & 1".  Currently not optimized
with "clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

unsigned a(unsigned x, unsigned y) { return x | (y & 1) | (y & 2);}
Should combine to "x | (y & 3)".  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

unsigned a(unsigned a) {return ((a | 1) & 3) | (a & -4);}
Should combine to "a | 1".  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int a, int b, int c) {return (~a & c) | ((c|a) & b);}
Should fold to "(~a & c) | (a & b)".  Currently not optimized with
"clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int a,int b) {return (~(a|b))|a;}
Should fold to "a|~b".  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int a, int b) {return (a&&b) || (a&&!b);}
Should fold to "a".  Currently not optimized with "clang -emit-llvm-bc
| opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int a, int b, int c) {return (a&&b) || (!a&&c);}
Should fold to "a ? b : c", or at least something sane.  Currently not
optimized with "clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int a, int b, int c) {return (a&&b) || (a&&c) || (a&&b&&c);}
Should fold to a && (b || c).  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int x) {return x | ((x & 8) ^ 8);}
Should combine to x | 8.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int x) {return x ^ ((x & 8) ^ 8);}
Should also combine to x | 8.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int x) {return (x & 8) == 0 ? -1 : -9;}
Should combine to (x | -9) ^ 8.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int x) {return (x & 8) == 0 ? -9 : -1;}
Should combine to x | -9.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

int a(int x) {return ((x | -9) ^ 8) & x;}
Should combine to x & -9.  Currently not optimized with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

unsigned a(unsigned a) {return a * 0x11111111 >> 28 & 1;}
Should combine to "a * 0x88888888 >> 31".  Currently not optimized
with "clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

unsigned a(char* x) {if ((*x & 32) == 0) return b();}
There's an unnecessary zext in the generated code with "clang
-emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

unsigned a(unsigned long long x) {return 40 * (x >> 1);}
Should combine to "20 * (((unsigned)x) & -2)".  Currently not
optimized with "clang -emit-llvm-bc | opt -std-compile-opts".

//===---------------------------------------------------------------------===//

We would like to do the following transform in the instcombiner:

  -X/C -> X/-C

However, this isn't valid if (-X) overflows. We can implement this when we
have the concept of a "C signed subtraction" operator that which is undefined
on overflow.

//===---------------------------------------------------------------------===//

This was noticed in the entryblock for grokdeclarator in 403.gcc:

        %tmp = icmp eq i32 %decl_context, 4          
        %decl_context_addr.0 = select i1 %tmp, i32 3, i32 %decl_context 
        %tmp1 = icmp eq i32 %decl_context_addr.0, 1 
        %decl_context_addr.1 = select i1 %tmp1, i32 0, i32 %decl_context_addr.0

tmp1 should be simplified to something like:
  (!tmp || decl_context == 1)

This allows recursive simplifications, tmp1 is used all over the place in
the function, e.g. by:

        %tmp23 = icmp eq i32 %decl_context_addr.1, 0            ; <i1> [#uses=1]
        %tmp24 = xor i1 %tmp1, true             ; <i1> [#uses=1]
        %or.cond8 = and i1 %tmp23, %tmp24               ; <i1> [#uses=1]

later.

//===---------------------------------------------------------------------===//

Store sinking: This code:

void f (int n, int *cond, int *res) {
    int i;
    *res = 0;
    for (i = 0; i < n; i++)
        if (*cond)
            *res ^= 234; /* (*) */
}

On this function GVN hoists the fully redundant value of *res, but nothing
moves the store out.  This gives us this code:

bb:		; preds = %bb2, %entry
	%.rle = phi i32 [ 0, %entry ], [ %.rle6, %bb2 ]	
	%i.05 = phi i32 [ 0, %entry ], [ %indvar.next, %bb2 ]
	%1 = load i32* %cond, align 4
	%2 = icmp eq i32 %1, 0
	br i1 %2, label %bb2, label %bb1

bb1:		; preds = %bb
	%3 = xor i32 %.rle, 234	
	store i32 %3, i32* %res, align 4
	br label %bb2

bb2:		; preds = %bb, %bb1
	%.rle6 = phi i32 [ %3, %bb1 ], [ %.rle, %bb ]	
	%indvar.next = add i32 %i.05, 1	
	%exitcond = icmp eq i32 %indvar.next, %n
	br i1 %exitcond, label %return, label %bb

DSE should sink partially dead stores to get the store out of the loop.

Here's another partial dead case:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=12395

//===---------------------------------------------------------------------===//

Scalar PRE hoists the mul in the common block up to the else:

int test (int a, int b, int c, int g) {
  int d, e;
  if (a)
    d = b * c;
  else
    d = b - c;
  e = b * c + g;
  return d + e;
}

It would be better to do the mul once to reduce codesize above the if.
This is GCC PR38204.

//===---------------------------------------------------------------------===//

GCC PR37810 is an interesting case where we should sink load/store reload
into the if block and outside the loop, so we don't reload/store it on the
non-call path.

for () {
  *P += 1;
  if ()
    call();
  else
    ...
->
tmp = *P
for () {
  tmp += 1;
  if () {
    *P = tmp;
    call();
    tmp = *P;
  } else ...
}
*P = tmp;

We now hoist the reload after the call (Transforms/GVN/lpre-call-wrap.ll), but
we don't sink the store.  We need partially dead store sinking.

//===---------------------------------------------------------------------===//

[PHI TRANSLATE GEPs]

GCC PR37166: Sinking of loads prevents SROA'ing the "g" struct on the stack
leading to excess stack traffic. This could be handled by GVN with some crazy
symbolic phi translation.  The code we get looks like (g is on the stack):

bb2:		; preds = %bb1
..
	%9 = getelementptr %struct.f* %g, i32 0, i32 0		
	store i32 %8, i32* %9, align  bel %bb3

bb3:		; preds = %bb1, %bb2, %bb
	%c_addr.0 = phi %struct.f* [ %g, %bb2 ], [ %c, %bb ], [ %c, %bb1 ]
	%b_addr.0 = phi %struct.f* [ %b, %bb2 ], [ %g, %bb ], [ %b, %bb1 ]
	%10 = getelementptr %struct.f* %c_addr.0, i32 0, i32 0
	%11 = load i32* %10, align 4

%11 is fully redundant, an in BB2 it should have the value %8.

GCC PR33344 is a similar case.

//===---------------------------------------------------------------------===//

There are many load PRE testcases in testsuite/gcc.dg/tree-ssa/loadpre* in the
GCC testsuite.  There are many pre testcases as ssa-pre-*.c

//===---------------------------------------------------------------------===//

There are some interesting cases in testsuite/gcc.dg/tree-ssa/pred-comm* in the
GCC testsuite.  For example, predcom-1.c is:

 for (i = 2; i < 1000; i++)
    fib[i] = (fib[i-1] + fib[i - 2]) & 0xffff;

which compiles into:

bb1:		; preds = %bb1, %bb1.thread
	%indvar = phi i32 [ 0, %bb1.thread ], [ %0, %bb1 ]	
	%i.0.reg2mem.0 = add i32 %indvar, 2		
	%0 = add i32 %indvar, 1		; <i32> [#uses=3]
	%1 = getelementptr [1000 x i32]* @fib, i32 0, i32 %0		
	%2 = load i32* %1, align 4		; <i32> [#uses=1]
	%3 = getelementptr [1000 x i32]* @fib, i32 0, i32 %indvar	
	%4 = load i32* %3, align 4		; <i32> [#uses=1]
	%5 = add i32 %4, %2		; <i32> [#uses=1]
	%6 = and i32 %5, 65535		; <i32> [#uses=1]
	%7 = getelementptr [1000 x i32]* @fib, i32 0, i32 %i.0.reg2mem.0
	store i32 %6, i32* %7, align 4
	%exitcond = icmp eq i32 %0, 998		; <i1> [#uses=1]
	br i1 %exitcond, label %return, label %bb1

This is basically:
  LOAD fib[i+1]
  LOAD fib[i]
  STORE fib[i+2]

instead of handling this as a loop or other xform, all we'd need to do is teach
load PRE to phi translate the %0 add (i+1) into the predecessor as (i'+1+1) =
(i'+2) (where i' is the previous iteration of i).  This would find the store
which feeds it.

predcom-2.c is apparently the same as predcom-1.c
predcom-3.c is very similar but needs loads feeding each other instead of
store->load.
predcom-4.c seems the same as the rest.


//===---------------------------------------------------------------------===//

Other simple load PRE cases:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=35287 [LPRE crit edge splitting]

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=34677 (licm does this, LPRE crit edge)
  llvm-gcc t2.c -S -o - -O0 -emit-llvm | llvm-as | opt -mem2reg -simplifycfg -gvn | llvm-dis

//===---------------------------------------------------------------------===//

Type based alias analysis:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=14705

//===---------------------------------------------------------------------===//

When GVN/PRE finds a store of float* to a must aliases pointer when expecting
an int*, it should turn it into a bitcast.  This is a nice generalization of
the SROA hack that would apply to other cases, e.g.:

int foo(int C, int *P, float X) {
  if (C) {
    bar();
    *P = 42;
  } else
    *(float*)P = X;

   return *P;
}


One example (that requires crazy phi translation) is:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=16799 [BITCAST PHI TRANS]

//===---------------------------------------------------------------------===//

A/B get pinned to the stack because we turn an if/then into a select instead
of PRE'ing the load/store.  This may be fixable in instcombine:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=37892



Interesting missed case because of control flow flattening (should be 2 loads):
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=26629
With: llvm-gcc t2.c -S -o - -O0 -emit-llvm | llvm-as | 
             opt -mem2reg -gvn -instcombine | llvm-dis
we miss it because we need 1) GEP PHI TRAN, 2) CRIT EDGE 3) MULTIPLE DIFFERENT
VALS PRODUCED BY ONE BLOCK OVER DIFFERENT PATHS

//===---------------------------------------------------------------------===//

http://gcc.gnu.org/bugzilla/show_bug.cgi?id=19633
We could eliminate the branch condition here, loading from null is undefined:

struct S { int w, x, y, z; };
struct T { int r; struct S s; };
void bar (struct S, int);
void foo (int a, struct T b)
{
  struct S *c = 0;
  if (a)
    c = &b.s;
  bar (*c, a);
}

//===---------------------------------------------------------------------===//

simplifylibcalls should do several optimizations for strspn/strcspn:

strcspn(x, "") -> strlen(x)
strcspn("", x) -> 0
strspn("", x) -> 0
strspn(x, "") -> strlen(x)
strspn(x, "a") -> strchr(x, 'a')-x

strcspn(x, "a") -> inlined loop for up to 3 letters (similarly for strspn):

size_t __strcspn_c3 (__const char *__s, int __reject1, int __reject2,
                     int __reject3) {
  register size_t __result = 0;
  while (__s[__result] != '\0' && __s[__result] != __reject1 &&
         __s[__result] != __reject2 && __s[__result] != __reject3)
    ++__result;
  return __result;
}

This should turn into a switch on the character.  See PR3253 for some notes on
codegen.

456.hmmer apparently uses strcspn and strspn a lot.  471.omnetpp uses strspn.

//===---------------------------------------------------------------------===//

"gas" uses this idiom:
  else if (strchr ("+-/*%|&^:[]()~", *intel_parser.op_string))
..
  else if (strchr ("<>", *intel_parser.op_string)

Those should be turned into a switch.

//===---------------------------------------------------------------------===//

252.eon contains this interesting code:

        %3072 = getelementptr [100 x i8]* %tempString, i32 0, i32 0
        %3073 = call i8* @strcpy(i8* %3072, i8* %3071) nounwind
        %strlen = call i32 @strlen(i8* %3072)    ; uses = 1
        %endptr = getelementptr [100 x i8]* %tempString, i32 0, i32 %strlen
        call void @llvm.memcpy.i32(i8* %endptr, 
          i8* getelementptr ([5 x i8]* @"\01LC42", i32 0, i32 0), i32 5, i32 1)
        %3074 = call i32 @strlen(i8* %endptr) nounwind readonly 
        
This is interesting for a couple reasons.  First, in this:

        %3073 = call i8* @strcpy(i8* %3072, i8* %3071) nounwind
        %strlen = call i32 @strlen(i8* %3072)  

The strlen could be replaced with: %strlen = sub %3072, %3073, because the
strcpy call returns a pointer to the end of the string.  Based on that, the
endptr GEP just becomes equal to 3073, which eliminates a strlen call and GEP.

Second, the memcpy+strlen strlen can be replaced with:

        %3074 = call i32 @strlen([5 x i8]* @"\01LC42") nounwind readonly 

Because the destination was just copied into the specified memory buffer.  This,
in turn, can be constant folded to "4".

In other code, it contains:

        %endptr6978 = bitcast i8* %endptr69 to i32*            
        store i32 7107374, i32* %endptr6978, align 1
        %3167 = call i32 @strlen(i8* %endptr69) nounwind readonly    

Which could also be constant folded.  Whatever is producing this should probably
be fixed to leave this as a memcpy from a string.

Further, eon also has an interesting partially redundant strlen call:

bb8:            ; preds = %_ZN18eonImageCalculatorC1Ev.exit
        %682 = getelementptr i8** %argv, i32 6          ; <i8**> [#uses=2]
        %683 = load i8** %682, align 4          ; <i8*> [#uses=4]
        %684 = load i8* %683, align 1           ; <i8> [#uses=1]
        %685 = icmp eq i8 %684, 0               ; <i1> [#uses=1]
        br i1 %685, label %bb10, label %bb9

bb9:            ; preds = %bb8
        %686 = call i32 @strlen(i8* %683) nounwind readonly          
        %687 = icmp ugt i32 %686, 254           ; <i1> [#uses=1]
        br i1 %687, label %bb10, label %bb11

bb10:           ; preds = %bb9, %bb8
        %688 = call i32 @strlen(i8* %683) nounwind readonly          

This could be eliminated by doing the strlen once in bb8, saving code size and
improving perf on the bb8->9->10 path.

//===---------------------------------------------------------------------===//

I see an interesting fully redundant call to strlen left in 186.crafty:InputMove
which looks like:
       %movetext11 = getelementptr [128 x i8]* %movetext, i32 0, i32 0 
 

bb62:           ; preds = %bb55, %bb53
        %promote.0 = phi i32 [ %169, %bb55 ], [ 0, %bb53 ]             
        %171 = call i32 @strlen(i8* %movetext11) nounwind readonly align 1
        %172 = add i32 %171, -1         ; <i32> [#uses=1]
        %173 = getelementptr [128 x i8]* %movetext, i32 0, i32 %172       

...  no stores ...
       br i1 %or.cond, label %bb65, label %bb72

bb65:           ; preds = %bb62
        store i8 0, i8* %173, align 1
        br label %bb72

bb72:           ; preds = %bb65, %bb62
        %trank.1 = phi i32 [ %176, %bb65 ], [ -1, %bb62 ]            
        %177 = call i32 @strlen(i8* %movetext11) nounwind readonly align 1

Note that on the bb62->bb72 path, that the %177 strlen call is partially
redundant with the %171 call.  At worst, we could shove the %177 strlen call
up into the bb65 block moving it out of the bb62->bb72 path.   However, note
that bb65 stores to the string, zeroing out the last byte.  This means that on
that path the value of %177 is actually just %171-1.  A sub is cheaper than a
strlen!

This pattern repeats several times, basically doing:

  A = strlen(P);
  P[A-1] = 0;
  B = strlen(P);
  where it is "obvious" that B = A-1.

//===---------------------------------------------------------------------===//

186.crafty contains this interesting pattern:

%77 = call i8* @strstr(i8* getelementptr ([6 x i8]* @"\01LC5", i32 0, i32 0),
                       i8* %30)
%phitmp648 = icmp eq i8* %77, getelementptr ([6 x i8]* @"\01LC5", i32 0, i32 0)
br i1 %phitmp648, label %bb70, label %bb76

bb70:           ; preds = %OptionMatch.exit91, %bb69
        %78 = call i32 @strlen(i8* %30) nounwind readonly align 1               ; <i32> [#uses=1]

This is basically:
  cststr = "abcdef";
  if (strstr(cststr, P) == cststr) {
     x = strlen(P);
     ...

The strstr call would be significantly cheaper written as:

cststr = "abcdef";
if (memcmp(P, str, strlen(P)))
  x = strlen(P);

This is memcmp+strlen instead of strstr.  This also makes the strlen fully
redundant.

//===---------------------------------------------------------------------===//

186.crafty also contains this code:

%1906 = call i32 @strlen(i8* getelementptr ([32 x i8]* @pgn_event, i32 0,i32 0))
%1907 = getelementptr [32 x i8]* @pgn_event, i32 0, i32 %1906
%1908 = call i8* @strcpy(i8* %1907, i8* %1905) nounwind align 1
%1909 = call i32 @strlen(i8* getelementptr ([32 x i8]* @pgn_event, i32 0,i32 0))
%1910 = getelementptr [32 x i8]* @pgn_event, i32 0, i32 %1909         

The last strlen is computable as 1908-@pgn_event, which means 1910=1908.

//===---------------------------------------------------------------------===//

186.crafty has this interesting pattern with the "out.4543" variable:

call void @llvm.memcpy.i32(
        i8* getelementptr ([10 x i8]* @out.4543, i32 0, i32 0),
       i8* getelementptr ([7 x i8]* @"\01LC28700", i32 0, i32 0), i32 7, i32 1) 
%101 = call@printf(i8* ...   @out.4543, i32 0, i32 0)) nounwind 

It is basically doing:

  memcpy(globalarray, "string");
  printf(...,  globalarray);
  
Anyway, by knowing that printf just reads the memory and forward substituting
the string directly into the printf, this eliminates reads from globalarray.
Since this pattern occurs frequently in crafty (due to the "DisplayTime" and
other similar functions) there are many stores to "out".  Once all the printfs
stop using "out", all that is left is the memcpy's into it.  This should allow
globalopt to remove the "stored only" global.

//===---------------------------------------------------------------------===//

This code:

define inreg i32 @foo(i8* inreg %p) nounwind {
  %tmp0 = load i8* %p
  %tmp1 = ashr i8 %tmp0, 5
  %tmp2 = sext i8 %tmp1 to i32
  ret i32 %tmp2
}

could be dagcombine'd to a sign-extending load with a shift.
For example, on x86 this currently gets this:

	movb	(%eax), %al
	sarb	$5, %al
	movsbl	%al, %eax

while it could get this:

	movsbl	(%eax), %eax
	sarl	$5, %eax

//===---------------------------------------------------------------------===//

GCC PR31029:

int test(int x) { return 1-x == x; }     // --> return false
int test2(int x) { return 2-x == x; }    // --> return x == 1 ?

Always foldable for odd constants, what is the rule for even?

//===---------------------------------------------------------------------===//

PR 3381: GEP to field of size 0 inside a struct could be turned into GEP
for next field in struct (which is at same address).

For example: store of float into { {{}}, float } could be turned into a store to
the float directly.

