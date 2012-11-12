//===- README.txt - Notes for improving PowerPC-specific code gen ---------===//

TODO:
* gpr0 allocation
* lmw/stmw pass a la arm load store optimizer for prolog/epilog

===-------------------------------------------------------------------------===

On PPC64, this:

long f2 (long x) { return 0xfffffff000000000UL; }
long f3 (long x) { return 0x1ffffffffUL; }

could compile into:

_f2:
	li r3,-1
	rldicr r3,r3,0,27
	blr
_f3:
	li r3,-1
	rldicl r3,r3,0,31
	blr

we produce:

_f2:
	lis r2, 4095
	ori r2, r2, 65535
	sldi r3, r2, 36
	blr 
_f3:
	li r2, 1
	sldi r2, r2, 32
	oris r2, r2, 65535
	ori r3, r2, 65535
	blr 

===-------------------------------------------------------------------------===

This code:

unsigned add32carry(unsigned sum, unsigned x) {
 unsigned z = sum + x;
 if (sum + x < x)
     z++;
 return z;
}

Should compile to something like:

	addc r3,r3,r4
	addze r3,r3

instead we get:

	add r3, r4, r3
	cmplw cr7, r3, r4
	mfcr r4 ; 1
	rlwinm r4, r4, 29, 31, 31
	add r3, r3, r4

Ick.

===-------------------------------------------------------------------------===

Support 'update' load/store instructions.  These are cracked on the G5, but are
still a codesize win.

With preinc enabled, this:

long *%test4(long *%X, long *%dest) {
        %Y = getelementptr long* %X, int 4
        %A = load long* %Y
        store long %A, long* %dest
        ret long* %Y
}

compiles to:

_test4:
        mr r2, r3
        lwzu r5, 32(r2)
        lwz r3, 36(r3)
        stw r5, 0(r4)
        stw r3, 4(r4)
        mr r3, r2
        blr 

with -sched=list-burr, I get:

_test4:
        lwz r2, 36(r3)
        lwzu r5, 32(r3)
        stw r2, 4(r4)
        stw r5, 0(r4)
        blr 

===-------------------------------------------------------------------------===

We compile the hottest inner loop of viterbi to:

        li r6, 0
        b LBB1_84       ;bb432.i
LBB1_83:        ;bb420.i
        lbzx r8, r5, r7
        addi r6, r7, 1
        stbx r8, r4, r7
LBB1_84:        ;bb432.i
        mr r7, r6
        cmplwi cr0, r7, 143
        bne cr0, LBB1_83        ;bb420.i

The CBE manages to produce:

	li r0, 143
	mtctr r0
loop:
	lbzx r2, r2, r11
	stbx r0, r2, r9
	addi r2, r2, 1
	bdz later
	b loop

This could be much better (bdnz instead of bdz) but it still beats us.  If we
produced this with bdnz, the loop would be a single dispatch group.

===-------------------------------------------------------------------------===

Compile:

void foo(int *P) {
 if (P)  *P = 0;
}

into:

_foo:
        cmpwi cr0,r3,0
        beqlr cr0
        li r0,0
        stw r0,0(r3)
        blr

This is effectively a simple form of predication.

===-------------------------------------------------------------------------===

Lump the constant pool for each function into ONE pic object, and reference
pieces of it as offsets from the start.  For functions like this (contrived
to have lots of constants obviously):

double X(double Y) { return (Y*1.23 + 4.512)*2.34 + 14.38; }

We generate:

_X:
        lis r2, ha16(.CPI_X_0)
        lfd f0, lo16(.CPI_X_0)(r2)
        lis r2, ha16(.CPI_X_1)
        lfd f2, lo16(.CPI_X_1)(r2)
        fmadd f0, f1, f0, f2
        lis r2, ha16(.CPI_X_2)
        lfd f1, lo16(.CPI_X_2)(r2)
        lis r2, ha16(.CPI_X_3)
        lfd f2, lo16(.CPI_X_3)(r2)
        fmadd f1, f0, f1, f2
        blr

It would be better to materialize .CPI_X into a register, then use immediates
off of the register to avoid the lis's.  This is even more important in PIC 
mode.

Note that this (and the static variable version) is discussed here for GCC:
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg00133.html

Here's another example (the sgn function):
double testf(double a) {
       return a == 0.0 ? 0.0 : (a > 0.0 ? 1.0 : -1.0);
}

it produces a BB like this:
LBB1_1: ; cond_true
        lis r2, ha16(LCPI1_0)
        lfs f0, lo16(LCPI1_0)(r2)
        lis r2, ha16(LCPI1_1)
        lis r3, ha16(LCPI1_2)
        lfs f2, lo16(LCPI1_2)(r3)
        lfs f3, lo16(LCPI1_1)(r2)
        fsub f0, f0, f1
        fsel f1, f0, f2, f3
        blr 

===-------------------------------------------------------------------------===

PIC Code Gen IPO optimization:

Squish small scalar globals together into a single global struct, allowing the 
address of the struct to be CSE'd, avoiding PIC accesses (also reduces the size
of the GOT on targets with one).

Note that this is discussed here for GCC:
http://gcc.gnu.org/ml/gcc-patches/2006-02/msg00133.html

===-------------------------------------------------------------------------===

Implement Newton-Rhapson method for improving estimate instructions to the
correct accuracy, and implementing divide as multiply by reciprocal when it has
more than one use.  Itanium would want this too.

===-------------------------------------------------------------------------===

Compile offsets from allocas:

int *%test() {
        %X = alloca { int, int }
        %Y = getelementptr {int,int}* %X, int 0, uint 1
        ret int* %Y
}

into a single add, not two:

_test:
        addi r2, r1, -8
        addi r3, r2, 4
        blr

--> important for C++.

===-------------------------------------------------------------------------===

No loads or stores of the constants should be needed:

struct foo { double X, Y; };
void xxx(struct foo F);
void bar() { struct foo R = { 1.0, 2.0 }; xxx(R); }

===-------------------------------------------------------------------------===

Darwin Stub removal:

We still generate calls to foo$stub, and stubs, on Darwin.  This is not
necessary when building with the Leopard (10.5) or later linker, as stubs are
generated by ld when necessary.  Parameterizing this based on the deployment
target (-mmacosx-version-min) is probably enough.  x86-32 does this right, see
its logic.

===-------------------------------------------------------------------------===

Darwin Stub LICM optimization:

Loops like this:
  
  for (...)  bar();

Have to go through an indirect stub if bar is external or linkonce.  It would 
be better to compile it as:

     fp = &bar;
     for (...)  fp();

which only computes the address of bar once (instead of each time through the 
stub).  This is Darwin specific and would have to be done in the code generator.
Probably not a win on x86.

===-------------------------------------------------------------------------===

Simple IPO for argument passing, change:
  void foo(int X, double Y, int Z) -> void foo(int X, int Z, double Y)

the Darwin ABI specifies that any integer arguments in the first 32 bytes worth
of arguments get assigned to r3 through r10. That is, if you have a function
foo(int, double, int) you get r3, f1, r6, since the 64 bit double ate up the
argument bytes for r4 and r5. The trick then would be to shuffle the argument
order for functions we can internalize so that the maximum number of 
integers/pointers get passed in regs before you see any of the fp arguments.

Instead of implementing this, it would actually probably be easier to just 
implement a PPC fastcc, where we could do whatever we wanted to the CC, 
including having this work sanely.

===-------------------------------------------------------------------------===

Fix Darwin FP-In-Integer Registers ABI

Darwin passes doubles in structures in integer registers, which is very very 
bad.  Add something like a BITCAST to LLVM, then do an i-p transformation that
percolates these things out of functions.

Check out how horrible this is:
http://gcc.gnu.org/ml/gcc/2005-10/msg01036.html

This is an extension of "interprocedural CC unmunging" that can't be done with
just fastcc.

===-------------------------------------------------------------------------===

Compile this:

int foo(int a) {
  int b = (a < 8);
  if (b) {
    return b * 3;     // ignore the fact that this is always 3.
  } else {
    return 2;
  }
}

into something not this:

_foo:
1)      cmpwi cr7, r3, 8
        mfcr r2, 1
        rlwinm r2, r2, 29, 31, 31
1)      cmpwi cr0, r3, 7
        bgt cr0, LBB1_2 ; UnifiedReturnBlock
LBB1_1: ; then
        rlwinm r2, r2, 0, 31, 31
        mulli r3, r2, 3
        blr
LBB1_2: ; UnifiedReturnBlock
        li r3, 2
        blr

In particular, the two compares (marked 1) could be shared by reversing one.
This could be done in the dag combiner, by swapping a BR_CC when a SETCC of the
same operands (but backwards) exists.  In this case, this wouldn't save us 
anything though, because the compares still wouldn't be shared.

===-------------------------------------------------------------------------===

We should custom expand setcc instead of pretending that we have it.  That
would allow us to expose the access of the crbit after the mfcr, allowing
that access to be trivially folded into other ops.  A simple example:

int foo(int a, int b) { return (a < b) << 4; }

compiles into:

_foo:
        cmpw cr7, r3, r4
        mfcr r2, 1
        rlwinm r2, r2, 29, 31, 31
        slwi r3, r2, 4
        blr

===-------------------------------------------------------------------------===

Fold add and sub with constant into non-extern, non-weak addresses so this:

static int a;
void bar(int b) { a = b; }
void foo(unsigned char *c) {
  *c = a;
}

So that 

_foo:
        lis r2, ha16(_a)
        la r2, lo16(_a)(r2)
        lbz r2, 3(r2)
        stb r2, 0(r3)
        blr

Becomes

_foo:
        lis r2, ha16(_a+3)
        lbz r2, lo16(_a+3)(r2)
        stb r2, 0(r3)
        blr

===-------------------------------------------------------------------------===

We generate really bad code for this:

int f(signed char *a, _Bool b, _Bool c) {
   signed char t = 0;
  if (b)  t = *a;
  if (c)  *a = t;
}

===-------------------------------------------------------------------------===

This:
int test(unsigned *P) { return *P >> 24; }

Should compile to:

_test:
        lbz r3,0(r3)
        blr

not:

_test:
        lwz r2, 0(r3)
        srwi r3, r2, 24
        blr

===-------------------------------------------------------------------------===

On the G5, logical CR operations are more expensive in their three
address form: ops that read/write the same register are half as expensive as
those that read from two registers that are different from their destination.

We should model this with two separate instructions.  The isel should generate
the "two address" form of the instructions.  When the register allocator 
detects that it needs to insert a copy due to the two-addresness of the CR
logical op, it will invoke PPCInstrInfo::convertToThreeAddress.  At this point
we can convert to the "three address" instruction, to save code space.

This only matters when we start generating cr logical ops.

===-------------------------------------------------------------------------===

We should compile these two functions to the same thing:

#include <stdlib.h>
void f(int a, int b, int *P) {
  *P = (a-b)>=0?(a-b):(b-a);
}
void g(int a, int b, int *P) {
  *P = abs(a-b);
}

Further, they should compile to something better than:

_g:
        subf r2, r4, r3
        subfic r3, r2, 0
        cmpwi cr0, r2, -1
        bgt cr0, LBB2_2 ; entry
LBB2_1: ; entry
        mr r2, r3
LBB2_2: ; entry
        stw r2, 0(r5)
        blr

GCC produces:

_g:
        subf r4,r4,r3
        srawi r2,r4,31
        xor r0,r2,r4
        subf r0,r2,r0
        stw r0,0(r5)
        blr

... which is much nicer.

This theoretically may help improve twolf slightly (used in dimbox.c:142?).

===-------------------------------------------------------------------------===

PR5945: This: 
define i32 @clamp0g(i32 %a) {
entry:
        %cmp = icmp slt i32 %a, 0
        %sel = select i1 %cmp, i32 0, i32 %a
        ret i32 %sel
}

Is compile to this with the PowerPC (32-bit) backend:

_clamp0g:
        cmpwi cr0, r3, 0
        li r2, 0
        blt cr0, LBB1_2
; BB#1:                                                     ; %entry
        mr r2, r3
LBB1_2:                                                     ; %entry
        mr r3, r2
        blr

This could be reduced to the much simpler:

_clamp0g:
        srawi r2, r3, 31
        andc r3, r3, r2
        blr

===-------------------------------------------------------------------------===

int foo(int N, int ***W, int **TK, int X) {
  int t, i;
  
  for (t = 0; t < N; ++t)
    for (i = 0; i < 4; ++i)
      W[t / X][i][t % X] = TK[i][t];
      
  return 5;
}

We generate relatively atrocious code for this loop compared to gcc.

We could also strength reduce the rem and the div:
http://www.lcs.mit.edu/pubs/pdf/MIT-LCS-TM-600.pdf

===-------------------------------------------------------------------------===

float foo(float X) { return (int)(X); }

Currently produces:

_foo:
        fctiwz f0, f1
        stfd f0, -8(r1)
        lwz r2, -4(r1)
        extsw r2, r2
        std r2, -16(r1)
        lfd f0, -16(r1)
        fcfid f0, f0
        frsp f1, f0
        blr

We could use a target dag combine to turn the lwz/extsw into an lwa when the 
lwz has a single use.  Since LWA is cracked anyway, this would be a codesize
win only.

===-------------------------------------------------------------------------===

We generate ugly code for this:

void func(unsigned int *ret, float dx, float dy, float dz, float dw) {
  unsigned code = 0;
  if(dx < -dw) code |= 1;
  if(dx > dw)  code |= 2;
  if(dy < -dw) code |= 4;
  if(dy > dw)  code |= 8;
  if(dz < -dw) code |= 16;
  if(dz > dw)  code |= 32;
  *ret = code;
}

===-------------------------------------------------------------------------===

Complete the signed i32 to FP conversion code using 64-bit registers
transformation, good for PI.  See PPCISelLowering.cpp, this comment:

     // FIXME: disable this lowered code.  This generates 64-bit register values,
     // and we don't model the fact that the top part is clobbered by calls.  We
     // need to flag these together so that the value isn't live across a call.
     //setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

Also, if the registers are spilled to the stack, we have to ensure that all
64-bits of them are save/restored, otherwise we will miscompile the code.  It
sounds like we need to get the 64-bit register classes going.

===-------------------------------------------------------------------------===

%struct.B = type { i8, [3 x i8] }

define void @bar(%struct.B* %b) {
entry:
        %tmp = bitcast %struct.B* %b to i32*              ; <uint*> [#uses=1]
        %tmp = load i32* %tmp          ; <uint> [#uses=1]
        %tmp3 = bitcast %struct.B* %b to i32*             ; <uint*> [#uses=1]
        %tmp4 = load i32* %tmp3                ; <uint> [#uses=1]
        %tmp8 = bitcast %struct.B* %b to i32*             ; <uint*> [#uses=2]
        %tmp9 = load i32* %tmp8                ; <uint> [#uses=1]
        %tmp4.mask17 = shl i32 %tmp4, i8 1          ; <uint> [#uses=1]
        %tmp1415 = and i32 %tmp4.mask17, 2147483648            ; <uint> [#uses=1]
        %tmp.masked = and i32 %tmp, 2147483648         ; <uint> [#uses=1]
        %tmp11 = or i32 %tmp1415, %tmp.masked          ; <uint> [#uses=1]
        %tmp12 = and i32 %tmp9, 2147483647             ; <uint> [#uses=1]
        %tmp13 = or i32 %tmp12, %tmp11         ; <uint> [#uses=1]
        store i32 %tmp13, i32* %tmp8
        ret void
}

We emit:

_foo:
        lwz r2, 0(r3)
        slwi r4, r2, 1
        or r4, r4, r2
        rlwimi r2, r4, 0, 0, 0
        stw r2, 0(r3)
        blr

We could collapse a bunch of those ORs and ANDs and generate the following
equivalent code:

_foo:
        lwz r2, 0(r3)
        rlwinm r4, r2, 1, 0, 0
        or r2, r2, r4
        stw r2, 0(r3)
        blr

===-------------------------------------------------------------------------===

We compile:

unsigned test6(unsigned x) { 
  return ((x & 0x00FF0000) >> 16) | ((x & 0x000000FF) << 16);
}

into:

_test6:
        lis r2, 255
        rlwinm r3, r3, 16, 0, 31
        ori r2, r2, 255
        and r3, r3, r2
        blr

GCC gets it down to:

_test6:
        rlwinm r0,r3,16,8,15
        rlwinm r3,r3,16,24,31
        or r3,r3,r0
        blr


===-------------------------------------------------------------------------===

Consider a function like this:

float foo(float X) { return X + 1234.4123f; }

The FP constant ends up in the constant pool, so we need to get the LR register.
 This ends up producing code like this:

_foo:
.LBB_foo_0:     ; entry
        mflr r11
***     stw r11, 8(r1)
        bl "L00000$pb"
"L00000$pb":
        mflr r2
        addis r2, r2, ha16(.CPI_foo_0-"L00000$pb")
        lfs f0, lo16(.CPI_foo_0-"L00000$pb")(r2)
        fadds f1, f1, f0
***     lwz r11, 8(r1)
        mtlr r11
        blr

This is functional, but there is no reason to spill the LR register all the way
to the stack (the two marked instrs): spilling it to a GPR is quite enough.

Implementing this will require some codegen improvements.  Nate writes:

"So basically what we need to support the "no stack frame save and restore" is a
generalization of the LR optimization to "callee-save regs".

Currently, we have LR marked as a callee-save reg.  The register allocator sees
that it's callee save, and spills it directly to the stack.

Ideally, something like this would happen:

LR would be in a separate register class from the GPRs. The class of LR would be
marked "unspillable".  When the register allocator came across an unspillable
reg, it would ask "what is the best class to copy this into that I *can* spill"
If it gets a class back, which it will in this case (the gprs), it grabs a free
register of that class.  If it is then later necessary to spill that reg, so be
it.

===-------------------------------------------------------------------------===

We compile this:
int test(_Bool X) {
  return X ? 524288 : 0;
}

to: 
_test:
        cmplwi cr0, r3, 0
        lis r2, 8
        li r3, 0
        beq cr0, LBB1_2 ;entry
LBB1_1: ;entry
        mr r3, r2
LBB1_2: ;entry
        blr 

instead of:
_test:
        addic r2,r3,-1
        subfe r0,r2,r3
        slwi r3,r0,19
        blr

This sort of thing occurs a lot due to globalopt.

===-------------------------------------------------------------------------===

We compile:

define i32 @bar(i32 %x) nounwind readnone ssp {
entry:
  %0 = icmp eq i32 %x, 0                          ; <i1> [#uses=1]
  %neg = sext i1 %0 to i32              ; <i32> [#uses=1]
  ret i32 %neg
}

to:

_bar:
	cntlzw r2, r3
	slwi r2, r2, 26
	srawi r3, r2, 31
	blr 

it would be better to produce:

_bar: 
        addic r3,r3,-1
        subfe r3,r3,r3
        blr

===-------------------------------------------------------------------------===

We currently compile 32-bit bswap:

declare i32 @llvm.bswap.i32(i32 %A)
define i32 @test(i32 %A) {
        %B = call i32 @llvm.bswap.i32(i32 %A)
        ret i32 %B
}

to:

_test:
        rlwinm r2, r3, 24, 16, 23
        slwi r4, r3, 24
        rlwimi r2, r3, 8, 24, 31
        rlwimi r4, r3, 8, 8, 15
        rlwimi r4, r2, 0, 16, 31
        mr r3, r4
        blr 

it would be more efficient to produce:

_foo:   mr r0,r3
        rlwinm r3,r3,8,0xffffffff
        rlwimi r3,r0,24,0,7
        rlwimi r3,r0,24,16,23
        blr

===-------------------------------------------------------------------------===

test/CodeGen/PowerPC/2007-03-24-cntlzd.ll compiles to:

__ZNK4llvm5APInt17countLeadingZerosEv:
        ld r2, 0(r3)
        cntlzd r2, r2
        or r2, r2, r2     <<-- silly.
        addi r3, r2, -64
        blr 

The dead or is a 'truncate' from 64- to 32-bits.

===-------------------------------------------------------------------------===

We generate horrible ppc code for this:

#define N  2000000
double   a[N],c[N];
void simpleloop() {
   int j;
   for (j=0; j<N; j++)
     c[j] = a[j];
}

LBB1_1: ;bb
        lfdx f0, r3, r4
        addi r5, r5, 1                 ;; Extra IV for the exit value compare.
        stfdx f0, r2, r4
        addi r4, r4, 8

        xoris r6, r5, 30               ;; This is due to a large immediate.
        cmplwi cr0, r6, 33920
        bne cr0, LBB1_1

//===---------------------------------------------------------------------===//

This:
        #include <algorithm>
        inline std::pair<unsigned, bool> full_add(unsigned a, unsigned b)
        { return std::make_pair(a + b, a + b < a); }
        bool no_overflow(unsigned a, unsigned b)
        { return !full_add(a, b).second; }

Should compile to:

__Z11no_overflowjj:
        add r4,r3,r4
        subfc r3,r3,r4
        li r3,0
        adde r3,r3,r3
        blr

(or better) not:

__Z11no_overflowjj:
        add r2, r4, r3
        cmplw cr7, r2, r3
        mfcr r2
        rlwinm r2, r2, 29, 31, 31
        xori r3, r2, 1
        blr 

//===---------------------------------------------------------------------===//

We compile some FP comparisons into an mfcr with two rlwinms and an or.  For
example:
#include <math.h>
int test(double x, double y) { return islessequal(x, y);}
int test2(double x, double y) {  return islessgreater(x, y);}
int test3(double x, double y) {  return !islessequal(x, y);}

Compiles into (all three are similar, but the bits differ):

_test:
	fcmpu cr7, f1, f2
	mfcr r2
	rlwinm r3, r2, 29, 31, 31
	rlwinm r2, r2, 31, 31, 31
	or r3, r2, r3
	blr 

GCC compiles this into:

 _test:
	fcmpu cr7,f1,f2
	cror 30,28,30
	mfcr r3
	rlwinm r3,r3,31,1
	blr
        
which is more efficient and can use mfocr.  See PR642 for some more context.

//===---------------------------------------------------------------------===//

void foo(float *data, float d) {
   long i;
   for (i = 0; i < 8000; i++)
      data[i] = d;
}
void foo2(float *data, float d) {
   long i;
   data--;
   for (i = 0; i < 8000; i++) {
      data[1] = d;
      data++;
   }
}

These compile to:

_foo:
	li r2, 0
LBB1_1:	; bb
	addi r4, r2, 4
	stfsx f1, r3, r2
	cmplwi cr0, r4, 32000
	mr r2, r4
	bne cr0, LBB1_1	; bb
	blr 
_foo2:
	li r2, 0
LBB2_1:	; bb
	addi r4, r2, 4
	stfsx f1, r3, r2
	cmplwi cr0, r4, 32000
	mr r2, r4
	bne cr0, LBB2_1	; bb
	blr 

The 'mr' could be eliminated to folding the add into the cmp better.

//===---------------------------------------------------------------------===//
Codegen for the following (low-probability) case deteriorated considerably 
when the correctness fixes for unordered comparisons went in (PR 642, 58871).
It should be possible to recover the code quality described in the comments.

; RUN: llvm-as < %s | llc -march=ppc32  | grep or | count 3
; This should produce one 'or' or 'cror' instruction per function.

; RUN: llvm-as < %s | llc -march=ppc32  | grep mfcr | count 3
; PR2964

define i32 @test(double %x, double %y) nounwind  {
entry:
	%tmp3 = fcmp ole double %x, %y		; <i1> [#uses=1]
	%tmp345 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp345
}

define i32 @test2(double %x, double %y) nounwind  {
entry:
	%tmp3 = fcmp one double %x, %y		; <i1> [#uses=1]
	%tmp345 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp345
}

define i32 @test3(double %x, double %y) nounwind  {
entry:
	%tmp3 = fcmp ugt double %x, %y		; <i1> [#uses=1]
	%tmp34 = zext i1 %tmp3 to i32		; <i32> [#uses=1]
	ret i32 %tmp34
}
//===----------------------------------------------------------------------===//
; RUN: llvm-as < %s | llc -march=ppc32 | not grep fneg

; This could generate FSEL with appropriate flags (FSEL is not IEEE-safe, and 
; should not be generated except with -enable-finite-only-fp-math or the like).
; With the correctness fixes for PR642 (58871) LowerSELECT_CC would need to
; recognize a more elaborate tree than a simple SETxx.

define double @test_FNEG_sel(double %A, double %B, double %C) {
        %D = fsub double -0.000000e+00, %A               ; <double> [#uses=1]
        %Cond = fcmp ugt double %D, -0.000000e+00               ; <i1> [#uses=1]
        %E = select i1 %Cond, double %B, double %C              ; <double> [#uses=1]
        ret double %E
}

//===----------------------------------------------------------------------===//
The save/restore sequence for CR in prolog/epilog is terrible:
- Each CR subreg is saved individually, rather than doing one save as a unit.
- On Darwin, the save is done after the decrement of SP, which means the offset
from SP of the save slot can be too big for a store instruction, which means we
need an additional register (currently hacked in 96015+96020; the solution there
is correct, but poor).
- On SVR4 the same thing can happen, and I don't think saving before the SP
decrement is safe on that target, as there is no red zone.  This is currently
broken AFAIK, although it's not a target I can exercise.
The following demonstrates the problem:
extern void bar(char *p);
void foo() {
  char x[100000];
  bar(x);
  __asm__("" ::: "cr2");
}
