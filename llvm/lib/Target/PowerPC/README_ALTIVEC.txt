//===- README_ALTIVEC.txt - Notes for improving Altivec code gen ----------===//

Implement PPCInstrInfo::isLoadFromStackSlot/isStoreToStackSlot for vector
registers, to generate better spill code.

//===----------------------------------------------------------------------===//

The first should be a single lvx from the constant pool, the second should be 
a xor/stvx:

void foo(void) {
  int x[8] __attribute__((aligned(128))) = { 1, 1, 1, 17, 1, 1, 1, 1 };
  bar (x);
}

#include <string.h>
void foo(void) {
  int x[8] __attribute__((aligned(128)));
  memset (x, 0, sizeof (x));
  bar (x);
}

//===----------------------------------------------------------------------===//

Altivec: Codegen'ing MUL with vector FMADD should add -0.0, not 0.0:
http://gcc.gnu.org/bugzilla/show_bug.cgi?id=8763

When -ffast-math is on, we can use 0.0.

//===----------------------------------------------------------------------===//

  Consider this:
  v4f32 Vector;
  v4f32 Vector2 = { Vector.X, Vector.X, Vector.X, Vector.X };

Since we know that "Vector" is 16-byte aligned and we know the element offset 
of ".X", we should change the load into a lve*x instruction, instead of doing
a load/store/lve*x sequence.

//===----------------------------------------------------------------------===//

For functions that use altivec AND have calls, we are VRSAVE'ing all call
clobbered regs.

//===----------------------------------------------------------------------===//

Implement passing vectors by value into calls and receiving them as arguments.

//===----------------------------------------------------------------------===//

GCC apparently tries to codegen { C1, C2, Variable, C3 } as a constant pool load
of C1/C2/C3, then a load and vperm of Variable.

//===----------------------------------------------------------------------===//

We need a way to teach tblgen that some operands of an intrinsic are required to
be constants.  The verifier should enforce this constraint.

//===----------------------------------------------------------------------===//

We currently codegen SCALAR_TO_VECTOR as a store of the scalar to a 16-byte
aligned stack slot, followed by a load/vperm.  We should probably just store it
to a scalar stack slot, then use lvsl/vperm to load it.  If the value is already
in memory this is a big win.

//===----------------------------------------------------------------------===//

extract_vector_elt of an arbitrary constant vector can be done with the 
following instructions:

vTemp = vec_splat(v0,2);    // 2 is the element the src is in.
vec_ste(&destloc,0,vTemp);

We can do an arbitrary non-constant value by using lvsr/perm/ste.

//===----------------------------------------------------------------------===//

If we want to tie instruction selection into the scheduler, we can do some
constant formation with different instructions.  For example, we can generate
"vsplti -1" with "vcmpequw R,R" and 1,1,1,1 with "vsubcuw R,R", and 0,0,0,0 with
"vsplti 0" or "vxor", each of which use different execution units, thus could
help scheduling.

This is probably only reasonable for a post-pass scheduler.

//===----------------------------------------------------------------------===//

For this function:

void test(vector float *A, vector float *B) {
  vector float C = (vector float)vec_cmpeq(*A, *B);
  if (!vec_any_eq(*A, *B))
    *B = (vector float){0,0,0,0};
  *A = C;
}

we get the following basic block:

	...
        lvx v2, 0, r4
        lvx v3, 0, r3
        vcmpeqfp v4, v3, v2
        vcmpeqfp. v2, v3, v2
        bne cr6, LBB1_2 ; cond_next

The vcmpeqfp/vcmpeqfp. instructions currently cannot be merged when the
vcmpeqfp. result is used by a branch.  This can be improved.

//===----------------------------------------------------------------------===//

The code generated for this is truly aweful:

vector float test(float a, float b) {
 return (vector float){ 0.0, a, 0.0, 0.0}; 
}

LCPI1_0:                                        ;  float
        .space  4
        .text
        .globl  _test
        .align  4
_test:
        mfspr r2, 256
        oris r3, r2, 4096
        mtspr 256, r3
        lis r3, ha16(LCPI1_0)
        addi r4, r1, -32
        stfs f1, -16(r1)
        addi r5, r1, -16
        lfs f0, lo16(LCPI1_0)(r3)
        stfs f0, -32(r1)
        lvx v2, 0, r4
        lvx v3, 0, r5
        vmrghw v3, v3, v2
        vspltw v2, v2, 0
        vmrghw v2, v2, v3
        mtspr 256, r2
        blr

//===----------------------------------------------------------------------===//

int foo(vector float *x, vector float *y) {
        if (vec_all_eq(*x,*y)) return 3245; 
        else return 12;
}

A predicate compare being used in a select_cc should have the same peephole
applied to it as a predicate compare used by a br_cc.  There should be no
mfcr here:

_foo:
        mfspr r2, 256
        oris r5, r2, 12288
        mtspr 256, r5
        li r5, 12
        li r6, 3245
        lvx v2, 0, r4
        lvx v3, 0, r3
        vcmpeqfp. v2, v3, v2
        mfcr r3, 2
        rlwinm r3, r3, 25, 31, 31
        cmpwi cr0, r3, 0
        bne cr0, LBB1_2 ; entry
LBB1_1: ; entry
        mr r6, r5
LBB1_2: ; entry
        mr r3, r6
        mtspr 256, r2
        blr

//===----------------------------------------------------------------------===//

CodeGen/PowerPC/vec_constants.ll has an and operation that should be
codegen'd to andc.  The issue is that the 'all ones' build vector is
SelectNodeTo'd a VSPLTISB instruction node before the and/xor is selected
which prevents the vnot pattern from matching.


//===----------------------------------------------------------------------===//

An alternative to the store/store/load approach for illegal insert element 
lowering would be:

1. store element to any ol' slot
2. lvx the slot
3. lvsl 0; splat index; vcmpeq to generate a select mask
4. lvsl slot + x; vperm to rotate result into correct slot
5. vsel result together.

//===----------------------------------------------------------------------===//

Should codegen branches on vec_any/vec_all to avoid mfcr.  Two examples:

#include <altivec.h>
 int f(vector float a, vector float b)
 {
  int aa = 0;
  if (vec_all_ge(a, b))
    aa |= 0x1;
  if (vec_any_ge(a,b))
    aa |= 0x2;
  return aa;
}

vector float f(vector float a, vector float b) { 
  if (vec_any_eq(a, b)) 
    return a; 
  else 
    return b; 
}

