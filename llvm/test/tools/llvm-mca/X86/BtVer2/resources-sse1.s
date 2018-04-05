# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=btver2 -instruction-tables < %s | FileCheck %s

addps       %xmm0, %xmm2
addps       (%rax), %xmm2

addss       %xmm0, %xmm2
addss       (%rax), %xmm2

andnps      %xmm0, %xmm2
andnps      (%rax), %xmm2

andps       %xmm0, %xmm2
andps       (%rax), %xmm2

cmpps       $0, %xmm0, %xmm2
cmpps       $0, (%rax), %xmm2

cmpss       $0, %xmm0, %xmm2
cmpss       $0, (%rax), %xmm2

comiss      %xmm0, %xmm1
comiss      (%rax), %xmm1

cvtsi2ss    %ecx, %xmm2
cvtsi2ss    %rcx, %xmm2
cvtsi2ss    (%rax), %xmm2
cvtsi2ss    (%rax), %xmm2

cvtss2si    %xmm0, %ecx
cvtss2si    %xmm0, %rcx
cvtss2si    (%rax), %ecx
cvtss2si    (%rax), %rcx

cvttss2si   %xmm0, %ecx
cvttss2si   %xmm0, %rcx
cvttss2si   (%rax), %ecx
cvttss2si   (%rax), %rcx

divps       %xmm0, %xmm2
divps       (%rax), %xmm2

divss       %xmm0, %xmm2
divss       (%rax), %xmm2

ldmxcsr     (%rax)

maxps       %xmm0, %xmm2
maxps       (%rax), %xmm2

maxss       %xmm0, %xmm2
maxss       (%rax), %xmm2

minps       %xmm0, %xmm2
minps       (%rax), %xmm2

minss       %xmm0, %xmm2
minss       (%rax), %xmm2

movaps      %xmm0, %xmm2
movaps      %xmm0, (%rax)
movaps      (%rax), %xmm2

movhlps     %xmm0, %xmm2
movlhps     %xmm0, %xmm2

movhps      %xmm0, (%rax)
movhps      (%rax), %xmm2

movlps      %xmm0, (%rax)
movlps      (%rax), %xmm2

movmskps    %xmm0, %rcx

movntps     %xmm0, (%rax)

movss       %xmm0, %xmm2
movss       %xmm0, (%rax)
movss       (%rax), %xmm2

movups      %xmm0, %xmm2
movups      %xmm0, (%rax)
movups      (%rax), %xmm2

mulps       %xmm0, %xmm2
mulps       (%rax), %xmm2

mulss       %xmm0, %xmm2
mulss       (%rax), %xmm2

orps        %xmm0, %xmm2
orps        (%rax), %xmm2

pinsrw      $1, %rax, %xmm1
pinsrw      $1, (%rax), %xmm1

rcpps       %xmm0, %xmm2
rcpps       (%rax), %xmm2

rcpss       %xmm0, %xmm2
rcpss       (%rax), %xmm2

rsqrtps     %xmm0, %xmm2
rsqrtps     (%rax), %xmm2

rsqrtss     %xmm0, %xmm2
rsqrtss     (%rax), %xmm2

shufps      $1, %xmm0, %xmm2
shufps      $1, (%rax), %xmm2

sqrtps      %xmm0, %xmm2
sqrtps      (%rax), %xmm2

sqrtss      %xmm0, %xmm2
sqrtss      (%rax), %xmm2

stmxcsr     (%rax)

subps       %xmm0, %xmm2
subps       (%rax), %xmm2

subss       %xmm0, %xmm2
subss       (%rax), %xmm2

ucomiss     %xmm0, %xmm1
ucomiss     (%rax), %xmm1

unpckhps    %xmm0, %xmm2
unpckhps    (%rax), %xmm2

unpcklps    %xmm0, %xmm2
unpcklps    (%rax), %xmm2

xorps       %xmm0, %xmm2
xorps       (%rax), %xmm2


# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]	Instructions:
# CHECK-NEXT:  1      3     1.00                    	addps	%xmm0, %xmm2
# CHECK-NEXT:  1      8     1.00    *               	addps	(%rax), %xmm2
# CHECK-NEXT:  1      3     1.00                    	addss	%xmm0, %xmm2
# CHECK-NEXT:  1      8     1.00    *               	addss	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	andnps	%xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	andnps	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	andps	%xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	andps	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	cmpps	$0, %xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	cmpps	$0, (%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	cmpss	$0, %xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	cmpss	$0, (%rax), %xmm2
# CHECK-NEXT:  1      3     1.00                    	comiss	%xmm0, %xmm1
# CHECK-NEXT:  1      8     1.00    *               	comiss	(%rax), %xmm1
# CHECK-NEXT:  2      9     1.00                    	cvtsi2ssl	%ecx, %xmm2
# CHECK-NEXT:  2      9     1.00                    	cvtsi2ssq	%rcx, %xmm2
# CHECK-NEXT:  2      14    1.00    *               	cvtsi2ssl	(%rax), %xmm2
# CHECK-NEXT:  2      14    1.00    *               	cvtsi2ssl	(%rax), %xmm2
# CHECK-NEXT:  2      7     1.00                    	cvtss2si	%xmm0, %ecx
# CHECK-NEXT:  2      7     1.00                    	cvtss2si	%xmm0, %rcx
# CHECK-NEXT:  2      12    1.00    *               	cvtss2si	(%rax), %ecx
# CHECK-NEXT:  2      12    1.00    *               	cvtss2si	(%rax), %rcx
# CHECK-NEXT:  2      7     1.00                    	cvttss2si	%xmm0, %ecx
# CHECK-NEXT:  2      7     1.00                    	cvttss2si	%xmm0, %rcx
# CHECK-NEXT:  2      12    1.00    *               	cvttss2si	(%rax), %ecx
# CHECK-NEXT:  2      12    1.00    *               	cvttss2si	(%rax), %rcx
# CHECK-NEXT:  1      19    19.00                   	divps	%xmm0, %xmm2
# CHECK-NEXT:  1      24    19.00   *               	divps	(%rax), %xmm2
# CHECK-NEXT:  1      19    19.00                   	divss	%xmm0, %xmm2
# CHECK-NEXT:  1      24    19.00   *               	divss	(%rax), %xmm2
# CHECK-NEXT:  1      5     1.00    *      *      * 	ldmxcsr	(%rax)
# CHECK-NEXT:  1      2     1.00                    	maxps	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	maxps	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	maxss	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	maxss	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	minps	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	minps	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	minss	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	minss	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	movaps	%xmm0, %xmm2
# CHECK-NEXT:  1      1     1.00           *        	movaps	%xmm0, (%rax)
# CHECK-NEXT:  1      5     1.00    *               	movaps	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	movhlps	%xmm0, %xmm2
# CHECK-NEXT:  1      1     0.50                    	movlhps	%xmm0, %xmm2
# CHECK-NEXT:  1      1     1.00           *        	movhps	%xmm0, (%rax)
# CHECK-NEXT:  1      6     1.00    *               	movhps	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00           *        	movlps	%xmm0, (%rax)
# CHECK-NEXT:  1      6     1.00    *               	movlps	(%rax), %xmm2
# CHECK-NEXT:  1      3     1.00                    	movmskps	%xmm0, %ecx
# CHECK-NEXT:  1      3     1.00           *        	movntps	%xmm0, (%rax)
# CHECK-NEXT:  1      1     0.50                    	movss	%xmm0, %xmm2
# CHECK-NEXT:  1      1     1.00           *        	movss	%xmm0, (%rax)
# CHECK-NEXT:  1      5     1.00    *               	movss	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	movups	%xmm0, %xmm2
# CHECK-NEXT:  1      1     1.00           *        	movups	%xmm0, (%rax)
# CHECK-NEXT:  1      5     1.00    *               	movups	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	mulps	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	mulps	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	mulss	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	mulss	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	orps	%xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	orps	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	pinsrw	$1, %eax, %xmm1
# CHECK-NEXT:  1      6     1.00    *               	pinsrw	$1, (%rax), %xmm1
# CHECK-NEXT:  1      2     1.00                    	rcpps	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	rcpps	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	rcpss	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	rcpss	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	rsqrtps	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	rsqrtps	(%rax), %xmm2
# CHECK-NEXT:  1      2     1.00                    	rsqrtss	%xmm0, %xmm2
# CHECK-NEXT:  1      7     1.00    *               	rsqrtss	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	shufps	$1, %xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	shufps	$1, (%rax), %xmm2
# CHECK-NEXT:  1      21    21.00                   	sqrtps	%xmm0, %xmm2
# CHECK-NEXT:  1      26    21.00   *               	sqrtps	(%rax), %xmm2
# CHECK-NEXT:  1      21    21.00                   	sqrtss	%xmm0, %xmm2
# CHECK-NEXT:  1      26    21.00   *               	sqrtss	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00    *      *      * 	stmxcsr	(%rax)
# CHECK-NEXT:  1      3     1.00                    	subps	%xmm0, %xmm2
# CHECK-NEXT:  1      8     1.00    *               	subps	(%rax), %xmm2
# CHECK-NEXT:  1      3     1.00                    	subss	%xmm0, %xmm2
# CHECK-NEXT:  1      8     1.00    *               	subss	(%rax), %xmm2
# CHECK-NEXT:  1      3     1.00                    	ucomiss	%xmm0, %xmm1
# CHECK-NEXT:  1      8     1.00    *               	ucomiss	(%rax), %xmm1
# CHECK-NEXT:  1      1     0.50                    	unpckhps	%xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	unpckhps	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	unpcklps	%xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	unpcklps	(%rax), %xmm2
# CHECK-NEXT:  1      1     0.50                    	xorps	%xmm0, %xmm2
# CHECK-NEXT:  1      6     1.00    *               	xorps	(%rax), %xmm2


# CHECK:      Resources:
# CHECK-NEXT: [0] - JALU0
# CHECK-NEXT: [1] - JALU1
# CHECK-NEXT: [2] - JDiv
# CHECK-NEXT: [3] - JFPA
# CHECK-NEXT: [4] - JFPM
# CHECK-NEXT: [5] - JFPU0
# CHECK-NEXT: [6] - JFPU1
# CHECK-NEXT: [7] - JLAGU
# CHECK-NEXT: [8] - JMul
# CHECK-NEXT: [9] - JSAGU
# CHECK-NEXT: [10] - JSTC
# CHECK-NEXT: [11] - JVALU0
# CHECK-NEXT: [12] - JVALU1
# CHECK-NEXT: [13] - JVIMUL


# CHECK: Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   [13]   	Instructions:
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	addps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	addps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	addss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	addss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	andnps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	andnps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	andps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	andps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	cmpps	$0, %xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	cmpps	$0, (%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	cmpss	$0, %xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	cmpss	$0, (%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	comiss	%xmm0, %xmm1
# CHECK-NEXT: 1.00    -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	comiss	(%rax), %xmm1
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -     1.00    -      -      -     	cvtsi2ssl	%ecx, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -      -     1.00    -      -      -     	cvtsi2ssq	%rcx, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -     1.00    -      -      -     	cvtsi2ssl	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00   1.00    -      -     1.00    -      -      -     	cvtsi2ssl	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00    -      -      -     1.00    -      -      -     	cvtss2si	%xmm0, %ecx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00    -      -      -     1.00    -      -      -     	cvtss2si	%xmm0, %rcx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   1.00    -      -     1.00    -      -      -     	cvtss2si	(%rax), %ecx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   1.00    -      -     1.00    -      -      -     	cvtss2si	(%rax), %rcx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00    -      -      -     1.00    -      -      -     	cvttss2si	%xmm0, %ecx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00    -      -      -     1.00    -      -      -     	cvttss2si	%xmm0, %rcx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   1.00    -      -     1.00    -      -      -     	cvttss2si	(%rax), %ecx
# CHECK-NEXT: 1.00    -      -     1.00    -      -     1.00   1.00    -      -     1.00    -      -      -     	cvttss2si	(%rax), %rcx
# CHECK-NEXT:  -      -      -      -     19.00   -     1.00    -      -      -      -      -      -      -     	divps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     19.00   -     1.00   1.00    -      -      -      -      -      -     	divps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     19.00   -     1.00    -      -      -      -      -      -      -     	divss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     19.00   -     1.00   1.00    -      -      -      -      -      -     	divss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -      -      -     1.00    -      -      -      -      -      -     	ldmxcsr	(%rax)
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	maxps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	maxps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	maxss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	maxss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	minps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	minps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	minss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	minss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	movaps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -     1.00   1.00    -      -      -     	movaps	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	movaps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	movhlps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	movlhps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -      -      -      -     1.00    -      -      -      -     	movhps	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	movhps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -      -      -      -      -     1.00    -      -      -      -     	movlps	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	movlps	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	movmskps	%xmm0, %ecx
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -     1.00   1.00    -      -      -     	movntps	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	movss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -      -      -      -     1.00    -      -      -      -     	movss	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -      -      -      -      -     1.00    -      -      -      -      -      -     	movss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	movups	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -      -      -     1.00    -      -     1.00   1.00    -      -      -     	movups	%xmm0, (%rax)
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	movups	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	mulps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	mulps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	mulss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	mulss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	orps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	orps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     0.50   0.50    -      -      -      -     0.50   0.50    -     	pinsrw	$1, %eax, %xmm1
# CHECK-NEXT:  -      -      -      -      -     0.50   0.50   1.00    -      -      -     0.50   0.50    -     	pinsrw	$1, (%rax), %xmm1
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	rcpps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	rcpps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	rcpss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	rcpss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	rsqrtps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	rsqrtps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00    -      -      -      -      -      -      -     	rsqrtss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     1.00    -     1.00   1.00    -      -      -      -      -      -     	rsqrtss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	shufps	$1, %xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	shufps	$1, (%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     21.00   -     1.00    -      -      -      -      -      -      -     	sqrtps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     21.00   -     1.00   1.00    -      -      -      -      -      -     	sqrtps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -     21.00   -     1.00    -      -      -      -      -      -      -     	sqrtss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -      -     21.00   -     1.00   1.00    -      -      -      -      -      -     	sqrtss	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -      -      -      -      -     1.00    -      -      -      -     	stmxcsr	(%rax)
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	subps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	subps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	subss	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	subss	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -     1.00    -     1.00    -      -      -      -      -      -      -      -     	ucomiss	%xmm0, %xmm1
# CHECK-NEXT: 1.00    -      -     1.00    -     1.00    -     1.00    -      -      -      -      -      -     	ucomiss	(%rax), %xmm1
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	unpckhps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	unpckhps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	unpcklps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	unpcklps	(%rax), %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50    -      -      -      -      -      -      -     	xorps	%xmm0, %xmm2
# CHECK-NEXT:  -      -      -     0.50   0.50   0.50   0.50   1.00    -      -      -      -      -      -     	xorps	(%rax), %xmm2
