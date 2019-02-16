; RUN: opt -analyze -scalar-evolution -S -debug-only=scalar-evolution,apint < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Use the following template to get a chrec {L,+,M,+,N}.
;
; define signext i32 @func() {
; entry:
;   br label %loop
;
; loop:
;   %ivr = phi i32 [ 0, %entry ], [ %ivr1, %loop ]
;   %inc = phi i32 [ X, %entry ], [ %inc1, %loop ]
;   %acc = phi i32 [ Y, %entry ], [ %acc1, %loop ]
;   %ivr1 = add i32 %ivr, %inc
;   %inc1 = add i32 %inc, Z                 ; M = inc1 = inc + N = X + N
;   %acc1 = add i32 %acc, %inc              ; L = acc1 = X + Y
;   %and  = and i32 %acc1, 2^W-1            ; iW
;   %cond = icmp eq i32 %and, 0
;   br i1 %cond, label %exit, label %loop
;
; exit:
;   %rv = phi i32 [ %acc1, %loop ]
;   ret i32 %rv
; }
;
; From
;       X + Y = L
;       X + Z = M
;           Z = N
; get
;       X = M - N
;       Y = N - M + L
;       Z = N

; The connection between the chrec coefficients {L,+,M,+,N} and the quadratic
; coefficients is that the quadratic equation is N x^2 + (2M-N) x + 2L = 0,
; where the equation was multiplied by 2 to make the coefficient at x^2 an
; integer (the actual equation is N/2 x^2 + (M-N/2) x + L = 0).

; Quadratic equation: 2x^2 + 2x + 4 in i4, solution (wrap): 4
; {14,+,14,+,14} -> X=0, Y=14, Z=14
;
; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test01'
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {-2,+,-2,+,-2}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 4
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation -2x^2 + -2x + -4, coeff bw: 5, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecExact{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving -2x^2 + -2x + -4, rw:5
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 2x^2 + 2x + -28, rw:5
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 4
; CHECK: Loop %loop: Unpredictable backedge-taken count
define signext i32 @test01() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [  0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [  0, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [ 14, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, 14
  %acc1 = add i32 %acc, %inc
  %and  = and i32 %acc1, 15
  %cond = icmp eq i32 %and, 0
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

; Quadratic equation: 1x^2 + -73x + -146 in i32, solution (wrap): 75
; {-72,+,-36,+,1} -> X=-37, Y=-35, Z=1
;
; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test02':
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {0,+,-36,+,1}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 32
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 1x^2 + -73x + 0, coeff bw: 33, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -73x + 4294967154, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -73x + -142, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 75
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -73x + 4294967154, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -73x + -4294967438, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 65573
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -73x + -146, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -73x + -146, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 75
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -73x + -146, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -73x + -146, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 75
; CHECK: Loop %loop: backedge-taken count is 75
define signext i32 @test02() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [  0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [ -37, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [ -35, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, 1
  %acc1 = add i32 %acc, %inc
  %and  = and i32 %acc1, -1
  %cond = icmp sgt i32 %and, 0
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

; Quadratic equation: 2x^2 - 4x + 34 in i4, solution (exact): 1.
; {17,+,-1,+,2} -> X=-3, Y=20, Z=2
;
; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test03':
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {1,+,-1,+,2}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 4
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 2x^2 + -4x + 2, coeff bw: 5, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecExact{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 2x^2 + -4x + 2, rw:5
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 2x^2 + -4x + 2, rw:5
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (root): 1
; CHECK: Loop %loop: backedge-taken count is 1
define signext i32 @test03() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [  0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [ -3, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [ 20, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, 2
  %acc1 = add i32 %acc, %inc
  %and  = and i32 %acc1, 15
  %cond = icmp eq i32 %and, 0
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

; Quadratic equation  4x^2 + 2x + 2 in i16, solution (wrap): 181
; {1,+,3,+,4} -> X=-1, Y=2, Z=4 (i16)
;
; This is an example where the returned solution is the first time an
; unsigned wrap occurs, whereas the actual exit condition occurs much
; later. The number of iterations returned by SolveQuadraticEquation
; is 181, but the loop will iterate 37174 times.
;
; Here is a C code that corresponds to this case that calculates the number
; of iterations:
;
; int test04() {
;   int c = 0;
;   int ivr = 0;
;   int inc = -1;
;   int acc = 2;
;
;   while (acc & 0xffff) {
;     c++;
;     ivr += inc;
;     inc += 4;
;     acc += inc;
;   }
;
;   return c;
; }
;

; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test04':
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {0,+,3,+,4}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 16
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 4x^2 + 2x + 0, coeff bw: 17, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 4x^2 + 2x + 2, rw:16
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 4x^2 + 2x + -65534, rw:16
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 128
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 4x^2 + 2x + 2, rw:17
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 4x^2 + 2x + -131070, rw:17
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 181
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 4x^2 + 2x + 2, rw:16
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 4x^2 + 2x + -65534, rw:16
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 128
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 4x^2 + 2x + 2, rw:17
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 4x^2 + 2x + -131070, rw:17
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 181
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {1,+,3,+,4}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 16
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 4x^2 + 2x + 2, coeff bw: 17, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecExact{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 4x^2 + 2x + 2, rw:17
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 4x^2 + 2x + -131070, rw:17
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 181
; CHECK: Loop %loop: Unpredictable backedge-taken count.
define signext i32 @test04() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [  0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [ -1, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [  2, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, 4
  %acc1 = add i32 %acc, %inc
  %and  = trunc i32 %acc1 to i16
  %cond = icmp eq i16 %and, 0
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

; A case with signed arithmetic, but unsigned comparison.

; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test05':
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {0,+,-1,+,-1}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 32
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation -1x^2 + -1x + 0, coeff bw: 33, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving -1x^2 + -1x + 4, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + 1x + -4, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving -1x^2 + -1x + 4, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + 1x + -4, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving -1x^2 + -1x + -2, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + 1x + -4294967294, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 65536
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving -1x^2 + -1x + -2, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + 1x + -8589934590, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 92682
; CHECK: Loop %loop: backedge-taken count is 2

define signext i32 @test05() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [ 0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [ 0, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [ -1, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, -1
  %acc1 = add i32 %acc, %inc
  %and  = and i32 %acc1, -1
  %cond = icmp ule i32 %and, -3
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

; A test that used to crash with one of the earlier versions of the code.

; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test06':
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {0,+,-99999,+,1}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 32
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 1x^2 + -199999x + 0, coeff bw: 33, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -199999x + -4294967294, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -199999x + 2, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 1
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -199999x + -4294967294, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -199999x + 4294967298, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 24469
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -199999x + -12, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -199999x + 4294967284, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 24469
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 1x^2 + -199999x + -12, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 1x^2 + -199999x + 8589934580, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solution (wrap): 62450
; CHECK: Loop %loop: backedge-taken count is 24469
define signext i32 @test06() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [ 0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [ -100000, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [ 100000, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, 1
  %acc1 = add i32 %acc, %inc
  %and  = and i32 %acc1, -1
  %cond = icmp sgt i32 %and, 5
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

; The equation
;   532052752x^2 + -450429774x + 71188414 = 0
; has two exact solutions (up to two decimal digits): 0.21 and 0.64.
; Since there is no integer between them, there is no integer n that either
; solves the equation exactly, or changes the sign of it between n and n+1.

; CHECK-LABEL: Printing analysis 'Scalar Evolution Analysis' for function 'test07':
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {0,+,40811489,+,532052752}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 32
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 532052752x^2 + -450429774x + 0, coeff bw: 33, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 532052752x^2 + -450429774x + 71188414, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 532052752x^2 + -450429774x + 71188414, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: no valid solution
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 532052752x^2 + -450429774x + 71188414, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 532052752x^2 + -450429774x + 71188414, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: no valid solution
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for signed overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 532052752x^2 + -450429774x + 71188414, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 532052752x^2 + -450429774x + 71188414, rw:32
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: no valid solution
; CHECK: {{.*}}SolveQuadraticAddRecRange{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 532052752x^2 + -450429774x + 71188414, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 532052752x^2 + -450429774x + 71188414, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: no valid solution
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: analyzing quadratic addrec: {35594207,+,40811489,+,532052752}<%loop>
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: addrec coeff bw: 32
; CHECK: {{.*}}GetQuadraticEquation{{.*}}: equation 532052752x^2 + -450429774x + 71188414, coeff bw: 33, multiplied by 2
; CHECK: {{.*}}SolveQuadraticAddRecExact{{.*}}: solving for unsigned overflow
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: solving 532052752x^2 + -450429774x + 71188414, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: updated coefficients 532052752x^2 + -450429774x + 71188414, rw:33
; CHECK: {{.*}}SolveQuadraticEquationWrap{{.*}}: no valid solution
; CHECK: Loop %loop: Unpredictable backedge-taken count.
define signext i32 @test07() {
entry:
  br label %loop

loop:
  %ivr = phi i32 [ 0, %entry ], [ %ivr1, %loop ]
  %inc = phi i32 [ -491241263, %entry ], [ %inc1, %loop ]
  %acc = phi i32 [ 526835470, %entry ], [ %acc1, %loop ]
  %ivr1 = add i32 %ivr, %inc
  %inc1 = add i32 %inc, 532052752
  %acc1 = add i32 %acc, %inc
  %and  = and i32 %acc1, -1
  %cond = icmp eq i32 %and, 0
  br i1 %cond, label %exit, label %loop

exit:
  %rv = phi i32 [ %acc1, %loop ]
  ret i32 %rv
}

