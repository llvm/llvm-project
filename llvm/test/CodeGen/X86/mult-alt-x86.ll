; RUN: llc < %s -mattr=+mmx,+sse2 -no-integrated-as
; ModuleID = 'mult-alt-x86.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i686-pc-win32"

@mout0 = common global i32 0, align 4
@min1 = common global i32 0, align 4
@dout0 = common global double 0.000000e+000, align 8
@din1 = common global double 0.000000e+000, align 8
@marray = common global [2 x i32] zeroinitializer, align 4

define void @single_R() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "=R,R,~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_q() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "=q,q,~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_Q() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "=Q,Q,~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_a() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "={ax},{ax},~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_b() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "={bx},{bx},~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_c() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "={cx},{cx},~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_d() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "={dx},{dx},~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_S() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "={si},{si},~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_D() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "={di},{di},~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_A() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  %0 = call i32 asm "foo $1,$0", "=A,A,~{dirflag},~{fpsr},~{flags}"(i32 %tmp) nounwind
  store i32 %0, i32* @mout0, align 4
  ret void
}

define void @single_f() nounwind {
entry:
  ret void
}

define void @single_t() nounwind {
entry:
  ret void
}

define void @single_u() nounwind {
entry:
  ret void
}

define void @single_y() nounwind {
entry:
  %tmp = load double, double* @din1, align 8
  %0 = call double asm "foo $1,$0", "=y,y,~{dirflag},~{fpsr},~{flags}"(double %tmp) nounwind
  store double %0, double* @dout0, align 8
  ret void
}

define void @single_x() nounwind {
entry:
  %tmp = load double, double* @din1, align 8
  %0 = call double asm "foo $1,$0", "=x,x,~{dirflag},~{fpsr},~{flags}"(double %tmp) nounwind
  store double %0, double* @dout0, align 8
  ret void
}

define void @single_Y0() nounwind {
entry:
  ret void
}

define void @single_I() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,I,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_J() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,J,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_K() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,K,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_L() nounwind {
entry:
; Missing lowering support for 'L'.
;  call void asm "foo $1,$0", "=*m,L,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_M() nounwind {
entry:
; Missing lowering support for 'M'.
;  call void asm "foo $1,$0", "=*m,M,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_N() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,N,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_G() nounwind {
entry:
; Missing lowering support for 'G'.
;  call void asm "foo $1,$0", "=*m,G,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, double 1.000000e+000) nounwind
  ret void
}

define void @single_C() nounwind {
entry:
; Missing lowering support for 'C'.
;  call void asm "foo $1,$0", "=*m,C,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, double 1.000000e+000) nounwind
  ret void
}

define void @single_e() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,e,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @single_Z() nounwind {
entry:
  call void asm "foo $1,$0", "=*m,Z,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_R() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|R|m,r|R|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_q() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|q|m,r|q|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_Q() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|Q|m,r|Q|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_a() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|{ax}|m,r|{ax}|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_b() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|{bx}|m,r|{bx}|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_c() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|{cx}|m,r|{cx}|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_d() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|{dx}|m,r|{dx},~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_S() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|{si}|m,r|{si}|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_D() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|{di}|m,r|{di}|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_A() nounwind {
entry:
  %tmp = load i32, i32* @min1, align 4
  call void asm "foo $1,$0", "=*r|A|m,r|A|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 %tmp) nounwind
  ret void
}

define void @multi_f() nounwind {
entry:
  ret void
}

define void @multi_t() nounwind {
entry:
  ret void
}

define void @multi_u() nounwind {
entry:
  ret void
}

define void @multi_y() nounwind {
entry:
  %tmp = load double, double* @din1, align 8
  call void asm "foo $1,$0", "=*r|y|m,r|y|m,~{dirflag},~{fpsr},~{flags}"(double* elementtype(double) @dout0, double %tmp) nounwind
  ret void
}

define void @multi_x() nounwind {
entry:
  %tmp = load double, double* @din1, align 8
  call void asm "foo $1,$0", "=*r|x|m,r|x|m,~{dirflag},~{fpsr},~{flags}"(double* elementtype(double) @dout0, double %tmp) nounwind
  ret void
}

define void @multi_Y0() nounwind {
entry:
  ret void
}

define void @multi_I() nounwind {
entry:
  call void asm "foo $1,$0", "=*r|m|m,r|I|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_J() nounwind {
entry:
  call void asm "foo $1,$0", "=*r|m|m,r|J|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_K() nounwind {
entry:
  call void asm "foo $1,$0", "=*r|m|m,r|K|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_L() nounwind {
entry:
; Missing lowering support for 'L'.
;  call void asm "foo $1,$0", "=*r|m|m,r|L|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_M() nounwind {
entry:
; Missing lowering support for 'M'.
;  call void asm "foo $1,$0", "=*r|m|m,r|M|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_N() nounwind {
entry:
  call void asm "foo $1,$0", "=*r|m|m,r|N|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_G() nounwind {
entry:
; Missing lowering support for 'G'.
;  call void asm "foo $1,$0", "=*r|m|m,r|G|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, double 1.000000e+000) nounwind
  ret void
}

define void @multi_C() nounwind {
entry:
; Missing lowering support for 'C'.
;  call void asm "foo $1,$0", "=*r|m|m,r|C|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, double 1.000000e+000) nounwind
  ret void
}

define void @multi_e() nounwind {
entry:
  call void asm "foo $1,$0", "=*r|m|m,r|e|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}

define void @multi_Z() nounwind {
entry:
  call void asm "foo $1,$0", "=*r|m|m,r|Z|m,~{dirflag},~{fpsr},~{flags}"(i32* elementtype(i32) @mout0, i32 1) nounwind
  ret void
}
