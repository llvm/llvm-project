; RUN: llc < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "armv7-none--gnueabihf"

%struct.s = type { float, float }
%union.t = type { [4 x float] }

; Equivalent C code:
; struct s { float a; float b; };
; float foo(float a, double b, struct s c) { return c.a; }
; Argument allocation:
; a -> s0
; b -> d1
; c -> s4, s5
; s1 is unused
; return in s0
define float @test1(float %a, double %b, %struct.s %c) {
entry:
; CHECK-LABEL: test1
; CHECK: vmov.f32  s0, s4
; CHECK-NOT: vmov.f32        s0, s1

  %result = extractvalue %struct.s %c, 0
  ret float %result
}

; Equivalent C code:
; union t { float a[4] };
; float foo(float a, double b, union s c) { return c.a[0]; }
; Argument allocation:
; a -> s0
; b -> d1
; c -> s4..s7
define float @test2(float %a, double %b, %union.t %c) #0 {
entry:
; CHECK-LABEL: test2
; CHECK: vmov.f32  s0, s4
; CHECK-NOT: vmov.f32        s0, s1

  %result = extractvalue %union.t %c, 0, 0
  ret float %result
}

; Equivalent C code:
; struct s { float a; float b; };
; float foo(float a, double b, struct s c, float d) { return d; }
; Argument allocation:
; a -> s0
; b -> d1
; c -> s4, s5
; d -> s1
; return in s0
define float @test3(float %a, double %b, %struct.s %c, float %d) {
entry:
; CHECK-LABEL: test3
; CHECK: vmov.f32  s0, s1
; CHECK-NOT: vmov.f32        s0, s5

  ret float %d
}

; Equivalent C code:
; struct s { float a; float b; };
; float foo(struct s a, struct s b) { return b.b; }
; Argument allocation:
; a -> s0, s1
; b -> s2, s3
; return in s0
define float @test4(%struct.s %a, %struct.s %b) {
entry:
; CHECK-LABEL: test4
; CHECK: vmov.f32  s0, s3

  %result = extractvalue %struct.s %b, 1
  ret float %result
}

; Equivalent C code:
; struct s { float a; float b; };
; float foo(struct s a, float b, struct s c) { return c.a; }
; Argument allocation:
; a -> s0, s1
; b -> s2
; c -> s3, s4
; return in s0
define float @test5(%struct.s %a, float %b, %struct.s %c) {
entry:
; CHECK-LABEL: test5
; CHECK: vmov.f32  s0, s3

  %result = extractvalue %struct.s %c, 0
  ret float %result
}
