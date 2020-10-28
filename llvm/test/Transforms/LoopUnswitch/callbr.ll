; RUN: opt -loop-unswitch -enable-new-pm=0 %s -S | FileCheck %s

; We want to check that the loop does not get split (so only 2 callbr's not 4).
; It's ok to modify this test in the future should we allow the loop containing
; callbr to be unswitched and are able to do so correctly.

; CHECK: callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %10))
; CHECK: to label %7 [label %10]
; CHECK: callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %10))
; CHECK: to label %9 [label %10]

; CHECK-NOT: callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %10))
; CHECK-NOT: to label %7 [label %10]
; CHECK-NOT: callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %10))
; CHECK-NOT: to label %9 [label %10]
; CHECK-NOT: callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %19))
; CHECK-NOT: to label %16 [label %19]
; CHECK-NOT: callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %19))
; CHECK-NOT: to label %18 [label %19]

; This test is essentially:
; void foo(int n) {
;   for (int i = 0; i < 1000; ++i)
;     if (n) {
;       asm goto("# %l0"::::bar);
;       bar:;
;     } else {
;       asm goto("# %l0"::::baz);
;       baz:;
;     }
;}

define dso_local void @foo(i32) #0 {
  br label %2

2:                                                ; preds = %10, %1
  %.0 = phi i32 [ 0, %1 ], [ %11, %10 ]
  %3 = icmp ult i32 %.0, 1000
  br i1 %3, label %4, label %12

4:                                                ; preds = %2
  %5 = icmp eq i32 %0, 0
  br i1 %5, label %8, label %6

6:                                                ; preds = %4
  callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %10)) #0
    to label %7 [label %10]

7:                                                ; preds = %6
  br label %10

8:                                                ; preds = %4
  callbr void asm sideeffect "# ${0:l}", "X,~{dirflag},~{fpsr},~{flags}"(i8* blockaddress(@foo, %10)) #0
    to label %9 [label %10]

9:                                                ; preds = %8
  br label %10

10:                                               ; preds = %7, %6, %9, %8
  %11 = add nuw nsw i32 %.0, 1
  br label %2

12:                                               ; preds = %2
  ret void
}

