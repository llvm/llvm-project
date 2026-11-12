; RUN: rm -f %t.ll
; RUN: cat %s | sed -e 's/ 0, %2 / 1, %2 /' > %t.ll
; RUN: not llvm-diff %s %t.ll 2>&1 | FileCheck %s

; CHECK:       in function foo:
; CHECK-NEXT:   in block %6 / %6:
; CHECK-NEXT:    >   %7 = phi i32 [ 1, %2 ], [ -1, %1 ]
; CHECK-NEXT:    >   ret i32 %7
; CHECK-NEXT:    <   %7 = phi i32 [ 0, %2 ], [ -1, %1 ]
; CHECK-NEXT:    <   ret i32 %7
define i32 @foo(i32 %0) #0 {
  callbr void asm sideeffect "", "!i,~{dirflag},~{fpsr},~{flags}"()
          to label %2 [label %6]

2:
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %6, label %4

4:
  br label %5

5:
  br label %5

6:
  %7 = phi i32 [ 0, %2 ], [ -1, %1 ]
  ret i32 %7
}
