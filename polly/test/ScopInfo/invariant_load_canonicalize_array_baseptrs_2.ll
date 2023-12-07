; RUN: opt %loadPolly -polly-print-scops -disable-output < %s \
; RUN:  -polly-invariant-load-hoisting \
; RUN:  | FileCheck %s

; Make sure we choose a canonical element that is not the first invariant load,
; but the first that is an array base pointer.

; CHECK:     Statements {
; CHECK-NEXT:     	Stmt_body0
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_body0[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_body0[i0] -> [i0, 0] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body0[i0] -> MemRef_X[0] };
; CHECK-NEXT:     	Stmt_body1
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_body1[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_body1[i0] -> [i0, 1] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body1[i0] -> MemRef_baseB[0] };
; CHECK-NEXT:     	Stmt_body2
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_body2[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_body2[i0] -> [i0, 2] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body2[i0] -> MemRef_X[0] };
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body2[i0] -> MemRef_ptr[] };
; CHECK-NEXT:     	Stmt_body3
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_body3[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_body3[i0] -> [i0, 3] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body3[i0] -> MemRef_baseB[0] };
; CHECK-NEXT:     	Stmt_body4
; CHECK-NEXT:             Domain :=
; CHECK-NEXT:                 { Stmt_body4[i0] : 0 <= i0 <= 1022 };
; CHECK-NEXT:             Schedule :=
; CHECK-NEXT:                 { Stmt_body4[i0] -> [i0, 4] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 { Stmt_body4[i0] -> MemRef_X[0] };
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 { Stmt_body4[i0] -> MemRef_ptr[] };
; CHECK-NEXT:     }

define void @foo(ptr %A, ptr %X) {
start:
  br label %loop

loop:
  %indvar = phi i64 [0, %start], [%indvar.next, %latch]
  %indvar.next = add nsw i64 %indvar, 1
  %icmp = icmp slt i64 %indvar.next, 1024
  br i1 %icmp, label %body0, label %exit

body0:
  %ptr = load ptr, ptr %A
  store ptr %ptr, ptr %X
  br label %body1

body1:
  %baseA = load ptr, ptr %A
  store float 42.0, ptr %baseA
  br label %body2

body2:
  %ptr2 = load ptr, ptr %A
  store ptr %ptr, ptr %X
  br label %body3

body3:
  %baseB = load ptr, ptr %A
  store float 42.0, ptr %baseB
  br label %body4

body4:
  %ptr3 = load ptr, ptr %A
  store ptr %ptr, ptr %X
  br label %latch

latch:
  br label %loop

exit:
  ret void

}
