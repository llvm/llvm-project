; RUN: opt %loadNPMPolly -polly-invariant-load-hoisting=true '-passes=print<polly-optree>' -disable-output < %s | FileCheck %s -match-full-lines
;

define void @foo(ptr %A, i32 %p, ptr %B) {
start:
  br label %branch

branch:
  %cmp = icmp eq i32 %p, 1024
  br i1 %cmp, label %next, label %end

next:
  %val = load float, ptr %A
  %fcmp = fcmp oeq float %val, 41.0
  br label %nonaffine

nonaffine:
  br i1 %fcmp, label %a, label %b

a:
  store float %val, ptr %A
  br label %end

b:
  store float 1.0, ptr %A
  br label %end

end:
  ret void
}

; CHECK:      After statements {
; CHECK-NEXT:     Stmt_next
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p] -> { Stmt_next[] -> MemRef_A[0] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p] -> { Stmt_next[] -> MemRef_fcmp[] };
; CHECK-NEXT:             MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:                 [p] -> { Stmt_next[] -> MemRef_val[] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load float, ptr %A, align 4
; CHECK-NEXT:                   %fcmp = fcmp oeq float %val, 4.100000e+01
; CHECK-NEXT:             }
; CHECK-NEXT:     Stmt_nonaffine__TO__end
; CHECK-NEXT:             ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 ;
; CHECK-NEXT:            new: [p] -> { Stmt_nonaffine__TO__end[] -> MemRef_A[0] };
; CHECK-NEXT:             MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p] -> { Stmt_nonaffine__TO__end[] -> MemRef_A[0] };
; CHECK-NEXT:             MayWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                 [p] -> { Stmt_nonaffine__TO__end[] -> MemRef_A[0] };
; CHECK-NEXT:             Instructions {
; CHECK-NEXT:                   %val = load float, ptr %A, align 4
; CHECK-NEXT:                   %val = load float, ptr %A, align 4
; CHECK-NEXT:                   %fcmp = fcmp oeq float %val, 4.100000e+01
; CHECK-NEXT:             }
; CHECK-NEXT: }
