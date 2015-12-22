; RUN: opt %loadPolly -pass-remarks-analysis="polly-scops" -polly-scops \
; RUN:     < %s 2>&1 | FileCheck %s

; We build a scop of the following form to check that the domain construction
; does not take a huge amount of time, but that we instead just bail out.
;
;       loop.header
;      /    |    \ \
;    A0    A2    A4 \
;      \  /  \  /    \
;       A1    A3      \
;      /  \  /  \     |
;    B0    B2    B4   |
;      \  /  \  /     |
;       B1    B3      ^
;      /  \  /  \     |
;    C0    C2    C4   |
;      \  /  \  /    /
;       C1    C3    /
;        \   /     /
;     loop backedge

; CHECK: Low number of domain conjuncts assumption: {  : 1 = 0 }

define void @foo(float* %A, float* %B, float* %C, float* %D, float* %E,
                 i64 %A1.p, i64 %A2.p, i64 %A3.p,
                 i64 %B1.p, i64 %B2.p, i64 %B3.p,
                 i64 %C1.p, i64 %C2.p, i64 %C3.p,
                 i64 %D1.p, i64 %D2.p, i64 %D3.p,
                 i64 %E1.p, i64 %E2.p, i64 %E3.p) {
entry:
  br label %loop.header

loop.header:
  %indvar = phi i64 [0, %entry], [%indvar.next, %loop.backedge]
  switch i2 0, label %A0 [i2 1, label %A2 i2 2, label %A4]

A0:
  %val.A0 = load float, float* %A
  store float %val.A0, float* %A
  br label %A1

A2:
  %val.A2 = load float, float* %A
  store float %val.A2, float* %A
  %A2.cmp = icmp eq i64 %A2.p, 0
  br i1 %A2.cmp, label %A1, label %A3

A4:
  %val.A4 = load float, float* %A
  store float %val.A4, float* %A
  br label %A3

A1:
  %val.A1 = load float, float* %A
  store float %val.A1, float* %A
  %A1.cmp = icmp eq i64 %A1.p, 0
  br i1 %A1.cmp, label %B0, label %B2

A3:
  %val.A3 = load float, float* %A
  store float %val.A3, float* %A
  %A3.cmp = icmp eq i64 %A3.p, 0
  br i1 %A3.cmp, label %B2, label %B4

B0:
  %val.B0 = load float, float* %B
  store float %val.B0, float* %B
  br label %B1

B2:
  %val.B2 = load float, float* %B
  store float %val.B2, float* %B
  %B2.cmp = icmp eq i64 %B2.p, 0
  br i1 %B2.cmp, label %B1, label %B3

B4:
  %val.B4 = load float, float* %B
  store float %val.B4, float* %B
  br label %B3

B1:
  %val.B1 = load float, float* %B
  store float %val.B1, float* %B
  %B1.cmp = icmp eq i64 %B1.p, 0
  br i1 %B1.cmp, label %C0, label %C2

B3:
  %val.B3 = load float, float* %A
  store float %val.B3, float* %A
  %B3.cmp = icmp eq i64 %A3.p, 0
  br i1 %B3.cmp, label %C2, label %C4

C0:
  %val.C0 = load float, float* %C
  store float %val.C0, float* %C
  br label %C1

C2:
  %val.C2 = load float, float* %C
  store float %val.C2, float* %C
  %C2.cmp = icmp eq i64 %C2.p, 0
  br i1 %C2.cmp, label %C1, label %C3

C4:
  %val.C4 = load float, float* %C
  store float %val.C4, float* %C
  br label %C3

C1:
  %val.C1 = load float, float* %C
  store float %val.C1, float* %C
  %C1.cmp = icmp eq i64 %C1.p, 0
  br i1 %C1.cmp, label %D0, label %D2

C3:
  %val.C3 = load float, float* %C
  store float %val.C3, float* %C
  %C3.cmp = icmp eq i64 %C3.p, 0
  br i1 %C3.cmp, label %D2, label %D4

D0:
  %val.D0 = load float, float* %D
  store float %val.D0, float* %D
  br label %D1

D2:
  %val.D2 = load float, float* %D
  store float %val.D2, float* %D
  %D2.cmp = icmp eq i64 %D2.p, 0
  br i1 %D2.cmp, label %D1, label %D3

D4:
  %val.D4 = load float, float* %D
  store float %val.D4, float* %D
  br label %D3

D1:
  %val.D1 = load float, float* %D
  store float %val.D1, float* %D
  %D1.cmp = icmp eq i64 %D1.p, 0
  br i1 %D1.cmp, label %E0, label %E2

D3:
  %val.D3 = load float, float* %D
  store float %val.D3, float* %D
  %D3.cmp = icmp eq i64 %D3.p, 0
  br i1 %D3.cmp, label %E2, label %E4

E0:
  %val.E0 = load float, float* %E
  store float %val.E0, float* %E
  br label %E1

E2:
  %val.E2 = load float, float* %E
  store float %val.E2, float* %E
  %E2.cmp = icmp eq i64 %E2.p, 0
  br i1 %E2.cmp, label %E1, label %E3

E4:
  %val.E4 = load float, float* %E
  store float %val.E4, float* %E
  br label %E3

E1:
  %val.E1 = load float, float* %E
  store float %val.E1, float* %E
  %E1.cmp = icmp eq i64 %E1.p, 0
  br i1 %E1.cmp, label %F0, label %F2

E3:
  %val.E3 = load float, float* %E
  store float %val.E3, float* %E
  %E3.cmp = icmp eq i64 %E3.p, 0
  br i1 %E3.cmp, label %F2, label %F4

F0:
  br label %loop.backedge

F2:
  br label %loop.backedge

F4:
  br label %loop.backedge

loop.backedge:
  %indvar.next = add i64 %indvar, 1
  %cmp = icmp ne i64 %indvar, 1000
  br i1 %cmp, label %loop.header, label %exit

exit:
  ret void

}
