; RUN: opt %s -disable-output -branch-prob -instcombine -block-freq -verify-dom-info
; RUN: opt %s -postdomtree -analyze | FileCheck --check-prefixes=CHECK-POSTDOM %s
; RUN: opt %s -passes='print<postdomtree>' 2>&1 | FileCheck --check-prefixes=CHECK-POSTDOM %s

; Demonstrate that Predicate Canonicalization (InstCombine) does not invalidate PostDomTree
; if the basic block is post-dom unreachable.

define void @test1(i24 %a, i24 %b) {
entry:
  br label %LOOP

LOOP:
  %f = icmp uge i24 %a, %b
  br i1 %f, label %B1, label %B2

B1:
  %x = add i24 %a, %b
  br label %B2

B2:
  br label %LOOP
}

; The same as @test1 except the LOOP condition canonicalized (as by instcombine).
define void @test1-canonicalized(i24 %a, i24 %b) {
entry:
  br label %LOOP

LOOP:
  %f.not = icmp ult i24 %a, %b
  br i1 %f.not, label %B2, label %B1

B1:
  %x = add i24 %a, %b
  br label %B2

B2:
  br label %LOOP
}

; The same as @test1 but different order of B1 and B2 in the function.
; The different order makes PostDomTree different in presense of postdom
; unreachable blocks.
define void @test2(i24 %a, i24 %b) {
entry:
  br label %LOOP

LOOP:
  %f = icmp uge i24 %a, %b
  br i1 %f, label %B1, label %B2

B2:
  br label %LOOP

B1:
  %x = add i24 %a, %b
  br label %B2
}

; The same as @test2 except the LOOP condition canonicalized (as by instcombine).
define void @test2-canonicalized(i24 %a, i24 %b) {
entry:
  br label %LOOP

LOOP:
  %f.not = icmp ult i24 %a, %b
  br i1 %f.not, label %B2, label %B1

B2:
  br label %LOOP

B1:
  %x = add i24 %a, %b
  br label %B2
}

; Two reverse unreachable subgraphs with RU1* and RU2* basic blocks respectively.
define void @test3(i24 %a, i24 %b, i32 %flag) {
entry:
  switch i32 %flag, label %EXIT [
    i32 1, label %RU1
    i32 2, label %RU2
    i32 3, label %RU2_B1
  ]

RU1:
  %f = icmp uge i24 %a, %b
  br label %RU1_LOOP

RU1_LOOP:
  br i1 %f, label %RU1_B1, label %RU1_B2

RU1_B1:
  %x = add i24 %a, %b
  br label %RU1_B2

RU1_B2:
  br label %RU1_LOOP

RU2:
  %f2 = icmp uge i24 %a, %b
  br i1 %f2, label %RU2_B1, label %RU2_B2

RU2_B1:
  br label %RU2_B2

RU2_B2:
  br label %RU2_B1

EXIT:
  ret void
}

; The same as @test3 except the icmp conditions are canonicalized (as by instcombine).
define void @test3-canonicalized(i24 %a, i24 %b, i32 %flag) {
entry:
  switch i32 %flag, label %EXIT [
    i32 1, label %RU1
    i32 2, label %RU2
    i32 3, label %RU2_B1
  ]

RU1:
  %f.not = icmp ult i24 %a, %b
  br label %RU1_LOOP

RU1_LOOP:
  br i1 %f.not, label %RU1_B2, label %RU1_B1

RU1_B1:
  %x = add i24 %a, %b
  br label %RU1_B2

RU1_B2:
  br label %RU1_LOOP

RU2:
  %f2.not = icmp ult i24 %a, %b
  br i1 %f2.not, label %RU2_B2, label %RU2_B1

RU2_B1:
  br label %RU2_B2

RU2_B2:
  br label %RU2_B1

EXIT:
  ret void
}

; PostDomTrees of @test1(), @test2() and @test3() are different.
; PostDomTrees of @testX() and @testX-canonicalize() are the same.

; CHECK-POSTDOM-LABEL: test1
; CHECK-POSTDOM-NEXT: =============================--------------------------------
; CHECK-POSTDOM-NEXT: Inorder PostDominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-POSTDOM-NEXT:   [1]  <<exit node>>
; CHECK-POSTDOM-NEXT:     [2] %B1
; CHECK-POSTDOM-NEXT:       [3] %LOOP
; CHECK-POSTDOM-NEXT:         [4] %entry
; CHECK-POSTDOM-NEXT:         [4] %B2
; CHECK-POSTDOM-NEXT: Roots: %B1

; CHECK-POSTDOM-LABEL: test1-canonicalized
; CHECK-POSTDOM-NEXT: =============================--------------------------------
; CHECK-POSTDOM-NEXT: Inorder PostDominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-POSTDOM-NEXT:   [1]  <<exit node>>
; CHECK-POSTDOM-NEXT:     [2] %B1
; CHECK-POSTDOM-NEXT:       [3] %LOOP
; CHECK-POSTDOM-NEXT:         [4] %entry
; CHECK-POSTDOM-NEXT:         [4] %B2
; CHECK-POSTDOM-NEXT: Roots: %B1

; CHECK-POSTDOM-LABEL: test2
; CHECK-POSTDOM-NEXT: =============================--------------------------------
; CHECK-POSTDOM-NEXT: Inorder PostDominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-POSTDOM-NEXT:   [1]  <<exit node>>
; CHECK-POSTDOM-NEXT:     [2] %B2
; CHECK-POSTDOM-NEXT:       [3] %LOOP
; CHECK-POSTDOM-NEXT:         [4] %entry
; CHECK-POSTDOM-NEXT:       [3] %B1
; CHECK-POSTDOM-NEXT: Roots: %B2

; CHECK-POSTDOM-LABEL: test2-canonicalized
; CHECK-POSTDOM-NEXT: =============================--------------------------------
; CHECK-POSTDOM-NEXT: Inorder PostDominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-POSTDOM-NEXT:   [1]  <<exit node>>
; CHECK-POSTDOM-NEXT:     [2] %B2
; CHECK-POSTDOM-NEXT:       [3] %LOOP
; CHECK-POSTDOM-NEXT:         [4] %entry
; CHECK-POSTDOM-NEXT:       [3] %B1
; CHECK-POSTDOM-NEXT: Roots: %B2

; CHECK-POSTDOM-LABEL: test3
; CHECK-POSTDOM-NEXT:=============================--------------------------------
; CHECK-POSTDOM-NEXT:Inorder PostDominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-POSTDOM-NEXT:  [1]  <<exit node>>
; CHECK-POSTDOM-NEXT:    [2] %EXIT
; CHECK-POSTDOM-NEXT:    [2] %entry
; CHECK-POSTDOM-NEXT:    [2] %RU1_B1
; CHECK-POSTDOM-NEXT:      [3] %RU1_LOOP
; CHECK-POSTDOM-NEXT:        [4] %RU1
; CHECK-POSTDOM-NEXT:        [4] %RU1_B2
; CHECK-POSTDOM-NEXT:    [2] %RU2_B1
; CHECK-POSTDOM-NEXT:      [3] %RU2
; CHECK-POSTDOM-NEXT:      [3] %RU2_B2
; CHECK-POSTDOM-NEXT:Roots: %EXIT %RU1_B1 %RU2_B1

; CHECK-POSTDOM-LABEL: test3-canonicalized
; CHECK-POSTDOM-NEXT:=============================--------------------------------
; CHECK-POSTDOM-NEXT:Inorder PostDominator Tree: DFSNumbers invalid: 0 slow queries.
; CHECK-POSTDOM-NEXT:  [1]  <<exit node>>
; CHECK-POSTDOM-NEXT:    [2] %EXIT
; CHECK-POSTDOM-NEXT:    [2] %entry
; CHECK-POSTDOM-NEXT:    [2] %RU1_B1
; CHECK-POSTDOM-NEXT:      [3] %RU1_LOOP
; CHECK-POSTDOM-NEXT:        [4] %RU1
; CHECK-POSTDOM-NEXT:        [4] %RU1_B2
; CHECK-POSTDOM-NEXT:    [2] %RU2_B1
; CHECK-POSTDOM-NEXT:      [3] %RU2
; CHECK-POSTDOM-NEXT:      [3] %RU2_B2
; CHECK-POSTDOM-NEXT:Roots: %EXIT %RU1_B1 %RU2_B1
