; RUN: opt -passes=indvars -disable-output %s

; Check that SCEV complexity comparisons in indvars avoid exponential work.
; The three multiplication chains repeatedly reuse each other's previous values,
; forming shared subexpressions. Without caching comparison results, recursive
; comparisons revisit the same pairs of SCEVs many times.
;
; This is a compile-time regression test: indvars should complete quickly.
; No particular transformed IR is required.

define void @test(i32 %a, i32 %b) {
entry:
  br label %loop

loop:
  %iv = phi i32 [ %next, %loop ], [ %a, %entry ]
  %a0 = mul i32 %iv, %b
  %b0 = mul i32 %a, %iv
  %c0 = mul i32 %b, %a
  %a1 = mul i32 %a0, %c0
  %b1 = mul i32 %b0, %a0
  %c1 = mul i32 %c0, %b0
  %a2 = mul i32 %a1, %c1
  %b2 = mul i32 %b1, %a1
  %c2 = mul i32 %c1, %b1
  %a3 = mul i32 %a2, %c2
  %b3 = mul i32 %b2, %a2
  %c3 = mul i32 %c2, %b2
  %a4 = mul i32 %a3, %c3
  %b4 = mul i32 %b3, %a3
  %c4 = mul i32 %c3, %b3
  %a5 = mul i32 %a4, %c4
  %b5 = mul i32 %b4, %a4
  %c5 = mul i32 %c4, %b4
  %a6 = mul i32 %a5, %c5
  %b6 = mul i32 %b5, %a5
  %c6 = mul i32 %c5, %b5
  %a7 = mul i32 %a6, %c6
  %b7 = mul i32 %b6, %a6
  %c7 = mul i32 %c6, %b6
  %a8 = mul i32 %a7, %c7
  %b8 = mul i32 %b7, %a7
  %c8 = mul i32 %c7, %b7
  %a9 = mul i32 %a8, %c8
  %b9 = mul i32 %b8, %a8
  %c9 = mul i32 %c8, %b8
  %a10 = mul i32 %a9, %c9
  %b10 = mul i32 %b9, %a9
  %c10 = mul i32 %c9, %b9
  %a11 = mul i32 %a10, %c10
  %b11 = mul i32 %b10, %a10
  %c11 = mul i32 %c10, %b10
  %a12 = mul i32 %a11, %c11
  %b12 = mul i32 %b11, %a11
  %c12 = mul i32 %c11, %b11
  %a13 = mul i32 %a12, %c12
  %b13 = mul i32 %b12, %a12
  %c13 = mul i32 %c12, %b12
  %a14 = mul i32 %a13, %c13
  %b14 = mul i32 %b13, %a13
  %c14 = mul i32 %c13, %b13
  %a15 = mul i32 %a14, %c14
  %b15 = mul i32 %b14, %a14
  %c15 = mul i32 %c14, %b14
  %a16 = mul i32 %a15, %c15
  %b16 = mul i32 %b15, %a15
  %c16 = mul i32 %c15, %b15
  %a17 = mul i32 %a16, %c16
  %b17 = mul i32 %b16, %a16
  %c17 = mul i32 %c16, %b16
  %a18 = mul i32 %a17, %c17
  %b18 = mul i32 %b17, %a17
  %c18 = mul i32 %c17, %b17
  %a19 = mul i32 %a18, %c18
  %b19 = mul i32 %b18, %a18
  %c19 = mul i32 %c18, %b18
  %a20 = mul i32 %a19, %c19
  %b20 = mul i32 %b19, %a19
  %c20 = mul i32 %c19, %b19
  %a21 = mul i32 %a20, %c20
  %b21 = mul i32 %b20, %a20
  %c21 = mul i32 %c20, %b20
  %a22 = mul i32 %a21, %c21
  %b22 = mul i32 %b21, %a21
  %c22 = mul i32 %c21, %b21
  %a23 = mul i32 %a22, %c22
  %b23 = mul i32 %b22, %a22
  %c23 = mul i32 %c22, %b22
  %a24 = mul i32 %a23, %c23
  %b24 = mul i32 %b23, %a23
  %c24 = mul i32 %c23, %b23
  %a25 = mul i32 %a24, %c24
  %b25 = mul i32 %b24, %a24
  %c25 = mul i32 %c24, %b24
  %a26 = mul i32 %a25, %c25
  %b26 = mul i32 %b25, %a25
  %c26 = mul i32 %c25, %b25
  %a27 = mul i32 %a26, %c26
  %b27 = mul i32 %b26, %a26
  %c27 = mul i32 %c26, %b26
  %a28 = mul i32 %a27, %c27
  %b28 = mul i32 %b27, %a27
  %c28 = mul i32 %c27, %b27
  %a29 = mul i32 %a28, %c28
  %b29 = mul i32 %b28, %a28
  %c29 = mul i32 %c28, %b28
  %a30 = mul i32 %a29, %c29
  %b30 = mul i32 %b29, %a29
  %c30 = mul i32 %c29, %b29
  %a31 = mul i32 %a30, %c30
  %b31 = mul i32 %b30, %a30
  %c31 = mul i32 %c30, %b30
  %a32 = mul i32 %a31, %c31
  %b32 = mul i32 %b31, %a31
  %c32 = mul i32 %c31, %b31
  %a33 = mul i32 %a32, %c32
  %b33 = mul i32 %b32, %a32
  %c33 = mul i32 %c32, %b32
  %a34 = mul i32 %a33, %c33
  %b34 = mul i32 %b33, %a33
  %c34 = mul i32 %c33, %b33
  %a35 = mul i32 %a34, %c34
  %b35 = mul i32 %b34, %a34
  %c35 = mul i32 %c34, %b34
  %a36 = mul i32 %a35, %c35
  %c36 = mul i32 %c35, %b35
  %next = mul i32 %a36, %c36
  ; The untaken backedge still forms a loop that indvars analyzes.
  br i1 false, label %loop, label %exit

exit:
  ret void
}
