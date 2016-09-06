; RUN: llc -mtriple=x86_64-apple-macosx -mattr=+sse2 -verify-machineinstrs < %s | FileCheck %s

; After tail duplication, two copies in an early exit BB can be cancelled out.
; rdar://10640363
define i32 @t1(i32 %a, i32 %b) nounwind  {
entry:
; CHECK-LABEL: t1:
; CHECK: je [[LABEL:.*BB.*]]
  %cmp1 = icmp eq i32 %b, 0
  br i1 %cmp1, label %while.end, label %while.body

; CHECK: [[LABEL]]:
; CHECK-NOT: mov
; CHECK: ret

while.body:                                       ; preds = %entry, %while.body
  %a.addr.03 = phi i32 [ %b.addr.02, %while.body ], [ %a, %entry ]
  %b.addr.02 = phi i32 [ %rem, %while.body ], [ %b, %entry ]
  %rem = srem i32 %a.addr.03, %b.addr.02
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %a.addr.0.lcssa = phi i32 [ %a, %entry ], [ %b.addr.02, %while.body ]
  ret i32 %a.addr.0.lcssa
}

; Two movdqa (from phi-elimination) in the entry BB cancels out.
; rdar://10428165
define <8 x i16> @t2(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
; CHECK-LABEL: t2:
; CHECK-NOT: movdqa
  %tmp8 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 undef, i32 undef, i32 7, i32 2, i32 8, i32 undef, i32 undef , i32 undef >
  ret <8 x i16> %tmp8
}

define i32 @t3(i64 %a, i64 %b) nounwind  {
entry:
; CHECK-LABEL: t3:
; CHECK: je [[LABEL:.*BB.*]]
  %cmp1 = icmp eq i64 %b, 0
  br i1 %cmp1, label %while.end, label %while.body

; CHECK: [[LABEL]]:
; CHECK-NOT: mov
; CHECK: ret

while.body:                                       ; preds = %entry, %while.body
  %a.addr.03 = phi i64 [ %b.addr.02, %while.body ], [ %a, %entry ]
  %b.addr.02 = phi i64 [ %rem, %while.body ], [ %b, %entry ]
  %rem = srem i64 %a.addr.03, %b.addr.02
  %cmp = icmp eq i64 %rem, 0
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body, %entry
  %a.addr.0.lcssa = phi i64 [ %a, %entry ], [ %b.addr.02, %while.body ]
  %t = trunc i64 %a.addr.0.lcssa to i32
  ret i32 %t
}

; Check that copy propagation does not kill thing like:
; dst = copy src <-- do not kill that.
; ... = op1 dst<undef>
; ... = op2 dst <-- this is used here.
;
; CHECK-LABEL: foo:
; CHECK: psllw $7,
; CHECK: psllw $7, [[SRC1:%xmm[0-9]+]]
; CHECK-NEXT: pand {{.*}}(%rip), [[SRC1]]
; CHECK-NEXT: pcmpgtb [[SRC1]], [[SRC2:%xmm[0-9]+]]
; CHECK-NEXT: pand %xmm{{[0-9]+}}, [[SRC2]]
; CHECK-NEXT: movdqa [[SRC2]], [[CPY1:%xmm[0-9]+]]
; CHECK-NEXT: punpcklbw %xmm{{[0-9]+}}, [[CPY1]]
; Check that CPY1 is not redefined.
; CHECK-NOT:  , [[CPY1]]
; CHECK: punpckhwd %xmm{{[0-9]+}}, [[CPY1]]
; CHECK-NEXT: pslld $31, [[CPY1]]
; CHECK-NEXT: psrad $31, [[CPY1]]
; CHECK: punpckhbw %xmm{{[0-9]+}}, [[CPY2:%xmm[0-9]+]]
; Check that CPY2 is not redefined.
; CHECK-NOT:  , [[CPY2]]
; CHECK: punpckhwd %xmm{{[0-9]+}}, [[CPY2]]
; CHECK-NEXT: pslld $31, [[CPY2]]
; CHECK-NEXT: psrad $31, [[CPY2]]
define <16 x float> @foo(<16 x float> %x) {
bb:
  %v3 = icmp slt <16 x i32> undef, zeroinitializer
  %v14 = zext <16 x i1> %v3 to <16 x i32>
  %v16 = fcmp olt <16 x float> %x, zeroinitializer
  %v17 = sext <16 x i1> %v16 to <16 x i32>
  %v18 = zext <16 x i1> %v16 to <16 x i32>
  %v19 = xor <16 x i32> %v14, %v18
  %v20 = or <16 x i32> %v17, undef
  %v21 = fptosi <16 x float> %x to <16 x i32>
  %v22 = sitofp <16 x i32> %v21 to <16 x float>
  %v69 = fcmp ogt <16 x float> %v22, zeroinitializer
  %v75 = and <16 x i1> %v69, %v3
  %v77 = bitcast <16 x float> %v22 to <16 x i32>
  %v79 = sext <16 x i1> %v75 to <16 x i32>
  %v80 = and <16 x i32> undef, %v79
  %v81 = xor <16 x i32> %v77, %v80
  %v82 = and <16 x i32> undef, %v81
  %v83 = xor <16 x i32> %v19, %v82
  %v84 = and <16 x i32> %v83, %v20
  %v85 = xor <16 x i32> %v19, %v84
  %v86 = bitcast <16 x i32> %v85 to <16 x float>
  ret <16 x float> %v86
}
