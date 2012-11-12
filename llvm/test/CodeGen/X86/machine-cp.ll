; RUN: llc -mtriple=x86_64-apple-macosx -mcpu=nocona < %s | FileCheck %s

; After tail duplication, two copies in an early exit BB can be cancelled out.
; rdar://10640363
define i32 @t1(i32 %a, i32 %b) nounwind  {
entry:
; CHECK: t1:
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
; CHECK: t2:
; CHECK-NOT: movdqa
  %tmp8 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 undef, i32 undef, i32 7, i32 2, i32 8, i32 undef, i32 undef , i32 undef >
  ret <8 x i16> %tmp8
}
