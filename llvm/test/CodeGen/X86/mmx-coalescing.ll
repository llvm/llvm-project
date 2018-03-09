; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+mmx,+sse2 | FileCheck %s

%SA = type <{ %union.anon, i32, [4 x i8], i8*, i8*, i8*, i32, [4 x i8] }>
%union.anon = type { <1 x i64> }

; Check that extra movd (copy) instructions aren't generated.

define i32 @test(%SA* %pSA, i16* %A, i32 %B, i32 %C, i32 %D, i8* %E) {
entry:
; CHECK-LABEL: test
; CHECK:       # %bb.0:
; CHECK-NEXT:  pshufw
; CHECK-NEXT:  movd
; CHECK-NOT:  movd
; CHECK-NEXT:  testl
  %shl = shl i32 1, %B
  %shl1 = shl i32 %C, %B
  %shl2 = shl i32 1, %D
  %v = getelementptr inbounds %SA, %SA* %pSA, i64 0, i32 0, i32 0
  %v0 = load <1 x i64>, <1 x i64>* %v, align 8
  %SA0 = getelementptr inbounds %SA, %SA* %pSA, i64 0, i32 1
  %v1 = load i32, i32* %SA0, align 4
  %SA1 = getelementptr inbounds %SA, %SA* %pSA, i64 0, i32 3
  %v2 = load i8*, i8** %SA1, align 8
  %SA2 = getelementptr inbounds %SA, %SA* %pSA, i64 0, i32 4
  %v3 = load i8*, i8** %SA2, align 8
  %v4 = bitcast <1 x i64> %v0 to <4 x i16>
  %v5 = bitcast <4 x i16> %v4 to x86_mmx
  %v6 = tail call x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx %v5, i8 -18)
  %v7 = bitcast x86_mmx %v6 to <4 x i16>
  %v8 = bitcast <4 x i16> %v7 to <1 x i64>
  %v9 = extractelement <1 x i64> %v8, i32 0
  %v10 = bitcast i64 %v9 to <2 x i32>
  %v11 = extractelement <2 x i32> %v10, i32 0
  %cmp = icmp eq i32 %v11, 0
  br i1 %cmp, label %if.A, label %if.B

if.A:
; CHECK: %if.A
; CHECK-NEXT:  movd
; CHECK-NEXT:  psllq
  %pa = phi <1 x i64> [ %v8, %entry ], [ %vx, %if.C ]
  %v17 = extractelement <1 x i64> %pa, i32 0
  %v18 = bitcast i64 %v17 to x86_mmx
  %v19 = tail call x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx %v18, i32 %B) #2
  %v20 = bitcast x86_mmx %v19 to i64
  %v21 = insertelement <1 x i64> undef, i64 %v20, i32 0
  %cmp3 = icmp eq i64 %v20, 0
  br i1 %cmp3, label %if.C, label %merge

if.B:
  %v34 = bitcast <1 x i64> %v8 to <4 x i16>
  %v35 = bitcast <4 x i16> %v34 to x86_mmx
  %v36 = tail call x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx %v35, i8 -18)
  %v37 = bitcast x86_mmx %v36 to <4 x i16>
  %v38 = bitcast <4 x i16> %v37 to <1 x i64>
  br label %if.C

if.C:
  %vx = phi <1 x i64> [ %v21, %if.A ], [ %v38, %if.B ]
  %cvt = bitcast <1 x i64> %vx to <2 x i32>
  %ex = extractelement <2 x i32> %cvt, i32 0
  %cmp2 = icmp eq i32 %ex, 0
  br i1 %cmp2, label %if.A, label %merge

merge:
; CHECK: %merge
; CHECK-NOT:  movd
; CHECK-NEXT:  pshufw
  %vy = phi <1 x i64> [ %v21, %if.A ], [ %vx, %if.C ]
  %v130 = bitcast <1 x i64> %vy to <4 x i16>
  %v131 = bitcast <4 x i16> %v130 to x86_mmx
  %v132 = tail call x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx %v131, i8 -18)
  %v133 = bitcast x86_mmx %v132 to <4 x i16>
  %v134 = bitcast <4 x i16> %v133 to <1 x i64>
  %v135 = extractelement <1 x i64> %v134, i32 0
  %v136 = bitcast i64 %v135 to <2 x i32>
  %v137 = extractelement <2 x i32> %v136, i32 0
  ret i32 %v137
}


declare x86_mmx @llvm.x86.sse.pshuf.w(x86_mmx, i8)
declare x86_mmx @llvm.x86.mmx.pslli.q(x86_mmx, i32)
