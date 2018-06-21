; RUN: opt %loadPolly -polly-stmt-granularity=bb -S -polly-scops -analyze \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.IP = type { i32****, i32***, %struct.P, %struct.S, %struct.m }
%struct.P = type { i32 }
%struct.S = type { i32 }
%struct.D = type { i32 }
%struct.B = type { i32 }
%struct.E = type { i32 }
%struct.s = type { i32 }
%struct.M = type { i32 }
%struct.C = type { i32 }
%struct.T = type { i32 }
%struct.R = type { i32 }
%struct.m = type { i32 }
%struct.d = type { i32 }


; Verify that we do not invariant load hoist very complex conditions.

; CHECK:      Statements {
; CHECK-NEXT: 	Stmt_entry_split
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [block_y, block_x] -> { Stmt_entry_split[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [block_y, block_x] -> { Stmt_entry_split[] -> [] };
; CHECK-NEXT:         ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [block_y, block_x] -> { Stmt_entry_split[] -> MemRef4[o0] : (-3 <= block_y < 0 and block_x <= -4 and -8 + block_x - 4o0 <= 8*floor((-1 + block_x)/8) <= -5 + block_x - 4o0) or (-3 <= block_y < 0 and block_x >= 0 and -3 + block_x - 4o0 <= 8*floor((block_x)/8) <= block_x - 4o0) or (block_y <= -4 and block_x <= -4 and -16 + block_x - 4o0 - 8*floor((-1 + block_x)/8) + 8*floor((-1 + block_y)/4) <= 16*floor((-1 + block_y)/8) <= -13 + block_x - 4o0 - 8*floor((-1 + block_x)/8) + 8*floor((-1 + block_y)/4)) or (block_y <= -4 and block_x >= 0 and -11 + block_x - 4o0 - 8*floor((block_x)/8) + 8*floor((-1 + block_y)/4) <= 16*floor((-1 + block_y)/8) <= -8 + block_x - 4o0 - 8*floor((block_x)/8) + 8*floor((-1 + block_y)/4)) or (block_y >= 0 and block_x <= -4 and -8 + block_x - 4o0 - 8*floor((-1 + block_x)/8) + 8*floor((block_y)/4) <= 16*floor((block_y)/8) <= -5 + block_x - 4o0 - 8*floor((-1 + block_x)/8) + 8*floor((block_y)/4)) or (block_y >= 0 and block_x >= 0 and -3 + block_x - 4o0 - 8*floor((block_x)/8) + 8*floor((block_y)/4) <= 16*floor((block_y)/8) <= block_x - 4o0 - 8*floor((block_x)/8) + 8*floor((block_y)/4)) or (4*floor((block_y)/8) = -o0 + 2*floor((block_y)/4) and block_y >= 0 and -3 <= block_x < 0 and 4*floor((block_y)/4) >= -7 + block_y + 2o0 and 4*floor((block_y)/4) <= block_y + 2o0) or (4*floor((-1 + block_y)/8) = -2 - o0 + 2*floor((-1 + block_y)/4) and block_y <= -4 and -3 <= block_x < 0 and 4*floor((-1 + block_y)/4) >= -4 + block_y + 2o0 and 4*floor((-1 + block_y)/4) <= 3 + block_y + 2o0); Stmt_entry_split[] -> MemRef4[0] : -3 <= block_y < 0 and -3 <= block_x < 0 };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [block_y, block_x] -> { Stmt_entry_split[] -> MemRef0[] };
; CHECK-NEXT: }

@img = external global %struct.IP*, align 8

; Function Attrs: nounwind uwtable
define void @dct_luma(i32 %block_x, i32 %block_y) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  %div = sdiv i32 %block_x, 4
  %div1 = sdiv i32 %block_y, 4
  %rem = srem i32 %div1, 2
  %mul4 = shl nsw i32 %rem, 1
  %rem5 = srem i32 %div, 2
  %add6 = add nsw i32 %mul4, %rem5
  %idxprom = sext i32 %add6 to i64
  %0 = load %struct.IP*, %struct.IP** @img, align 8
  %cofAC = getelementptr inbounds %struct.IP, %struct.IP* %0, i32 0, i32 0
  %1 = load i32****, i32***** %cofAC, align 8
  %arrayidx = getelementptr inbounds i32***, i32**** %1, i64 0
  %2 = load i32***, i32**** %arrayidx, align 8
  %arrayidx8 = getelementptr inbounds i32**, i32*** %2, i64 %idxprom
  %3 = load i32**, i32*** %arrayidx8, align 8
  %mb_data = getelementptr inbounds %struct.IP, %struct.IP* %0, i64 0, i32 4
  %4 = load %struct.m, %struct.m* %mb_data, align 8
  br i1 false, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry.split
  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry.split
  %5 = phi i1 [ false, %entry.split ], [ undef, %land.rhs ]
  br i1 %5, label %for.cond104.preheader, label %for.cond34.preheader

for.cond34.preheader:                             ; preds = %land.end
  ret void

for.cond104.preheader:                            ; preds = %land.end
  ret void
}
