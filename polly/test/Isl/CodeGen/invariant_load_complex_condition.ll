; RUN: opt %loadPolly -S -polly-codegen -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; Extracted from h246 in SPEC 2006.
;
; TODO: We check that we do compile this benchmark in reasonable time.
;       To do so we currently bail out due to the complex access range
;       (multiple modulos) of the invariant load.
;
; FIXME: We should not bail with a false RTC here.
;
; CHECK-LABEL: polly.preload.begin:
; CHECK-NOT:     br i1
; CHECK-NOT:     br label
; CHECK:         br i1 false, label %polly.start, label %entry.split
;
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
