; RUN: llc < %s -block-placement-exit-block-bias=20 -no-phi-elim-live-out-early-exit | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; This is longest_match, the hot function from zlib's deflate implementation.

%struct.internal_state = type { ptr, i32, ptr, i64, ptr, i32, i32, ptr, i32, i8, i32, i32, i32, i32, ptr, i64, ptr, ptr, i32, i32, i32, i32, i32, i64, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, [573 x %struct.ct_data_s], [61 x %struct.ct_data_s], [39 x %struct.ct_data_s], %struct.tree_desc_s, %struct.tree_desc_s, %struct.tree_desc_s, [16 x i16], [573 x i32], i32, i32, [573 x i8], ptr, i32, i32, ptr, i64, i64, i32, i32, i16, i32, i64 }
%struct.z_stream_s = type { ptr, i32, i64, ptr, i32, i64, ptr, ptr, ptr, ptr, ptr, i32, i64, i64 }
%struct.gz_header_s = type { i32, i64, i32, i32, ptr, i32, i32, ptr, i32, ptr, i32, i32, i32 }
%struct.ct_data_s = type { %union.anon, %union.anon.0 }
%union.anon = type { i16 }
%union.anon.0 = type { i16 }
%struct.tree_desc_s = type { ptr, i32, ptr }
%struct.static_tree_desc_s = type { i32 }

; CHECK-LABEL: longest_match:

; Verify that there are no spills or reloads in the loop exit block. This loop
; is mostly cold, only %do.cond125 and %land.rhs131 are hot.
; CHECK: %do.cond125
; CHECK-NOT: {{Spill|Reload}}
; CHECK: jbe

; Verify that block placement doesn't destroy source order. It's important that
; the two hot blocks are laid out close to each other.
; CHECK-NEXT: %land.rhs131
; CHECK: jne
define i32 @longest_match(ptr nocapture %s, i32 %cur_match) nounwind {
entry:
  %max_chain_length = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 31
  %0 = load i32, ptr %max_chain_length, align 4
  %window = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 14
  %1 = load ptr, ptr %window, align 8
  %strstart = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 27
  %2 = load i32, ptr %strstart, align 4
  %idx.ext = zext i32 %2 to i64
  %add.ptr = getelementptr inbounds i8, ptr %1, i64 %idx.ext
  %prev_length = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 30
  %3 = load i32, ptr %prev_length, align 4
  %nice_match1 = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 36
  %4 = load i32, ptr %nice_match1, align 4
  %w_size = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 11
  %5 = load i32, ptr %w_size, align 4
  %sub = add i32 %5, -262
  %cmp = icmp ugt i32 %2, %sub
  %sub6 = sub i32 %2, %sub
  %sub6. = select i1 %cmp, i32 %sub6, i32 0
  %prev7 = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 16
  %6 = load ptr, ptr %prev7, align 8
  %w_mask = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 13
  %7 = load i32, ptr %w_mask, align 4
  %add.ptr11.sum = add i64 %idx.ext, 258
  %add.ptr12 = getelementptr inbounds i8, ptr %1, i64 %add.ptr11.sum
  %sub13 = add nsw i32 %3, -1
  %idxprom = sext i32 %sub13 to i64
  %add.ptr.sum = add i64 %idxprom, %idx.ext
  %arrayidx = getelementptr inbounds i8, ptr %1, i64 %add.ptr.sum
  %8 = load i8, ptr %arrayidx, align 1
  %idxprom14 = sext i32 %3 to i64
  %add.ptr.sum213 = add i64 %idxprom14, %idx.ext
  %arrayidx15 = getelementptr inbounds i8, ptr %1, i64 %add.ptr.sum213
  %9 = load i8, ptr %arrayidx15, align 1
  %good_match = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 35
  %10 = load i32, ptr %good_match, align 4
  %cmp17 = icmp ult i32 %3, %10
  %shr = lshr i32 %0, 2
  %chain_length.0 = select i1 %cmp17, i32 %0, i32 %shr
  %lookahead = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 29
  %11 = load i32, ptr %lookahead, align 4
  %cmp18 = icmp ugt i32 %4, %11
  %. = select i1 %cmp18, i32 %11, i32 %4
  %match_start = getelementptr inbounds %struct.internal_state, ptr %s, i64 0, i32 28
  %add.ptr.sum217 = add i64 %idx.ext, 1
  %arrayidx44 = getelementptr inbounds i8, ptr %1, i64 %add.ptr.sum217
  %add.ptr.sum218 = add i64 %idx.ext, 2
  %add.ptr50 = getelementptr inbounds i8, ptr %1, i64 %add.ptr.sum218
  %sub.ptr.lhs.cast = ptrtoint ptr %add.ptr12 to i64
  br label %do.body

do.body:                                          ; preds = %land.rhs131, %entry
  %best_len.0 = phi i32 [ %best_len.1, %land.rhs131 ], [ %3, %entry ]
  %chain_length.1 = phi i32 [ %dec, %land.rhs131 ], [ %chain_length.0, %entry ]
  %cur_match.addr.0 = phi i32 [ %conv128, %land.rhs131 ], [ %cur_match, %entry ]
  %scan_end1.0 = phi i8 [ %scan_end1.1, %land.rhs131 ], [ %8, %entry ]
  %scan_end.0 = phi i8 [ %scan_end.1, %land.rhs131 ], [ %9, %entry ]
  %idx.ext23 = zext i32 %cur_match.addr.0 to i64
  %add.ptr24 = getelementptr inbounds i8, ptr %1, i64 %idx.ext23
  %idxprom25 = sext i32 %best_len.0 to i64
  %add.ptr24.sum = add i64 %idx.ext23, %idxprom25
  %arrayidx26 = getelementptr inbounds i8, ptr %1, i64 %add.ptr24.sum
  %12 = load i8, ptr %arrayidx26, align 1
  %cmp28 = icmp eq i8 %12, %scan_end.0
  br i1 %cmp28, label %lor.lhs.false, label %do.cond125

lor.lhs.false:                                    ; preds = %do.body
  %sub30 = add nsw i32 %best_len.0, -1
  %idxprom31 = sext i32 %sub30 to i64
  %add.ptr24.sum214 = add i64 %idx.ext23, %idxprom31
  %arrayidx32 = getelementptr inbounds i8, ptr %1, i64 %add.ptr24.sum214
  %13 = load i8, ptr %arrayidx32, align 1
  %cmp35 = icmp eq i8 %13, %scan_end1.0
  br i1 %cmp35, label %lor.lhs.false37, label %do.cond125

lor.lhs.false37:                                  ; preds = %lor.lhs.false
  %14 = load i8, ptr %add.ptr24, align 1
  %15 = load i8, ptr %add.ptr, align 1
  %cmp40 = icmp eq i8 %14, %15
  br i1 %cmp40, label %lor.lhs.false42, label %do.cond125

lor.lhs.false42:                                  ; preds = %lor.lhs.false37
  %add.ptr24.sum215 = add i64 %idx.ext23, 1
  %incdec.ptr = getelementptr inbounds i8, ptr %1, i64 %add.ptr24.sum215
  %16 = load i8, ptr %incdec.ptr, align 1
  %17 = load i8, ptr %arrayidx44, align 1
  %cmp46 = icmp eq i8 %16, %17
  br i1 %cmp46, label %if.end49, label %do.cond125

if.end49:                                         ; preds = %lor.lhs.false42
  %incdec.ptr.sum = add i64 %idx.ext23, 2
  %incdec.ptr51 = getelementptr inbounds i8, ptr %1, i64 %incdec.ptr.sum
  br label %do.cond

do.cond:                                          ; preds = %land.lhs.true100, %if.end49
  %match.0 = phi ptr [ %incdec.ptr51, %if.end49 ], [ %incdec.ptr103, %land.lhs.true100 ]
  %scan.1 = phi ptr [ %add.ptr50, %if.end49 ], [ %incdec.ptr101, %land.lhs.true100 ]
  %incdec.ptr53 = getelementptr inbounds i8, ptr %scan.1, i64 1
  %18 = load i8, ptr %incdec.ptr53, align 1
  %incdec.ptr55 = getelementptr inbounds i8, ptr %match.0, i64 1
  %19 = load i8, ptr %incdec.ptr55, align 1
  %cmp57 = icmp eq i8 %18, %19
  br i1 %cmp57, label %land.lhs.true, label %do.end

land.lhs.true:                                    ; preds = %do.cond
  %incdec.ptr59 = getelementptr inbounds i8, ptr %scan.1, i64 2
  %20 = load i8, ptr %incdec.ptr59, align 1
  %incdec.ptr61 = getelementptr inbounds i8, ptr %match.0, i64 2
  %21 = load i8, ptr %incdec.ptr61, align 1
  %cmp63 = icmp eq i8 %20, %21
  br i1 %cmp63, label %land.lhs.true65, label %do.end

land.lhs.true65:                                  ; preds = %land.lhs.true
  %incdec.ptr66 = getelementptr inbounds i8, ptr %scan.1, i64 3
  %22 = load i8, ptr %incdec.ptr66, align 1
  %incdec.ptr68 = getelementptr inbounds i8, ptr %match.0, i64 3
  %23 = load i8, ptr %incdec.ptr68, align 1
  %cmp70 = icmp eq i8 %22, %23
  br i1 %cmp70, label %land.lhs.true72, label %do.end

land.lhs.true72:                                  ; preds = %land.lhs.true65
  %incdec.ptr73 = getelementptr inbounds i8, ptr %scan.1, i64 4
  %24 = load i8, ptr %incdec.ptr73, align 1
  %incdec.ptr75 = getelementptr inbounds i8, ptr %match.0, i64 4
  %25 = load i8, ptr %incdec.ptr75, align 1
  %cmp77 = icmp eq i8 %24, %25
  br i1 %cmp77, label %land.lhs.true79, label %do.end

land.lhs.true79:                                  ; preds = %land.lhs.true72
  %incdec.ptr80 = getelementptr inbounds i8, ptr %scan.1, i64 5
  %26 = load i8, ptr %incdec.ptr80, align 1
  %incdec.ptr82 = getelementptr inbounds i8, ptr %match.0, i64 5
  %27 = load i8, ptr %incdec.ptr82, align 1
  %cmp84 = icmp eq i8 %26, %27
  br i1 %cmp84, label %land.lhs.true86, label %do.end

land.lhs.true86:                                  ; preds = %land.lhs.true79
  %incdec.ptr87 = getelementptr inbounds i8, ptr %scan.1, i64 6
  %28 = load i8, ptr %incdec.ptr87, align 1
  %incdec.ptr89 = getelementptr inbounds i8, ptr %match.0, i64 6
  %29 = load i8, ptr %incdec.ptr89, align 1
  %cmp91 = icmp eq i8 %28, %29
  br i1 %cmp91, label %land.lhs.true93, label %do.end

land.lhs.true93:                                  ; preds = %land.lhs.true86
  %incdec.ptr94 = getelementptr inbounds i8, ptr %scan.1, i64 7
  %30 = load i8, ptr %incdec.ptr94, align 1
  %incdec.ptr96 = getelementptr inbounds i8, ptr %match.0, i64 7
  %31 = load i8, ptr %incdec.ptr96, align 1
  %cmp98 = icmp eq i8 %30, %31
  br i1 %cmp98, label %land.lhs.true100, label %do.end

land.lhs.true100:                                 ; preds = %land.lhs.true93
  %incdec.ptr101 = getelementptr inbounds i8, ptr %scan.1, i64 8
  %32 = load i8, ptr %incdec.ptr101, align 1
  %incdec.ptr103 = getelementptr inbounds i8, ptr %match.0, i64 8
  %33 = load i8, ptr %incdec.ptr103, align 1
  %cmp105 = icmp eq i8 %32, %33
  %cmp107 = icmp ult ptr %incdec.ptr101, %add.ptr12
  %or.cond = and i1 %cmp105, %cmp107
  br i1 %or.cond, label %do.cond, label %do.end

do.end:                                           ; preds = %land.lhs.true100, %land.lhs.true93, %land.lhs.true86, %land.lhs.true79, %land.lhs.true72, %land.lhs.true65, %land.lhs.true, %do.cond
  %scan.2 = phi ptr [ %incdec.ptr101, %land.lhs.true100 ], [ %incdec.ptr94, %land.lhs.true93 ], [ %incdec.ptr87, %land.lhs.true86 ], [ %incdec.ptr80, %land.lhs.true79 ], [ %incdec.ptr73, %land.lhs.true72 ], [ %incdec.ptr66, %land.lhs.true65 ], [ %incdec.ptr59, %land.lhs.true ], [ %incdec.ptr53, %do.cond ]
  %sub.ptr.rhs.cast = ptrtoint ptr %scan.2 to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %conv109 = trunc i64 %sub.ptr.sub to i32
  %sub110 = sub nsw i32 258, %conv109
  %cmp112 = icmp sgt i32 %sub110, %best_len.0
  br i1 %cmp112, label %if.then114, label %do.cond125

if.then114:                                       ; preds = %do.end
  store i32 %cur_match.addr.0, ptr %match_start, align 4
  %cmp115 = icmp slt i32 %sub110, %.
  br i1 %cmp115, label %if.end118, label %do.end135

if.end118:                                        ; preds = %if.then114
  %sub119 = add nsw i32 %sub110, -1
  %idxprom120 = sext i32 %sub119 to i64
  %add.ptr111.sum = add i64 %idxprom120, %idx.ext
  %arrayidx121 = getelementptr inbounds i8, ptr %1, i64 %add.ptr111.sum
  %34 = load i8, ptr %arrayidx121, align 1
  %idxprom122 = sext i32 %sub110 to i64
  %add.ptr111.sum216 = add i64 %idxprom122, %idx.ext
  %arrayidx123 = getelementptr inbounds i8, ptr %1, i64 %add.ptr111.sum216
  %35 = load i8, ptr %arrayidx123, align 1
  br label %do.cond125

do.cond125:                                       ; preds = %if.end118, %do.end, %lor.lhs.false42, %lor.lhs.false37, %lor.lhs.false, %do.body
  %best_len.1 = phi i32 [ %best_len.0, %do.body ], [ %best_len.0, %lor.lhs.false ], [ %best_len.0, %lor.lhs.false37 ], [ %best_len.0, %lor.lhs.false42 ], [ %sub110, %if.end118 ], [ %best_len.0, %do.end ]
  %scan_end1.1 = phi i8 [ %scan_end1.0, %do.body ], [ %scan_end1.0, %lor.lhs.false ], [ %scan_end1.0, %lor.lhs.false37 ], [ %scan_end1.0, %lor.lhs.false42 ], [ %34, %if.end118 ], [ %scan_end1.0, %do.end ]
  %scan_end.1 = phi i8 [ %scan_end.0, %do.body ], [ %scan_end.0, %lor.lhs.false ], [ %scan_end.0, %lor.lhs.false37 ], [ %scan_end.0, %lor.lhs.false42 ], [ %35, %if.end118 ], [ %scan_end.0, %do.end ]
  %and = and i32 %cur_match.addr.0, %7
  %idxprom126 = zext i32 %and to i64
  %arrayidx127 = getelementptr inbounds i16, ptr %6, i64 %idxprom126
  %36 = load i16, ptr %arrayidx127, align 2
  %conv128 = zext i16 %36 to i32
  %cmp129 = icmp ugt i32 %conv128, %sub6.
  br i1 %cmp129, label %land.rhs131, label %do.end135

land.rhs131:                                      ; preds = %do.cond125
  %dec = add i32 %chain_length.1, -1
  %cmp132 = icmp eq i32 %dec, 0
  br i1 %cmp132, label %do.end135, label %do.body

do.end135:                                        ; preds = %land.rhs131, %do.cond125, %if.then114
  %best_len.2 = phi i32 [ %best_len.1, %land.rhs131 ], [ %best_len.1, %do.cond125 ], [ %sub110, %if.then114 ]
  %cmp137 = icmp ugt i32 %best_len.2, %11
  %.best_len.2 = select i1 %cmp137, i32 %11, i32 %best_len.2
  ret i32 %.best_len.2
}
