; RUN: llc < %s -disable-fp-elim -verify-coalescing
; PR13414
;
; During coalescing, remat triggers DCE which deletes the penultimate use of a
; load. This load should not be folded into the remaining use because it is not
; safe to move, and it would extend the live range of the address.
;
; LiveRangeEdit::foldAsLoad() doesn't extend live ranges, so -verify-coalescing
; catches the problem.

target triple = "i386-unknown-linux-gnu"

%type_a = type { %type_a*, %type_b }
%type_b = type { %type_c, i32 }
%type_c = type { i32, %type_d }
%type_d = type { i64 }
%type_e = type { %type_c, i64 }

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind

define linkonce_odr void @test() nounwind {
entry:
  br i1 undef, label %while.end.while.end26_crit_edge, label %while.body12.lr.ph

while.end.while.end26_crit_edge:                  ; preds = %entry
  br label %while.end26

while.body12.lr.ph:                               ; preds = %entry
  br label %while.body12

while.body12:                                     ; preds = %if.end24, %while.body12.lr.ph
  %tmp = phi %type_a* [ undef, %while.body12.lr.ph ], [ %tmp18, %if.end24 ]
  %ins151154161 = phi i128 [ 0, %while.body12.lr.ph ], [ %phitmp, %if.end24 ]
  %ins135156160 = phi i128 [ 0, %while.body12.lr.ph ], [ %phitmp158, %if.end24 ]
  %ins151 = or i128 0, %ins151154161
  %cmp.i.i.i.i.i67 = icmp sgt i32 undef, 8
  br i1 %cmp.i.i.i.i.i67, label %if.then.i.i.i.i71, label %if.else.i.i.i.i74

if.then.i.i.i.i71:                                ; preds = %while.body12
  %call4.i.i.i.i68 = call noalias i8* @malloc(i32 undef) nounwind
  %tmp1 = getelementptr inbounds %type_a, %type_a* %tmp, i32 0, i32 1, i32 0, i32 1
  %buf_6.i.i.i.i70 = bitcast %type_d* %tmp1 to i8**
  %tmp2 = load i8*, i8** %buf_6.i.i.i.i70, align 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* undef, i8* %tmp2, i32 undef, i1 false) nounwind
  unreachable

if.else.i.i.i.i74:                                ; preds = %while.body12
  %i_.i.i.i.i72 = getelementptr inbounds %type_a, %type_a* %tmp, i32 0, i32 1, i32 0, i32 1, i32 0
  %tmp3 = load i64, i64* %i_.i.i.i.i72, align 4
  %tmp4 = zext i64 %tmp3 to i128
  %tmp5 = shl nuw nsw i128 %tmp4, 32
  %ins148 = or i128 %tmp5, %ins151
  %second3.i.i76 = getelementptr inbounds %type_a, %type_a* %tmp, i32 0, i32 1, i32 1
  %tmp6 = load i32, i32* %second3.i.i76, align 4
  %tmp7 = zext i32 %tmp6 to i128
  %tmp8 = shl nuw i128 %tmp7, 96
  %mask144 = and i128 %ins148, 79228162495817593519834398720
  %tmp9 = load %type_e*, %type_e** undef, align 4
  %len_.i.i.i.i86 = getelementptr inbounds %type_e, %type_e* %tmp9, i32 0, i32 0, i32 0
  %tmp10 = load i32, i32* %len_.i.i.i.i86, align 4
  %tmp11 = zext i32 %tmp10 to i128
  %ins135 = or i128 %tmp11, %ins135156160
  %cmp.i.i.i.i.i88 = icmp sgt i32 %tmp10, 8
  br i1 %cmp.i.i.i.i.i88, label %if.then.i.i.i.i92, label %if.else.i.i.i.i95

if.then.i.i.i.i92:                                ; preds = %if.else.i.i.i.i74
  %call4.i.i.i.i89 = call noalias i8* @malloc(i32 %tmp10) nounwind
  %ins126 = or i128 0, %ins135
  %tmp12 = getelementptr inbounds %type_e, %type_e* %tmp9, i32 0, i32 0, i32 1
  %buf_6.i.i.i.i91 = bitcast %type_d* %tmp12 to i8**
  %tmp13 = load i8*, i8** %buf_6.i.i.i.i91, align 4
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %call4.i.i.i.i89, i8* %tmp13, i32 %tmp10, i1 false) nounwind
  br label %A

if.else.i.i.i.i95:                                ; preds = %if.else.i.i.i.i74
  %i_.i.i.i.i93 = getelementptr inbounds %type_e, %type_e* %tmp9, i32 0, i32 0, i32 1, i32 0
  br label %A

A:                                                ; preds = %if.else.i.i.i.i95, %if.then.i.i.i.i92
  %ins135157 = phi i128 [ %ins126, %if.then.i.i.i.i92 ], [ undef, %if.else.i.i.i.i95 ]
  %second3.i.i97 = getelementptr inbounds %type_e, %type_e* %tmp9, i32 0, i32 1
  %tmp14 = load i64, i64* %second3.i.i97, align 4
  %tmp15 = trunc i64 %tmp14 to i32
  %cmp.i99 = icmp sgt i32 %tmp6, %tmp15
  %tmp16 = trunc i128 %ins135157 to i32
  %cmp.i.i.i.i.i.i101 = icmp sgt i32 %tmp16, 8
  br i1 %cmp.i.i.i.i.i.i101, label %if.then.i.i.i.i.i103, label %B

if.then.i.i.i.i.i103:                             ; preds = %A
  unreachable

B:                                                ; preds = %A
  %tmp17 = trunc i128 %ins148 to i32
  %cmp.i.i.i.i.i.i83 = icmp sgt i32 %tmp17, 8
  br i1 %cmp.i.i.i.i.i.i83, label %if.then.i.i.i.i.i85, label %C

if.then.i.i.i.i.i85:                              ; preds = %B
  unreachable

C:                                                ; preds = %B
  br i1 %cmp.i99, label %if.then17, label %if.end24

if.then17:                                        ; preds = %C
  br i1 false, label %if.then.i.i.i.i.i43, label %D

if.then.i.i.i.i.i43:                              ; preds = %if.then17
  unreachable

D:                                                ; preds = %if.then17
  br i1 undef, label %if.then.i.i.i.i.i, label %E

if.then.i.i.i.i.i:                                ; preds = %D
  unreachable

E:                                                ; preds = %D
  br label %if.end24

if.end24:                                         ; preds = %E, %C
  %phitmp = or i128 %tmp8, %mask144
  %phitmp158 = or i128 undef, undef
  %tmp18 = load %type_a*, %type_a** undef, align 4
  %tmp19 = load %type_a*, %type_a** undef, align 4
  %cmp.i49 = icmp eq %type_a* %tmp18, %tmp19
  br i1 %cmp.i49, label %while.cond10.while.end26_crit_edge, label %while.body12

while.cond10.while.end26_crit_edge:               ; preds = %if.end24
  %.pre = load %type_e*, %type_e** undef, align 4
  br label %while.end26

while.end26:                                      ; preds = %while.cond10.while.end26_crit_edge, %while.end.while.end26_crit_edge
  br i1 undef, label %while.body.lr.ph.i, label %F

while.body.lr.ph.i:                               ; preds = %while.end26
  br label %while.body.i

while.body.i:                                     ; preds = %while.body.i, %while.body.lr.ph.i
  br i1 false, label %while.body.i, label %F

F:                                                ; preds = %while.body.i, %while.end26
  ret void
}

declare noalias i8* @malloc(i32) nounwind
