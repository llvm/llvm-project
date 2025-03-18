; RUN: opt -passes=loop-vectorize -force-vector-width=2 -S < %s
;
; Cleaned up version of fe_tools.all_dimensions.ll from PR36311.
; Forcing VF=2 to trigger vector code gen
;
; This is a test case that let's vectorizer's code gen to modify CFG and get
; DomTree out of date, such that an assert from SCEV would trigger if
; reanalysis of SCEV happens subsequently. Once vector code gen starts,
; vectorizer should not invoke recomputation of Analysis.

$test = comdat any

declare i32 @__gxx_personality_v0(...)

; Function Attrs: uwtable
define dso_local void @test(i1 %arg) local_unnamed_addr #0 comdat align 2 personality ptr @__gxx_personality_v0 {
entry:
  br label %for.body51

for.body51:                                       ; preds = %for.body51, %entry
  br i1 %arg, label %for.body51, label %for.body89.lr.ph

for.cond80.loopexit:                              ; preds = %for.body89
  %inc94.lcssa = phi i32 [ %inc94, %for.body89 ]
  br i1 %arg, label %for.body89.lr.ph, label %nrvo.skipdtor.loopexit

for.body89.lr.ph:                                 ; preds = %for.cond80.loopexit, %for.body51
  %i79.0179 = phi i32 [ %add90, %for.cond80.loopexit ], [ 0, %for.body51 ]
  %next_index.4178 = phi i32 [ %inc94.lcssa, %for.cond80.loopexit ], [ undef, %for.body51 ]
  %add90 = add nuw i32 %i79.0179, 1
  %mul91 = mul i32 %add90, undef
  br label %for.body89

for.body89:                                       ; preds = %for.body89, %for.body89.lr.ph
  %j.0175 = phi i32 [ 0, %for.body89.lr.ph ], [ %add92, %for.body89 ]
  %next_index.5174 = phi i32 [ %next_index.4178, %for.body89.lr.ph ], [ %inc94, %for.body89 ]
  %add92 = add nuw i32 %j.0175, 1
  %add93 = add i32 %add92, %mul91
  %inc94 = add i32 %next_index.5174, 1
  %conv95 = zext i32 %next_index.5174 to i64
  %arrayidx.i160 = getelementptr inbounds i32, ptr undef, i64 %conv95
  store i32 %add93, ptr %arrayidx.i160, align 4
;, !tbaa !1
  %cmp87 = icmp ult i32 %add92, undef
  br i1 %cmp87, label %for.body89, label %for.cond80.loopexit

nrvo.skipdtor.loopexit:                           ; preds = %for.cond80.loopexit
  ret void
}
