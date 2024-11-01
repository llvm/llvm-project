; RUN: opt < %s -passes=loop-simplify -S | FileCheck %s

; Make sure the preheader exists.
; CHECK: sw.bb103:
; CHECK: indirectbr {{.*}}label %while.cond112
; CHECK: while.cond112:
; But the tail is not split.
; CHECK: for.body:
; CHECK: indirectbr {{.*}}label %while.cond112
define fastcc void @build_regex_nfa() nounwind uwtable ssp {
entry:
  indirectbr i8* blockaddress(@build_regex_nfa, %while.cond), [label %while.cond]

while.cond:                                       ; preds = %if.then439, %entry
  indirectbr i8* blockaddress(@build_regex_nfa, %sw.bb103), [label %do.body785, label %sw.bb103]

sw.bb103:                                         ; preds = %while.body
  indirectbr i8* blockaddress(@build_regex_nfa, %while.cond112), [label %while.cond112]

while.cond112:                                    ; preds = %for.body, %for.cond.preheader, %sw.bb103
  %pc.0 = phi i8 [ -1, %sw.bb103 ], [ 0, %for.body ], [ %pc.0, %for.cond.preheader ]
  indirectbr i8* blockaddress(@build_regex_nfa, %Lsetdone), [label %sw.bb118, label %Lsetdone]

sw.bb118:                                         ; preds = %while.cond112
  indirectbr i8* blockaddress(@build_regex_nfa, %for.cond.preheader), [label %Lerror.loopexit, label %for.cond.preheader]

for.cond.preheader:                               ; preds = %sw.bb118
  indirectbr i8* blockaddress(@build_regex_nfa, %for.body), [label %while.cond112, label %for.body]

for.body:                                         ; preds = %for.body, %for.cond.preheader
  indirectbr i8* blockaddress(@build_regex_nfa, %for.body), [label %while.cond112, label %for.body]

Lsetdone:                                         ; preds = %while.cond112
  unreachable

do.body785:                                       ; preds = %while.cond, %while.body
  ret void

Lerror.loopexit:                                  ; preds = %sw.bb118
  unreachable
}
