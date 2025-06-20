; RUN: llc -mtriple=msp430 < %s
; PR5558

define i64 @_strtoll_r(i16 %base) nounwind {
entry:
  br i1 undef, label %if.then, label %if.end27

if.then:                                          ; preds = %do.end
  br label %if.end27

if.end27:                                         ; preds = %if.then, %do.end
  %cond66 = select i1 undef, i64 -9223372036854775808, i64 9223372036854775807 ; <i64> [#uses=3]
  %conv69 = sext i16 %base to i64                 ; <i64> [#uses=1]
  %div = udiv i64 %cond66, %conv69                ; <i64> [#uses=1]
  br label %for.cond

for.cond:                                         ; preds = %if.end116, %if.end27
  br i1 undef, label %if.then152, label %if.then93

if.then93:                                        ; preds = %for.cond
  br i1 undef, label %if.end116, label %if.then152

if.end116:                                        ; preds = %if.then93
  %cmp123 = icmp ugt i64 undef, %div              ; <i1> [#uses=1]
  %or.cond = or i1 undef, %cmp123                 ; <i1> [#uses=0]
  br label %for.cond

if.then152:                                       ; preds = %if.then93, %for.cond
  br i1 undef, label %if.end182, label %if.then172

if.then172:                                       ; preds = %if.then152
  ret i64 %cond66

if.end182:                                        ; preds = %if.then152
  ret i64 %cond66
}
