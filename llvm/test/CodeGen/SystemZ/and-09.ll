; RUN: llc < %s -mtriple=s390x-linux-gnu -o /dev/null -O0
;
; Check that tryRISBGZero() does not crash when LHS (and RHS) of an AND node
; is constant.

define void @fun() {
  %const = bitcast i64 1064831134304126 to i64
  %xor.i = xor i64 0, %const
  %sub.i = add nsw i64 0, -1064831134304126
  %xor1.i = xor i64 %sub.i, %const
  %and.i = and i64 %xor1.i, %xor.i
  %tobool5.not = icmp eq i64 %and.i, 0
  %spec.store.select = select i1 %tobool5.not, i64 %const, i64 6
  store i64 %spec.store.select, ptr undef
  ret void
}
