; RUN: llc -mtriple=x86_64 < %s -o /dev/null
define void @snork(i64 %arg) {
bbl:
  %shl = shl nsw <2 x i64> zeroinitializer, zeroinitializer
  %extractelement = extractelement <2 x i64> %shl, i64 0
  %sub = sub i64 %arg, %extractelement
  %and = and i64 %arg, 1
  %sub1 = sub i64 %sub, %and
  %add = add i64 %extractelement, %sub1
  br label %bbl2

bbl2:
  %phi = phi i64 [ %add3, %bbl2 ], [ %add, %bbl ]
  %add3 = add i64 %phi, 1
  %icmp = icmp eq i64 %add3, 0
  br label %bbl2
}
