; RUN: llc -mtriple="x86_64-apple-darwin" -o - -verify-machineinstrs %s

define void @make_wanted() #0 {
entry:
  br i1 undef, label %for.end20, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:
  br label %for.body3

for.body3:
  %cmp20.i = icmp eq i32 undef, 0
  %.col.057 = select i1 %cmp20.i, i32 0, i32 undef
  br i1 undef, label %while.cond.i, label %for.body5.lr.ph.i

for.body5.lr.ph.i:
  %0 = sext i32 %.col.057 to i64
  %1 = sub i32 0, %.col.057
  %2 = zext i32 %1 to i64
  %3 = add nuw nsw i64 %2, 1
  %n.vec110 = and i64 %3, 8589934588
  %end.idx.rnd.down111 = add nsw i64 %n.vec110, %0
  br i1 undef, label %middle.block105, label %vector.ph103

vector.ph103:
  br i1 undef, label %middle.block105, label %vector.body104

vector.body104:
  %4 = icmp eq i64 undef, %end.idx.rnd.down111
  br i1 %4, label %middle.block105, label %vector.body104

middle.block105:
  %resume.val114 = phi i64 [ %0, %for.body5.lr.ph.i ], [ %end.idx.rnd.down111, %vector.body104 ], [ %end.idx.rnd.down111, %vector.ph103 ]
  %cmp.n116 = icmp eq i64 undef, %resume.val114
  br i1 %cmp.n116, label %while.cond.i, label %for.body5.i.preheader

for.body5.i.preheader:
  %lcmp.or182 = or i1 undef, undef
  br i1 %lcmp.or182, label %for.body5.i.prol, label %while.cond.i

for.body5.i.prol:
  br i1 undef, label %for.body5.i.prol, label %while.cond.i

while.cond.i:
  br i1 undef, label %while.cond.i, label %if.then

if.then:
  br label %for.body3

for.end20:
  ret void
}
