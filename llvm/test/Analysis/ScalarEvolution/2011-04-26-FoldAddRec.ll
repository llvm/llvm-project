; RUN: opt < %s -analyze -iv-users
; PR9633: Tests that SCEV handles the mul.i2 recurrence being folded to
; constant zero.

define signext i8 @func_14(i8 signext %p_18) nounwind readnone ssp {
entry:
  br label %for.inc

for.inc:
  %p_17.addr.012 = phi i32 [ 0, %entry ], [ %add, %for.inc ]
  %add = add nsw i32 %p_17.addr.012, 1
  br i1 false, label %for.inc, label %for.cond

for.cond:
  %tobool.i = icmp ult i32 %add, 8192
  %shl.i = select i1 %tobool.i, i32 13, i32 0
  %shl.left.i = shl i32 %add, %shl.i
  %conv.i4 = trunc i32 %shl.left.i to i8
  br i1 undef, label %for.inc9, label %if.then

for.inc9:
  %p_18.addr.011 = phi i8 [ %add12, %for.inc9 ], [ %p_18, %for.cond ]
  %add12 = add i8 %p_18.addr.011, 1
  %mul.i2 = mul i8 %add12, %conv.i4
  %mul.i2.lobit = lshr i8 %mul.i2, 7
  %lor.ext.shr.i = select i1 undef, i8 %mul.i2.lobit, i8 %mul.i2
  %tobool = icmp eq i8 %lor.ext.shr.i, 0
  br i1 %tobool, label %for.inc9, label %if.then

if.then:
  ret i8 0

}