; RUN: llc -march=hexagon -mcpu=hexagonv60 < %s
; REQUIRES: asserts

; Test that the pipeliner reuses an existing Phi when generating the epilog
; block. In this case, the original loops has a Phi whose operand is another
; Phi. When the loop is pipelined, the Phi that generates the operand value
; is used in two stages. This means the Phi for the second stage can
; be reused. The bug causes an assert due to an invalid virtual register error
; in the live variable analysis.

define void @test(ptr %a, ptr %b)  #0 {
entry:
  br label %for.body6.us.prol

for.body6.us.prol:
  %i.065.us.prol = phi i32 [ 0, %entry ], [ %inc.us.prol, %for.body6.us.prol ]
  %im1.064.us.prol = phi i32 [ undef, %entry ], [ %i.065.us.prol, %for.body6.us.prol ]
  %prol.iter = phi i32 [ undef, %entry ], [ %prol.iter.sub, %for.body6.us.prol ]
  %arrayidx8.us.prol = getelementptr inbounds i8, ptr %b, i32 %im1.064.us.prol
  %0 = load i8, ptr %arrayidx8.us.prol, align 1
  %conv9.us.prol = sext i8 %0 to i32
  %add.us.prol = add nsw i32 %conv9.us.prol, 0
  %add12.us.prol = add nsw i32 %add.us.prol, 0
  %mul.us.prol = mul nsw i32 %add12.us.prol, 3
  %conv13.us.prol = trunc i32 %mul.us.prol to i8
  %arrayidx14.us.prol = getelementptr inbounds i8, ptr %a, i32 %i.065.us.prol
  store i8 %conv13.us.prol, ptr %arrayidx14.us.prol, align 1
  %inc.us.prol = add nuw nsw i32 %i.065.us.prol, 1
  %prol.iter.sub = add i32 %prol.iter, -1
  %prol.iter.cmp = icmp eq i32 %prol.iter.sub, 0
  br i1 %prol.iter.cmp, label %for.body6.us, label %for.body6.us.prol

for.body6.us:
  %im2.063.us = phi i32 [ undef, %for.body6.us ], [ %im1.064.us.prol, %for.body6.us.prol ]
  %arrayidx10.us = getelementptr inbounds i8, ptr %b, i32 %im2.063.us
  %1 = load i8, ptr %arrayidx10.us, align 1
  %conv11.us = sext i8 %1 to i32
  %add12.us = add nsw i32 0, %conv11.us
  %mul.us = mul nsw i32 %add12.us, 3
  %conv13.us = trunc i32 %mul.us to i8
  store i8 %conv13.us, ptr undef, align 1
  br label %for.body6.us
}

