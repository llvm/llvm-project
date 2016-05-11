; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -misched-postra=true

define i32 @test_cpsr() {
entry:
  %a = alloca [10 x i32], align 4
  %0 = bitcast [10 x i32]* %a to i8*
  %arrayidx.gep = getelementptr [10 x i32], [10 x i32]* %a, i32 0, i32 0
  br label %for.body

for.cond.cleanup:
  %c.1.reg2mem.0.lcssa = phi i32 [ %c.1.reg2mem.0, %for.inc ]
  ret i32 %c.1.reg2mem.0.lcssa

for.body:
  %1 = phi i32 [ 0, %entry ], [ %.pre, %for.inc.for.body_crit_edge ]
  %c.018.reg2mem.0 = phi i32 [ 0, %entry ], [ %c.1.reg2mem.0, %for.inc.for.body_crit_edge ]
  %b.017.reg2mem.0 = phi double [ 0.000000e+00, %entry ], [ %b.1.reg2mem.0, %for.inc.for.body_crit_edge ]
  %arrayidx.phi = phi i32* [ %arrayidx.gep, %entry ], [ %arrayidx.inc, %for.inc.for.body_crit_edge ]
  %i.019 = phi i32 [ 0, %entry ], [ %inc, %for.inc.for.body_crit_edge ]
  %cmp1 = icmp slt i32 %1, 10
  %arrayidx.inc = getelementptr i32, i32* %arrayidx.phi, i32 1
  br i1 %cmp1, label %for.inc, label %if.end

if.end:
  %conv = sitofp i32 %i.019 to double
  %cmp2 = fcmp nsz ogt double %conv, %b.017.reg2mem.0
  %selv = select i1 %cmp2, double %conv, double %b.017.reg2mem.0
  %selv7 = select i1 %cmp2, i32 %i.019, i32 %c.018.reg2mem.0
  br label %for.inc

for.inc:
  %b.1.reg2mem.0 = phi double [ %b.017.reg2mem.0, %for.body ], [ %selv, %if.end ]
  %c.1.reg2mem.0 = phi i32 [ %c.018.reg2mem.0, %for.body ], [ %selv7, %if.end ]
  %exitcond = icmp eq i32 %i.019, 9
  br i1 %exitcond, label %for.cond.cleanup, label %for.inc.for.body_crit_edge

for.inc.for.body_crit_edge:
  %inc = add nuw nsw i32 %i.019, 1
  %.pre = load i32, i32* %arrayidx.inc, align 4
  br label %for.body
}
