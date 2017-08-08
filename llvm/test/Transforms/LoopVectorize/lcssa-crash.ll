; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

%type1 = type { %type2 }
%type2 = type { [0 x i8*], i8**, i32, i32, i32 }

define void @test() nounwind uwtable align 2 {
  br label %for.body.lr.ph.i.i.i

for.body.lr.ph.i.i.i:
  br label %for.body.i.i.i

for.body.i.i.i:
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc.i.i.i ], [ 0, %for.body.lr.ph.i.i.i ]
  br label %for.inc.i.i.i

for.inc.i.i.i:
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, undef
  br i1 %exitcond, label %for.body.i.i.i, label %for.end.i.i.i

for.end.i.i.i:
  %lcssa = phi %type1* [ undef, %for.inc.i.i.i ]
  unreachable
}

; PR16139
define void @test2(i8* %x) {
entry:
  indirectbr i8* %x, [ label %L0, label %L1 ]

L0:
  br label %L0

L1:
  ret void
}

; This loop has different uniform instructions before and after LCSSA.
define void @test3() {
entry:
  %add41 = add i32 undef, undef
  %idxprom4736 = zext i32 %add41 to i64
  br label %while.body

while.body:
  %idxprom4738 = phi i64 [ %idxprom47, %while.body ], [ %idxprom4736, %entry ]
  %pos.337 = phi i32 [ %inc46, %while.body ], [ %add41, %entry ]
  %inc46 = add i32 %pos.337, 1
  %arrayidx48 = getelementptr inbounds [1024 x i8], [1024 x i8]* undef, i64 0, i64 %idxprom4738
  store i8 0, i8* %arrayidx48, align 1
  %and43 = and i32 %inc46, 3
  %cmp44 = icmp eq i32 %and43, 0
  %idxprom47 = zext i32 %inc46 to i64
  br i1 %cmp44, label %while.end, label %while.body

while.end:
  %add58 = add i32 %inc46, 4
  ret void
}
