; RUN: llc -mtriple=aarch64-linux-gnu -o - %s

; Regression test for a crash in the ShrinkWrap pass not handling targets
; requiring a register scavenger.

%type1 = type { i32, i32, i32 }

@g1 = external unnamed_addr global i32, align 4
@g2 = external unnamed_addr global i1
@g3 = external unnamed_addr global [144 x i32], align 4
@g4 = external unnamed_addr constant [144 x i32], align 4
@g5 = external unnamed_addr constant [144 x i32], align 4
@g6 = external unnamed_addr constant [144 x i32], align 4
@g7 = external unnamed_addr constant [144 x i32], align 4
@g8 = external unnamed_addr constant [144 x i32], align 4
@g9 = external unnamed_addr constant [144 x i32], align 4
@g10 = external unnamed_addr constant [144 x i32], align 4
@g11 = external unnamed_addr global i32, align 4
@g12 = external unnamed_addr global [144 x [144 x i8]], align 1
@g13 = external unnamed_addr global ptr, align 8
@g14 = external unnamed_addr global [144 x [144 x i8]], align 1
@g15 = external unnamed_addr global [144 x [144 x i8]], align 1
@g16 = external unnamed_addr global [144 x [144 x i8]], align 1
@g17 = external unnamed_addr global [62 x i32], align 4
@g18 = external unnamed_addr global i32, align 4
@g19 = external unnamed_addr constant [144 x i32], align 4
@g20 = external unnamed_addr global [144 x [144 x i8]], align 1
@g21 = external unnamed_addr global i32, align 4

declare fastcc i32 @foo()

declare fastcc i32 @bar()

define internal fastcc i32 @func(i32 %alpha, i32 %beta) {
entry:
  %v1 = alloca [2 x [11 x i32]], align 4
  %v2 = alloca [11 x i32], align 16
  %v3 = alloca [11 x i32], align 16
  switch i32 undef, label %if.end.9 [
    i32 4, label %if.then.6
    i32 3, label %if.then.2
  ]

if.then.2:
  %call3 = tail call fastcc i32 @bar()
  br label %cleanup

if.then.6:
  %call7 = tail call fastcc i32 @foo()
  unreachable

if.end.9:
  %tmp = load i32, ptr @g1, align 4
  %rem.i = urem i32 %tmp, 1000000
  %idxprom.1.i = zext i32 %rem.i to i64
  %tmp1 = load ptr, ptr @g13, align 8
  %v4 = getelementptr inbounds %type1, ptr %tmp1, i64 %idxprom.1.i, i32 0
  %.b = load i1, ptr @g2, align 1
  %v5 = select i1 %.b, i32 2, i32 0
  %tmp2 = load i32, ptr @g18, align 4
  %tmp3 = load i32, ptr @g11, align 4
  %idxprom58 = sext i32 %tmp3 to i64
  %tmp4 = load i32, ptr @g21, align 4
  %idxprom69 = sext i32 %tmp4 to i64
  br label %for.body

for.body:
  %v6 = phi i32 [ 0, %if.end.9 ], [ %v7, %for.inc ]
  %a.0983 = phi i32 [ 1, %if.end.9 ], [ %a.1, %for.inc ]
  %arrayidx = getelementptr inbounds [62 x i32], ptr @g17, i64 0, i64 undef
  %tmp5 = load i32, ptr %arrayidx, align 4
  br i1 undef, label %for.inc, label %if.else.51

if.else.51:
  %idxprom53 = sext i32 %tmp5 to i64
  %arrayidx54 = getelementptr inbounds [144 x i32], ptr @g3, i64 0, i64 %idxprom53
  %tmp6 = load i32, ptr %arrayidx54, align 4
  switch i32 %tmp6, label %for.inc [
    i32 1, label %block.bb
    i32 10, label %block.bb.159
    i32 7, label %block.bb.75
    i32 8, label %block.bb.87
    i32 9, label %block.bb.147
    i32 12, label %block.bb.111
    i32 3, label %block.bb.123
    i32 4, label %block.bb.135
  ]

block.bb:
  %arrayidx56 = getelementptr inbounds [144 x i32], ptr @g6, i64 0, i64 %idxprom53
  %tmp7 = load i32, ptr %arrayidx56, align 4
  %shr = ashr i32 %tmp7, %v5
  %add57 = add nsw i32 %shr, 0
  %arrayidx61 = getelementptr inbounds [144 x [144 x i8]], ptr @g14, i64 0, i64 %idxprom53, i64 %idxprom58
  %tmp8 = load i8, ptr %arrayidx61, align 1
  %conv = zext i8 %tmp8 to i32
  %add62 = add nsw i32 %conv, %add57
  br label %for.inc

block.bb.75:
  %arrayidx78 = getelementptr inbounds [144 x i32], ptr @g10, i64 0, i64 %idxprom53
  %tmp9 = load i32, ptr %arrayidx78, align 4
  %shr79 = ashr i32 %tmp9, %v5
  %add80 = add nsw i32 %shr79, 0
  %add86 = add nsw i32 0, %add80
  br label %for.inc

block.bb.87:
  %arrayidx90 = getelementptr inbounds [144 x i32], ptr @g9, i64 0, i64 %idxprom53
  %tmp10 = load i32, ptr %arrayidx90, align 4
  %shr91 = ashr i32 %tmp10, 0
  %sub92 = sub nsw i32 0, %shr91
  %arrayidx96 = getelementptr inbounds [144 x [144 x i8]], ptr @g15, i64 0, i64 %idxprom53, i64 %idxprom69
  %tmp11 = load i8, ptr %arrayidx96, align 1
  %conv97 = zext i8 %tmp11 to i32
  %sub98 = sub nsw i32 %sub92, %conv97
  br label %for.inc

block.bb.111:
  %arrayidx114 = getelementptr inbounds [144 x i32], ptr @g19, i64 0, i64 %idxprom53
  %tmp12 = load i32, ptr %arrayidx114, align 4
  %shr115 = ashr i32 %tmp12, 0
  %sub116 = sub nsw i32 0, %shr115
  %arrayidx120 = getelementptr inbounds [144 x [144 x i8]], ptr @g12, i64 0, i64 %idxprom53, i64 %idxprom69
  %tmp13 = load i8, ptr %arrayidx120, align 1
  %conv121 = zext i8 %tmp13 to i32
  %sub122 = sub nsw i32 %sub116, %conv121
  br label %for.inc

block.bb.123:
  %arrayidx126 = getelementptr inbounds [144 x i32], ptr @g5, i64 0, i64 %idxprom53
  %tmp14 = load i32, ptr %arrayidx126, align 4
  %shr127 = ashr i32 %tmp14, %v5
  %add128 = add nsw i32 %shr127, 0
  %add134 = add nsw i32 0, %add128
  br label %for.inc

block.bb.135:
  %arrayidx138 = getelementptr inbounds [144 x i32], ptr @g4, i64 0, i64 %idxprom53
  %tmp15 = load i32, ptr %arrayidx138, align 4
  %shr139 = ashr i32 %tmp15, 0
  %sub140 = sub nsw i32 0, %shr139
  %arrayidx144 = getelementptr inbounds [144 x [144 x i8]], ptr @g20, i64 0, i64 %idxprom53, i64 %idxprom69
  %tmp16 = load i8, ptr %arrayidx144, align 1
  %conv145 = zext i8 %tmp16 to i32
  %sub146 = sub nsw i32 %sub140, %conv145
  br label %for.inc

block.bb.147:
  %arrayidx150 = getelementptr inbounds [144 x i32], ptr @g8, i64 0, i64 %idxprom53
  %tmp17 = load i32, ptr %arrayidx150, align 4
  %shr151 = ashr i32 %tmp17, %v5
  %add152 = add nsw i32 %shr151, 0
  %arrayidx156 = getelementptr inbounds [144 x [144 x i8]], ptr @g16, i64 0, i64 %idxprom53, i64 %idxprom58
  %tmp18 = load i8, ptr %arrayidx156, align 1
  %conv157 = zext i8 %tmp18 to i32
  %add158 = add nsw i32 %conv157, %add152
  br label %for.inc

block.bb.159:
  %sub160 = add nsw i32 %v6, -450
  %arrayidx162 = getelementptr inbounds [144 x i32], ptr @g7, i64 0, i64 %idxprom53
  %tmp19 = load i32, ptr %arrayidx162, align 4
  %shr163 = ashr i32 %tmp19, 0
  %sub164 = sub nsw i32 %sub160, %shr163
  %sub170 = sub nsw i32 %sub164, 0
  br label %for.inc

for.inc:
  %v7 = phi i32 [ %v6, %for.body ], [ %v6, %if.else.51 ], [ %sub170, %block.bb.159 ], [ %add158, %block.bb.147 ], [ %sub146, %block.bb.135 ], [ %add134, %block.bb.123 ], [ %sub122, %block.bb.111 ], [ %sub98, %block.bb.87 ], [ %add86, %block.bb.75 ], [ %add62, %block.bb ]
  %a.1 = phi i32 [ %a.0983, %for.body ], [ undef, %if.else.51 ], [ undef, %block.bb.159 ], [ undef, %block.bb.147 ], [ undef, %block.bb.135 ], [ undef, %block.bb.123 ], [ undef, %block.bb.111 ], [ undef, %block.bb.87 ], [ undef, %block.bb.75 ], [ undef, %block.bb ]
  %cmp48 = icmp sgt i32 %a.1, %tmp2
  br i1 %cmp48, label %for.end, label %for.body

for.end:
  store i32 %tmp, ptr %v4, align 4
  %hold_hash.i.7 = getelementptr inbounds %type1, ptr %tmp1, i64 %idxprom.1.i, i32 1
  store i32 0, ptr %hold_hash.i.7, align 4
  br label %cleanup

cleanup:
  %retval.0 = phi i32 [ %call3, %if.then.2 ], [ undef, %for.end ]
  ret i32 %retval.0
}
