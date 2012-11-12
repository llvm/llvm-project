; RUN: llc -mtriple=x86_64-apple-darwin < %s | FileCheck %s

@.str2 = private unnamed_addr constant [7 x i8] c"memchr\00", align 1
@.str3 = private unnamed_addr constant [11 x i8] c"bsd_memchr\00", align 1
@str4 = private unnamed_addr constant [5 x i8] c"Bug!\00"

; Make sure at end of do.cond.i, we jump to do.body.i first to have a tighter
; inner loop.
define i32 @test_branches_order() uwtable ssp {
; CHECK: test_branches_order:
; CHECK: [[L0:LBB0_[0-9]+]]: ## %do.body.i
; CHECK: je
; CHECK: %do.cond.i
; CHECK: jne [[L0]]
; CHECK: jmp
; CHECK: %exit
entry:
  %strs = alloca [1000 x [1001 x i8]], align 16
  br label %for.cond

for.cond:
  %j.0 = phi i32 [ 0, %entry ], [ %inc10, %for.inc9 ]
  %cmp = icmp slt i32 %j.0, 1000
  br i1 %cmp, label %for.cond1, label %for.end11

for.cond1:
  %indvars.iv50 = phi i64 [ %indvars.iv.next51, %for.body3 ], [ 0, %for.cond ]
  %0 = trunc i64 %indvars.iv50 to i32
  %cmp2 = icmp slt i32 %0, 1000
  br i1 %cmp2, label %for.body3, label %for.inc9

for.body3:
  %arraydecay = getelementptr inbounds [1000 x [1001 x i8]]* %strs, i64 0, i64 %indvars.iv50, i64 0
  %call = call i8* @memchr(i8* %arraydecay, i32 120, i64 1000)
  %add.ptr = getelementptr inbounds [1000 x [1001 x i8]]* %strs, i64 0, i64 %indvars.iv50, i64 %indvars.iv50
  %cmp7 = icmp eq i8* %call, %add.ptr
  %indvars.iv.next51 = add i64 %indvars.iv50, 1
  br i1 %cmp7, label %for.cond1, label %if.then

if.then:
  %puts = call i32 @puts(i8* getelementptr inbounds ([5 x i8]* @str4, i64 0, i64 0))
  call void @exit(i32 1) noreturn
  unreachable

for.inc9:
  %inc10 = add nsw i32 %j.0, 1
  br label %for.cond

for.end11:
  %puts42 = call i32 @puts(i8* getelementptr inbounds ([7 x i8]* @.str2, i64 0, i64 0))
  br label %for.cond14

for.cond14:
  %j13.0 = phi i32 [ 0, %for.end11 ], [ %inc39, %for.inc38 ]
  %cmp15 = icmp slt i32 %j13.0, 1000
  br i1 %cmp15, label %for.cond18, label %for.end40

for.cond18:
  %indvars.iv = phi i64 [ %indvars.iv.next, %exit ], [ 0, %for.cond14 ]
  %1 = trunc i64 %indvars.iv to i32
  %cmp19 = icmp slt i32 %1, 1000
  br i1 %cmp19, label %for.body20, label %for.inc38

for.body20:
  %arraydecay24 = getelementptr inbounds [1000 x [1001 x i8]]* %strs, i64 0, i64 %indvars.iv, i64 0
  br label %do.body.i

do.body.i:
  %n.addr.0.i = phi i64 [ %dec.i, %do.cond.i ], [ 1000, %for.body20 ]
  %p.0.i = phi i8* [ %incdec.ptr.i, %do.cond.i ], [ %arraydecay24, %for.body20 ]
  %2 = load i8* %p.0.i, align 1
  %cmp3.i = icmp eq i8 %2, 120
  br i1 %cmp3.i, label %exit, label %do.cond.i

do.cond.i:
  %incdec.ptr.i = getelementptr inbounds i8* %p.0.i, i64 1
  %dec.i = add i64 %n.addr.0.i, -1
  %cmp5.i = icmp eq i64 %dec.i, 0
  br i1 %cmp5.i, label %if.then32, label %do.body.i

exit:
  %add.ptr30 = getelementptr inbounds [1000 x [1001 x i8]]* %strs, i64 0, i64 %indvars.iv, i64 %indvars.iv
  %cmp31 = icmp eq i8* %p.0.i, %add.ptr30
  %indvars.iv.next = add i64 %indvars.iv, 1
  br i1 %cmp31, label %for.cond18, label %if.then32

if.then32:
  %puts43 = call i32 @puts(i8* getelementptr inbounds ([5 x i8]* @str4, i64 0, i64 0))
  call void @exit(i32 1) noreturn
  unreachable

for.inc38:
  %inc39 = add nsw i32 %j13.0, 1
  br label %for.cond14

for.end40:
  %puts44 = call i32 @puts(i8* getelementptr inbounds ([11 x i8]* @.str3, i64 0, i64 0))
  ret i32 0
}

declare i8* @memchr(i8*, i32, i64) nounwind readonly
declare void @exit(i32) noreturn
declare i32 @puts(i8* nocapture) nounwind

