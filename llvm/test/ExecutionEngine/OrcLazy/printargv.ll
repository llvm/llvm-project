; RUN: lli -jit-kind=orc-lazy %s a b c | FileCheck %s

; CHECK: argc = 4
; CHECK-NEXT: argv = ["{{.*}}printargv.ll", "a", "b", "c"]
; CHECK-NEXT; argv[4] = null

@.str = private unnamed_addr constant [11 x i8] c"argc = %i\0A\00", align 1
@.str.1 = private unnamed_addr constant [9 x i8] c"argv = [\00", align 1
@.str.3 = private unnamed_addr constant [5 x i8] c"\22%s\22\00", align 1
@.str.4 = private unnamed_addr constant [5 x i8] c"null\00", align 1
@.str.5 = private unnamed_addr constant [7 x i8] c", \22%s\22\00", align 1
@.str.6 = private unnamed_addr constant [15 x i8] c"argv[%i] = %s\0A\00", align 1
@.str.7 = private unnamed_addr constant [5 x i8] c"junk\00", align 1
@str.8 = private unnamed_addr constant [2 x i8] c"]\00", align 1

define i32 @main(i32 %argc, i8** nocapture readonly %argv)  {
entry:
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @.str, i64 0, i64 0), i32 %argc)
  %call1 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.1, i64 0, i64 0))
  %cmp = icmp eq i32 %argc, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %puts36 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str.8, i64 0, i64 0))
  br label %if.end

if.end:
  %0 = load i8*, i8** %argv, align 8
  %tobool = icmp eq i8* %0, null
  br i1 %tobool, label %if.else, label %if.then3

if.then3:
  %call5 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.3, i64 0, i64 0), i8* %0)
  br label %if.end7

if.else:
  %call6 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i64 0, i64 0))
  br label %if.end7

if.end7:
  %cmp837 = icmp eq i32 %argc, 1
  br i1 %cmp837, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:
  %1 = zext i32 %argc to i64
  br label %for.body

for.cond.cleanup:
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str.8, i64 0, i64 0))
  %idxprom19 = sext i32 %argc to i64
  %arrayidx20 = getelementptr inbounds i8*, i8** %argv, i64 %idxprom19
  %2 = load i8*, i8** %arrayidx20, align 8
  %tobool21 = icmp eq i8* %2, null
  %cond = select i1 %tobool21, i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i64 0, i64 0), i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.7, i64 0, i64 0)
  %call22 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.6, i64 0, i64 0), i32 %argc, i8* %cond)
  ret i32 0

for.body:
  %indvars.iv = phi i64 [ 1, %for.body.preheader ], [ %indvars.iv.next, %for.inc ]
  %arrayidx9 = getelementptr inbounds i8*, i8** %argv, i64 %indvars.iv
  %3 = load i8*, i8** %arrayidx9, align 8
  %tobool10 = icmp eq i8* %3, null
  br i1 %tobool10, label %if.else15, label %if.then11

if.then11:
  %call14 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.5, i64 0, i64 0), i8* %3)
  br label %for.inc

if.else15:
  %call16 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @.str.4, i64 0, i64 0))
  br label %for.inc

for.inc:
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %cmp8 = icmp eq i64 %indvars.iv.next, %1
  br i1 %cmp8, label %for.cond.cleanup, label %for.body
}

declare i32 @printf(i8* nocapture readonly, ...)

declare i32 @puts(i8* nocapture readonly)
