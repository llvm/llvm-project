; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s

; This code contains three loops. One is triple-nested, the
; second is double nested and the third is a single loop. At
; runtime, all three loops execute 1,000,000 times each. We use to
; give different frequencies to each of the loops because loop
; scales were limited to no more than 4,096.
;
; This was penalizing the hotness of the second and third loops
; because BFI was reducing the loop scale for for.cond16 and
; for.cond26 to a max of 4,096.
;
; Without this restriction, all loops are now correctly given the same
; frequency values.
;
; Original C code:
;
;
; int g;
; __attribute__((noinline)) void bar() {
;  g++;
; }
;
; extern int printf(const char*, ...);
;
; int main()
; {
;   int i, j, k;
;
;   g = 0;
;   for (i = 0; i < 100; i++)
;     for (j = 0; j < 100; j++)
;        for (k = 0; k < 100; k++)
;            bar();
;
;   printf ("g = %d\n", g);
;   g = 0;
;
;   for (i = 0; i < 100; i++)
;     for (j = 0; j < 10000; j++)
;         bar();
;
;   printf ("g = %d\n", g);
;   g = 0;
;
;
;   for (i = 0; i < 1000000; i++)
;     bar();
;
;   printf ("g = %d\n", g);
;   g = 0;
; }

@g = common global i32 0, align 4
@.str = private unnamed_addr constant [8 x i8] c"g = %d\0A\00", align 1

declare void @bar()
declare i32 @printf(i8*, ...)

; CHECK: Printing analysis {{.*}} for function 'main':
; CHECK-NEXT: block-frequency-info: main
define i32 @main() !prof !6 {
entry:
  %retval = alloca i32, align 4
  %i = alloca i32, align 4
  %j = alloca i32, align 4
  %k = alloca i32, align 4
  store i32 0, i32* %retval
  store i32 0, i32* @g, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc10, %entry
  %0 = load i32, i32* %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end12, !prof !1

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %j, align 4
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc7, %for.body
  %1 = load i32, i32* %j, align 4
  %cmp2 = icmp slt i32 %1, 100
  br i1 %cmp2, label %for.body3, label %for.end9, !prof !2

for.body3:                                        ; preds = %for.cond1
  store i32 0, i32* %k, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %2 = load i32, i32* %k, align 4
  %cmp5 = icmp slt i32 %2, 100
  br i1 %cmp5, label %for.body6, label %for.end, !prof !3

; CHECK: - for.body6: float = 1000000.0,{{.*}}count = 1000000
for.body6:                                        ; preds = %for.cond4
  call void @bar()
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %3 = load i32, i32* %k, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, i32* %k, align 4
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc7

for.inc7:                                         ; preds = %for.end
  %4 = load i32, i32* %j, align 4
  %inc8 = add nsw i32 %4, 1
  store i32 %inc8, i32* %j, align 4
  br label %for.cond1

for.end9:                                         ; preds = %for.cond1
  br label %for.inc10

for.inc10:                                        ; preds = %for.end9
  %5 = load i32, i32* %i, align 4
  %inc11 = add nsw i32 %5, 1
  store i32 %inc11, i32* %i, align 4
  br label %for.cond

for.end12:                                        ; preds = %for.cond
  %6 = load i32, i32* @g, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0), i32 %6)
  store i32 0, i32* @g, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond13

for.cond13:                                       ; preds = %for.inc22, %for.end12
  %7 = load i32, i32* %i, align 4
  %cmp14 = icmp slt i32 %7, 100
  br i1 %cmp14, label %for.body15, label %for.end24, !prof !1

for.body15:                                       ; preds = %for.cond13
  store i32 0, i32* %j, align 4
  br label %for.cond16

for.cond16:                                       ; preds = %for.inc19, %for.body15
  %8 = load i32, i32* %j, align 4
  %cmp17 = icmp slt i32 %8, 10000
  br i1 %cmp17, label %for.body18, label %for.end21, !prof !4

; CHECK: - for.body18: float = 999999.5,{{.*}}count = 1000000
for.body18:                                       ; preds = %for.cond16
  call void @bar()
  br label %for.inc19

for.inc19:                                        ; preds = %for.body18
  %9 = load i32, i32* %j, align 4
  %inc20 = add nsw i32 %9, 1
  store i32 %inc20, i32* %j, align 4
  br label %for.cond16

for.end21:                                        ; preds = %for.cond16
  br label %for.inc22

for.inc22:                                        ; preds = %for.end21
  %10 = load i32, i32* %i, align 4
  %inc23 = add nsw i32 %10, 1
  store i32 %inc23, i32* %i, align 4
  br label %for.cond13

for.end24:                                        ; preds = %for.cond13
  %11 = load i32, i32* @g, align 4
  %call25 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0), i32 %11)
  store i32 0, i32* @g, align 4
  store i32 0, i32* %i, align 4
  br label %for.cond26

for.cond26:                                       ; preds = %for.inc29, %for.end24
  %12 = load i32, i32* %i, align 4
  %cmp27 = icmp slt i32 %12, 1000000
  br i1 %cmp27, label %for.body28, label %for.end31, !prof !5

; CHECK: - for.body28: float = 1000224.3,{{.*}}count = 1000224
for.body28:                                       ; preds = %for.cond26
  call void @bar()
  br label %for.inc29

for.inc29:                                        ; preds = %for.body28
  %13 = load i32, i32* %i, align 4
  %inc30 = add nsw i32 %13, 1
  store i32 %inc30, i32* %i, align 4
  br label %for.cond26

for.end31:                                        ; preds = %for.cond26
  %14 = load i32, i32* @g, align 4
  %call32 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str, i32 0, i32 0), i32 %14)
  store i32 0, i32* @g, align 4
  %15 = load i32, i32* %retval
  ret i32 %15
}

!llvm.ident = !{!0}

!0 = !{!"clang version 3.7.0 (trunk 232635) (llvm/trunk 232636)"}
!1 = !{!"branch_weights", i32 100, i32 1}
!2 = !{!"branch_weights", i32 10000, i32 100}
!3 = !{!"branch_weights", i32 1000000, i32 10000}
!4 = !{!"branch_weights", i32 1000000, i32 100}
!5 = !{!"branch_weights", i32 1000000, i32 1}
!6 = !{!"function_entry_count", i32 1}
