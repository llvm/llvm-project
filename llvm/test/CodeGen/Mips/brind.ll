; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@main.L = internal unnamed_addr constant [5 x i8*] [i8* blockaddress(@main, %L1), i8* blockaddress(@main, %L2), i8* blockaddress(@main, %L3), i8* blockaddress(@main, %L4), i8* null], align 4
@str = private unnamed_addr constant [2 x i8] c"A\00"
@str5 = private unnamed_addr constant [2 x i8] c"B\00"
@str6 = private unnamed_addr constant [2 x i8] c"C\00"
@str7 = private unnamed_addr constant [2 x i8] c"D\00"
@str8 = private unnamed_addr constant [2 x i8] c"E\00"

define i32 @main() nounwind {
entry:
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8]* @str, i32 0, i32 0))
  br label %L1

L1:                                               ; preds = %entry, %L3
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L3 ]
  %puts5 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8]* @str5, i32 0, i32 0))
  br label %L2

L2:                                               ; preds = %L1, %L3
  %i.1 = phi i32 [ %i.0, %L1 ], [ %inc, %L3 ]
  %puts6 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8]* @str6, i32 0, i32 0))
  br label %L3

L3:                                               ; preds = %L2, %L3
  %i.2 = phi i32 [ %i.1, %L2 ], [ %inc, %L3 ]
  %puts7 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8]* @str7, i32 0, i32 0))
  %inc = add i32 %i.2, 1
  %arrayidx = getelementptr inbounds [5 x i8*]* @main.L, i32 0, i32 %i.2
  %0 = load i8** %arrayidx, align 4
  indirectbr i8* %0, [label %L1, label %L2, label %L3, label %L4]
; 16: 	jrc	 ${{[0-9]+}}
L4:                                               ; preds = %L3
  %puts8 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8]* @str8, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind


