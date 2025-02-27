; RUN: llc  -mtriple=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@main.L = internal unnamed_addr constant [5 x ptr] [ptr blockaddress(@main, %L1), ptr blockaddress(@main, %L2), ptr blockaddress(@main, %L3), ptr blockaddress(@main, %L4), ptr null], align 4
@str = private unnamed_addr constant [2 x i8] c"A\00"
@str5 = private unnamed_addr constant [2 x i8] c"B\00"
@str6 = private unnamed_addr constant [2 x i8] c"C\00"
@str7 = private unnamed_addr constant [2 x i8] c"D\00"
@str8 = private unnamed_addr constant [2 x i8] c"E\00"

define i32 @main() nounwind {
entry:
  %puts = tail call i32 @puts(ptr @str)
  br label %L1

L1:                                               ; preds = %entry, %L3
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L3 ]
  %puts5 = tail call i32 @puts(ptr @str5)
  br label %L2

L2:                                               ; preds = %L1, %L3
  %i.1 = phi i32 [ %i.0, %L1 ], [ %inc, %L3 ]
  %puts6 = tail call i32 @puts(ptr @str6)
  br label %L3

L3:                                               ; preds = %L2, %L3
  %i.2 = phi i32 [ %i.1, %L2 ], [ %inc, %L3 ]
  %puts7 = tail call i32 @puts(ptr @str7)
  %inc = add i32 %i.2, 1
  %arrayidx = getelementptr inbounds [5 x ptr], ptr @main.L, i32 0, i32 %i.2
  %0 = load ptr, ptr %arrayidx, align 4
  indirectbr ptr %0, [label %L1, label %L2, label %L3, label %L4]
; 16: 	jrc	 ${{[0-9]+}}
L4:                                               ; preds = %L3
  %puts8 = tail call i32 @puts(ptr @str8)
  ret i32 0
}

declare i32 @puts(ptr nocapture) nounwind


