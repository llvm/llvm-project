; Test that the instructions have the correct whitespace.
; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck -strict-whitespace %s -check-prefix=16
; RUN: llc  -march=mips -mcpu=mips32r2 < %s | FileCheck %s -strict-whitespace -check-prefix=32R2

@main.L = internal unnamed_addr constant [5 x i8*] [i8* blockaddress(@main, %L1), i8* blockaddress(@main, %L2), i8* blockaddress(@main, %L3), i8* blockaddress(@main, %L4), i8* null], align 4
@str = private unnamed_addr constant [2 x i8] c"A\00"
@str5 = private unnamed_addr constant [2 x i8] c"B\00"
@str6 = private unnamed_addr constant [2 x i8] c"C\00"
@str7 = private unnamed_addr constant [2 x i8] c"D\00"
@str8 = private unnamed_addr constant [2 x i8] c"E\00"

define i32 @main() nounwind {
entry:
; 16: jalrc	${{[0-9]+}}
; 16: jrc	${{[0-9]+}}
; 16: jrc	$ra
  %puts = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str, i32 0, i32 0))
  br label %L1

L1:                                               ; preds = %entry, %L3
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %L3 ]
  %puts5 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str5, i32 0, i32 0))
  br label %L2

L2:                                               ; preds = %L1, %L3
  %i.1 = phi i32 [ %i.0, %L1 ], [ %inc, %L3 ]
  %puts6 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str6, i32 0, i32 0))
  br label %L3

L3:                                               ; preds = %L2, %L3
  %i.2 = phi i32 [ %i.1, %L2 ], [ %inc, %L3 ]
  %puts7 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str7, i32 0, i32 0))
  %inc = add i32 %i.2, 1
  %arrayidx = getelementptr inbounds [5 x i8*], [5 x i8*]* @main.L, i32 0, i32 %i.2
  %0 = load i8*, i8** %arrayidx, align 4
  indirectbr i8* %0, [label %L1, label %L2, label %L3, label %L4]
L4:                                               ; preds = %L3
  %puts8 = tail call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @str8, i32 0, i32 0))
  ret i32 0
}

declare i32 @puts(i8* nocapture) nounwind

define i32 @ext(i32 %s, i32 %pos, i32 %sz) nounwind readnone {
entry:
; 32R2: ext	${{[0-9]+}}, $4, 5, 9
  %shr = lshr i32 %s, 5
  %and = and i32 %shr, 511
  ret i32 %and
}

define void @ins(i32 %s, i32* nocapture %d) nounwind {
entry:
; 32R2: ins	${{[0-9]+}}, $4, 5, 9
  %and = shl i32 %s, 5
  %shl = and i32 %and, 16352
  %tmp3 = load i32, i32* %d, align 4
  %and5 = and i32 %tmp3, -16353
  %or = or i32 %and5, %shl
  store i32 %or, i32* %d, align 4
  ret void
}
