; Test that the instructions have the correct whitespace.
; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic < %s | FileCheck -strict-whitespace %s -check-prefix=16
; RUN: llc  -march=mips -mcpu=mips32r2 < %s | FileCheck %s -strict-whitespace -check-prefix=32R2

@main.L = internal unnamed_addr constant [5 x ptr] [ptr blockaddress(@main, %L1), ptr blockaddress(@main, %L2), ptr blockaddress(@main, %L3), ptr blockaddress(@main, %L4), ptr null], align 4
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
L4:                                               ; preds = %L3
  %puts8 = tail call i32 @puts(ptr @str8)
  ret i32 0
}

declare i32 @puts(ptr nocapture) nounwind

define i32 @ext(i32 %s, i32 %pos, i32 %sz) nounwind readnone {
entry:
; 32R2: ext	${{[0-9]+}}, $4, 5, 9
  %shr = lshr i32 %s, 5
  %and = and i32 %shr, 511
  ret i32 %and
}

define void @ins(i32 %s, ptr nocapture %d) nounwind {
entry:
; 32R2: ins	${{[0-9]+}}, $4, 5, 9
  %and = shl i32 %s, 5
  %shl = and i32 %and, 16352
  %tmp3 = load i32, ptr %d, align 4
  %and5 = and i32 %tmp3, -16353
  %or = or i32 %and5, %shl
  store i32 %or, ptr %d, align 4
  ret void
}
