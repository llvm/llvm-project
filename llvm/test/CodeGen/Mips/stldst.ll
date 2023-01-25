; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@kkkk = global i32 67, align 4
@llll = global i32 33, align 4
@mmmm = global i32 44, align 4
@nnnn = global i32 55, align 4
@oooo = global i32 32, align 4
@pppp = global i32 41, align 4
@qqqq = global i32 59, align 4
@rrrr = global i32 60, align 4
@.str = private unnamed_addr constant [32 x i8] c"%i %i %i %i %i %i %i %i %i %i \0A\00", align 1

define i32 @main() nounwind {
entry:
  %0 = load i32, ptr @kkkk, align 4
  %1 = load i32, ptr @llll, align 4
  %add = add nsw i32 %0, 10
  %add1 = add nsw i32 %1, 10
  %2 = load i32, ptr @mmmm, align 4
  %sub = add nsw i32 %2, -3
  %3 = load i32, ptr @nnnn, align 4
  %add2 = add nsw i32 %3, 10
  %4 = load i32, ptr @oooo, align 4
  %add3 = add nsw i32 %4, 4
  %5 = load i32, ptr @pppp, align 4
  %sub4 = add nsw i32 %5, -5
  %6 = load i32, ptr @qqqq, align 4
  %sub5 = add nsw i32 %6, -10
  %7 = load i32, ptr @rrrr, align 4
  %add6 = add nsw i32 %7, 6

  %call = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %sub5, i32 %add6, i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7) nounwind
  %call7 = tail call i32 (ptr, ...) @printf(ptr @.str, i32 %0, i32 %1, i32 %add, i32 %add1, i32 %sub, i32 %add2, i32 %add3, i32 %sub4, i32 %sub5, i32 %add6) nounwind
  ret i32 0
}
; 16:	sw	${{[0-9]+}}, {{[0-9]+}}($sp)         # 4-byte Folded Spill
; 16:	lw	${{[0-9]+}}, {{[0-9]+}}($sp)         # 4-byte Folded Reload
; 16:	sw	${{[0-9]+}}, {{[0-9]+}}($sp)         # 4-byte Folded Spill
; 16:	lw	${{[0-9]+}}, {{[0-9]+}}($sp)         # 4-byte Folded Reload

declare i32 @printf(ptr nocapture, ...) nounwind
