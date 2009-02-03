; RUN: llvm-as < %s | llc -march=x86-64 -o %t -f -stats -info-output-file - | \
; RUN:   grep {asm-printer} | grep {Number of machine instrs printed} | grep 5
; RUN: grep {leal	1(\%rsi),} %t

define fastcc zeroext i8 @fullGtU(i32 %i1, i32 %i2) nounwind {
entry:
  %0 = add i32 %i2, 1           ; <i32> [#uses=1]
  %1 = sext i32 %0 to i64               ; <i64> [#uses=1]
  %2 = getelementptr i8* null, i64 %1           ; <i8*> [#uses=1]
  %3 = load i8* %2, align 1             ; <i8> [#uses=1]
  %4 = icmp eq i8 0, %3         ; <i1> [#uses=1]
  br i1 %4, label %bb3, label %bb34

bb3:            ; preds = %entry
  %5 = add i32 %i2, 4           ; <i32> [#uses=0]
  ret i8 0

bb34:           ; preds = %entry
  ret i8 0
}

