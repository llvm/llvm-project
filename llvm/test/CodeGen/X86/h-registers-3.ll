; RUN: llc < %s -march=x86    | grep mov | count 1
; RUN: llc < %s -march=x86-64 | grep mov | count 1

define zeroext i8 @foo() nounwind ssp {
entry:
  %0 = tail call zeroext i16 (...)* @bar() nounwind
  %1 = lshr i16 %0, 8
  %2 = trunc i16 %1 to i8
  ret i8 %2
}

declare zeroext i16 @bar(...)
