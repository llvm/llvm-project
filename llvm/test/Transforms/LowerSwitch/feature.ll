; RUN: llvm-as %s -o - | opt -lowerswitch | llvm-dis > %t
; RUN: grep slt %t | count 10
; RUN: grep ule %t | count 3
; RUN: grep eq  %t | count 9

define i32 @main(i32 %tmp158) {
entry:
        switch i32 %tmp158, label %bb336 [
                 i32 -2, label %bb338
                 i32 -3, label %bb338
                 i32 -4, label %bb338
                 i32 -5, label %bb338
                 i32 -6, label %bb338
                 i32 0, label %bb338
                 i32 1, label %bb338
                 i32 2, label %bb338
                 i32 3, label %bb338
                 i32 4, label %bb338
                 i32 5, label %bb338
                 i32 6, label %bb338
                 i32 7, label %bb
                 i32 8, label %bb338
                 i32 9, label %bb322
                 i32 10, label %bb324
                 i32 11, label %bb326
                 i32 12, label %bb328
                 i32 13, label %bb330
                 i32 14, label %bb332
                 i32 15, label %bb334
        ]
bb:
  ret i32 2
bb322:
  ret i32 3
bb324:
  ret i32 4
bb326:
  ret i32 5
bb328:
  ret i32 6
bb330:
  ret i32 7
bb332:
  ret i32 8
bb334:
  ret i32 9
bb336:
  ret i32 10
bb338:
  ret i32 11
}
