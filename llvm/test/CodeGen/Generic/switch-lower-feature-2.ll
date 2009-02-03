; RUN: llvm-as < %s | llc -march=x86 -o %t -f
; RUN: grep jb %t | count 1
; RUN: grep \\\$6 %t | count 2
; RUN: grep 1024 %t | count 1
; RUN: grep 1023 %t | count 1
; RUN: grep 119  %t | count 1
; RUN: grep JTI %t | count 2
; RUN: grep jg %t | count 1
; RUN: grep ja %t | count 1
; RUN: grep js %t | count 1

target triple = "i686-pc-linux-gnu"

define i32 @main(i32 %tmp158) {
entry:
        switch i32 %tmp158, label %bb336 [
	         i32 -2147483648, label %bb338
		 i32 -2147483647, label %bb338
		 i32 -2147483646, label %bb338
	         i32 120, label %bb338
	         i32 121, label %bb339
                 i32 122, label %bb340
                 i32 123, label %bb341
                 i32 124, label %bb342
                 i32 125, label %bb343
                 i32 126, label %bb336
		 i32 1024, label %bb338
                 i32 0, label %bb338
                 i32 1, label %bb338
                 i32 2, label %bb338
                 i32 3, label %bb338
                 i32 4, label %bb338
		 i32 5, label %bb338
        ]
bb336:
  ret i32 10
bb338:
  ret i32 11
bb339:
  ret i32 12
bb340:
  ret i32 13
bb341:
  ret i32 14
bb342:
  ret i32 15
bb343:
  ret i32 18

}
