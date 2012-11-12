; RUN: llc < %s

define i32 @test(i32 %tmp158) {
entry:
        switch i32 %tmp158, label %bb336 [
	         i32 120, label %bb338
	         i32 121, label %bb338
                 i32 122, label %bb338
                 i32 123, label %bb338
                 i32 124, label %bb338
                 i32 125, label %bb338
                 i32 126, label %bb338
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
}

define i32 @test2(i32 %tmp158) {
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
