; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

  %i35 = type i35


define i35 @"squared"(%i35 %i0)
begin
	switch i35 %i0, label %Default [ 
		i35 1, label %Case1
		i35 2, label %Case2
		i35 4, label %Case4 ]

Default:
    ret i35 -1                      ; Unrecognized input value

Case1:
    ret i35 1
Case2:
    ret i35 4
Case4:
    ret i35 16
end
