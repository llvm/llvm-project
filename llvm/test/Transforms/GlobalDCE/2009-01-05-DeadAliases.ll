; RUN: llvm-as < %s | opt -globaldce | llvm-dis | not grep @D
; RUN: llvm-as < %s | opt -globaldce | llvm-dis | grep @L | count 3

@A = global i32 0
@D = alias internal i32* @A
@L1 = alias i32* @A
@L2 = alias internal i32* @L1
@L3 = alias i32* @L2
