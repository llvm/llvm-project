; RUN: llvm-as < %s | llc -march=x86    | grep and | count 1
; RUN: llvm-as < %s | llc -march=x86-64 | not grep and 

define i32 @t1(i32 %t, i32 %val) nounwind {
       %shamt = and i32 %t, 31
       %res = shl i32 %val, %shamt
       ret i32 %res
}

@X = internal global i16 0

define void @t2(i16 %t) nounwind {
       %shamt = and i16 %t, 31
       %tmp = load i16* @X
       %tmp1 = ashr i16 %tmp, %shamt
       store i16 %tmp1, i16* @X
       ret void
}

define i64 @t3(i64 %t, i64 %val) nounwind {
       %shamt = and i64 %t, 63
       %res = lshr i64 %val, %shamt
       ret i64 %res
}
