; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

declare void @"foo"(i31 %i, i33 %j)


; foo test basic bitwise operations
define void @"foo"(i31 %i, i33 %j)
begin
	%t1 = trunc i33 %j to i31 
        %t2 = and i31 %t1, %i
        %t3 = sext i31 %i to i33
        %t4 = or i33 %t3, %j 
        %t5 = xor i31 %t2, 7 
        %t6 = shl i31 %i, 2
        %t7 = trunc i31 %i to i8
        %t8 = shl i8 %t7, 3
        %t9 = lshr i33 %j, 31
        %t7z = zext i8 %t7 to i33
        %t10 = ashr i33 %j, %t7z
	ret void
end

