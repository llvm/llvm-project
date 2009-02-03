; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


define i31 @"simpleIcmp"(i31 %i0, i31 %j0)
begin
	%t1 = icmp eq i31 %i0, %j0
	%t2 = icmp ne i31 %i0, %j0
	%t3 = icmp ult i31 %i0, %j0
        %t4 = icmp sgt i31 %i0, %j0
	%t5 = icmp ule i31 %i0, %j0
        %t6 = icmp sge i31 %i0, %j0

	%t7 = icmp eq i31 %i0, 1098765432
        %t8 = icmp ne i31 %i0, -31415926

        %t9 = icmp ult i31 10000, %j0
        %t10 = icmp sgt i31 -10000, %j0


	ret i31 %i0
end
