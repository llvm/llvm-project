; RUN: llvm-as < %s | opt -sccp | llvm-dis | grep {ret i101 12}

@Y = constant [6 x i101] [ i101 12, i101 123456789000000, i101 -12,i101 
-123456789000000, i101 0,i101 9123456789000000]

define i101 @array()
{
Head:
   %A = getelementptr [6 x i101]* @Y, i32 0, i32 1

   %B = load i101* %A
   %C = icmp sge i101 %B, 1
   br i1 %C, label %True, label %False
True:
   %D = and i101 %B, 1
   %E = trunc i101 %D to i32
   %F = getelementptr [6 x i101]* @Y, i32 0, i32 %E
   %G = load i101* %F
   br label %False
False:
   %H = phi i101 [%G, %True], [-1, %Head]
   ret i101 %H
}
