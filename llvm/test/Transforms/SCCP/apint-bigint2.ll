; RUN:  llvm-as < %s | opt -sccp | llvm-dis | not grep load

@Y = constant [6 x i101] [ i101 12, i101 123456789000000, i101 -12,
                           i101 -123456789000000, i101 0,i101 9123456789000000]

define i101 @array()
{
Head:
   %A = getelementptr [6 x i101]* @Y, i32 0, i32 1
   %B = load i101* %A
   %D = and i101 %B, 1
   %DD = or i101 %D, 1
   %E = trunc i101 %DD to i32
   %F = getelementptr [6 x i101]* @Y, i32 0, i32 %E
   %G = load i101* %F
 
   ret i101 %G
}
