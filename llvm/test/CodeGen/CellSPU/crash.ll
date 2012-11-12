; RUN: llc %s -march=cellspu -o -
declare i8 @return_i8()
declare i16 @return_i16()
define void @testfunc() {
 %rv1 = call i8 @return_i8()
 %rv2 = call i16 @return_i16()
 ret void
}