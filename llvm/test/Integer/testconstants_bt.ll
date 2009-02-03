; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

@somestr = constant [11x i8] c"hello world"
@array   = constant [2 x i55] [ i55 12, i55 52 ]
           constant { i55, i55 } { i55 4, i55 3 }

 
define [2 x i55]* @testfunction(i55 %i0, i55 %j0)
begin
	ret [2x i55]* @array
end

define  i8* @otherfunc(i55, double)
begin
	%somestr = getelementptr [11x i8]* @somestr, i64 0, i64 0
	ret i8* %somestr
end

define i8* @yetanotherfunc(i55, double)
begin
	ret i8* null            ; Test null
end

define i55 @negativeUnsigned() {
        ret i55 -1
}

define i55 @largeSigned() {
       ret i55 3900000000
}
