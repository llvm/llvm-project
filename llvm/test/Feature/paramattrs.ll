; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%ZFunTy = type i32(i8 zeroext)
%SFunTy = type i32(i8 signext)

declare i16 @"test"(i16 signext %arg) signext 
declare i8 @"test2" (i16 zeroext %a2) zeroext 

declare i32 @"test3"(i32* noalias %p)

declare void @exit(i32) noreturn nounwind

define i32 @main(i32 inreg %argc, i8 ** inreg %argv) nounwind {
    %val = trunc i32 %argc to i16
    %res1 = call i16 (i16 signext) signext *@test(i16 signext %val) signext
    %two = add i16 %res1, %res1
    %res2 = call i8 @test2(i16 %two zeroext) zeroext 
    %retVal = sext i16 %two to i32
    ret i32 %retVal
}
