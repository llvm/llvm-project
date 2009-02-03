; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%ZFunTy = type i33(i8 zeroext)
%SFunTy = type i33(i8 signext)

declare i16 @"test"(i16 signext %arg) signext 
declare i8  @"test2" (i16 zeroext %a2) zeroext 


define i33 @main(i33 %argc, i8 **%argv) {
    %val = trunc i33 %argc to i16
    %res = call i16 (i16 signext) signext *@test(i16 signext %val) signext
    %two = add i16 %res, %res
    %res2 = call i8 @test2(i16 %two zeroext) zeroext 
    %retVal = sext i16 %two to i33
    ret i33 %retVal
}
