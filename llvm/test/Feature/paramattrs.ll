; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

%ZFunTy = type i32(i8)
%SFunTy = type i32(i8)

declare signext i16 @"test"(i16 signext %arg)  
declare zeroext i8 @"test2" (i16 zeroext %a2) 

declare i32 @"test3"(ptr noalias %p)

declare void @exit(i32) noreturn nounwind

define i32 @main(i32 inreg %argc, ptr inreg %argv) nounwind {
    %val = trunc i32 %argc to i16
    %res1 = call signext i16 (i16 )@test(i16 signext %val) 
    %two = add i16 %res1, %res1
    %res2 = call zeroext i8 @test2(i16 zeroext %two )  
    %retVal = sext i16 %two to i32
    ret i32 %retVal
}

declare void @function_to_resolve_eagerly() nonlazybind
