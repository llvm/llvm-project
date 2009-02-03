; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep fcmp | count 3
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep ret | grep 0

define zeroext i8 @t1(float %x, float %y) nounwind {
       %a = fcmp ueq float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

define zeroext i8 @t2(float %x, float %y) nounwind {
       %a = fcmp olt float %x, %y
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

define zeroext i8 @t3(float %x, float %y) nounwind {
       %a = fcmp oge float %x, %y
       %b = fcmp uno float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

define zeroext i8 @t4(float %x, float %y) nounwind {
       %a = fcmp one float %y, %x
       %b = fcmp ord float %x, %y
       %c = and i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}
