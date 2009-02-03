; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep fcmp | count 3
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep ret | grep 1

define zeroext i8 @t1(float %x, float %y) nounwind {
       %a = fcmp ueq float %x, %y             ; <i1> [#uses=1]
       %b = fcmp uno float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

define zeroext i8 @t2(float %x, float %y) nounwind {
       %a = fcmp olt float %x, %y             ; <i1> [#uses=1]
       %b = fcmp oeq float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

define zeroext i8 @t3(float %x, float %y) nounwind {
       %a = fcmp ult float %x, %y             ; <i1> [#uses=1]
       %b = fcmp uge float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}

define zeroext i8 @t4(float %x, float %y) nounwind {
       %a = fcmp ult float %x, %y             ; <i1> [#uses=1]
       %b = fcmp ugt float %x, %y               ; <i1> [#uses=1]
       %c = or i1 %a, %b
       %retval = zext i1 %c to i8
       ret i8 %retval
}
