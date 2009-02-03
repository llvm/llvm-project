; RUN: llvm-as < %s | llc -march=arm | grep moveq 
; RUN: llvm-as < %s | llc -march=arm -mattr=+vfp2 | grep movvs

define i32 @f7(float %a, float %b) {
entry:
    %tmp = fcmp ueq float %a,%b
    %retval = select i1 %tmp, i32 666, i32 42
    ret i32 %retval
}

