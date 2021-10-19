; RUN: %lli -jit-kind=mcjit -O0 %s
; RUN: %lli -O0 %s

; This test checks that common symbols have been allocated addresses honouring
; the alignment requirement.

@CS1 = common global i32 0, align 16
@CS2 = common global i8 0, align 1
@CS3 = common global i32 0, align 16

define i32 @main() nounwind {
entry:
    %retval = alloca i32, align 4
    %ptr = alloca i32, align 4
    store i32 0, i32* %retval
    store i32 ptrtoint (i32* @CS3 to i32), i32* %ptr, align 4
    %0 = load i32, i32* %ptr, align 4
    %and = and i32 %0, 15
    %tobool = icmp ne i32 %and, 0
    br i1 %tobool, label %if.then, label %if.else

if.then:                                          ; preds = %entry
    store i32 1, i32* %retval
    br label %return

if.else:                                          ; preds = %entry
    store i32 0, i32* %retval
    br label %return

return:                                           ; preds = %if.else, %if.then
    %1 = load i32, i32* %retval
    ret i32 %1
}
