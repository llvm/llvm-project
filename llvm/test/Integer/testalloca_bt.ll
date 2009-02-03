; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


%inners = type {float, {i8 } }
%struct = type { i33 , {float, {i8 } } , i64 }


define i33 @testfunction(i33 %i0, i33 %j0)
begin
    alloca i8, i32 5
    %ptr = alloca i33                       ; yields {i33*}:ptr
    store i33 3, i33* %ptr                  ; yields {void}
    %val = load i33* %ptr                   ; yields {i33}:val = i33 %3

    %sptr = alloca %struct                  ; yields {%struct*}:sptr
    %nsptr = getelementptr %struct * %sptr, i64 0, i32 1  ; yields {inners*}:nsptr
    %ubsptr = getelementptr %inners * %nsptr, i64 0, i32 1  ; yields {{i8}*}:ubsptr
    %idx = getelementptr {i8} * %ubsptr, i64 0, i32 0
    store i8 4, i8* %idx
    
    %fptr = getelementptr %struct * %sptr, i64 0, i32 1, i32 0  ; yields {float*}:fptr
    store float 4.0, float * %fptr
    
    ret i33 3
end

