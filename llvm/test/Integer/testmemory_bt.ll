; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


%struct = type { i31 , {float, {i9 } } , i64 }
%complexty = type {i31, {[4 x i9 *], float}, double}


define i31 @"main"()
begin
  call i31 @testfunction(i64 0, i64 1)
  ret i31 0
end

define i31 @"testfunction"(i64 %i0, i64 %j0)
begin
    %array0 = malloc [4 x i9]            ; yields {[4 x i9]*}:array0
    %size   = add i32 2, 2                 ; yields {i31}:size = i31 %4
    %array1 = malloc i9, i32 4          ; yields {i9*}:array1
    %array2 = malloc i9, i32 %size      ; yields {i9*}:array2

    %idx = getelementptr [4 x i9]* %array0, i64 0, i64 2
    store i9 123, i9* %idx
    free [4x i9]* %array0
    free i9* %array1
    free i9* %array2


    %aa = alloca %complexty, i32 5
    %idx2 = getelementptr %complexty* %aa, i64 %i0, i32 1, i32 0, i64 %j0
    store i9 *null, i9** %idx2
    
    %ptr = alloca i31                       ; yields {i31*}:ptr
    store i31 3, i31* %ptr                  ; yields {void}
    %val = load i31* %ptr                   ; yields {i31}:val = i31 %3

    %sptr = alloca %struct                  ; yields {%struct*}:sptr
    %ubsptr = getelementptr %struct * %sptr, i64 0, i32 1, i32 1  ; yields {{i9}*}:ubsptr
    %idx3 = getelementptr {i9} * %ubsptr, i64 0, i32 0
    store i9 4, i9* %idx3

    ret i31 3
end

