; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

        %inners = type { float, { i8 } }
        %struct = type { i32, %inners, i64 }

define i32 @testfunction(i32 %i0, i32 %j0) {
        alloca i8, i32 5                ; <i8*>:1 [#uses=0]
        %ptr = alloca i32               ; <i32*> [#uses=2]
        store i32 3, i32* %ptr
        %val = load i32, i32* %ptr           ; <i32> [#uses=0]
        %sptr = alloca %struct          ; <%struct*> [#uses=2]
        %nsptr = getelementptr %struct, %struct* %sptr, i64 0, i32 1             ; <%inners*> [#uses=1]
        %ubsptr = getelementptr %inners, %inners* %nsptr, i64 0, i32 1           ; <{ i8 }*> [#uses=1]
        %idx = getelementptr { i8 }, { i8 }* %ubsptr, i64 0, i32 0              ; <i8*> [#uses=1]
        store i8 4, i8* %idx
        %fptr = getelementptr %struct, %struct* %sptr, i64 0, i32 1, i32 0               ; <float*> [#uses=1]
        store float 4.000000e+00, float* %fptr
        ret i32 3
}

