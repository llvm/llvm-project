; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll


        %complexty = type { i32, { [4 x i8*], float }, double }
        %struct = type { i32, { float, { i8 } }, i64 }

define i32 @main() {
        call i32 @testfunction( i64 0, i64 1 )          ; <i32>:1 [#uses=0]
        ret i32 0
}

define i32 @testfunction(i64 %i0, i64 %j0) {
        %array0 = malloc [4 x i8]               ; <[4 x i8]*> [#uses=2]
        %size = add i32 2, 2            ; <i32> [#uses=1]
        %array1 = malloc i8, i32 4              ; <i8*> [#uses=1]
        %array2 = malloc i8, i32 %size          ; <i8*> [#uses=1]
        %idx = getelementptr [4 x i8]* %array0, i64 0, i64 2            ; <i8*> [#uses=1]
        store i8 123, i8* %idx
        free [4 x i8]* %array0
        free i8* %array1
        free i8* %array2
        %aa = alloca %complexty, i32 5          ; <%complexty*> [#uses=1]
        %idx2 = getelementptr %complexty* %aa, i64 %i0, i32 1, i32 0, i64 %j0           ; <i8**> [#uses=1]
        store i8* null, i8** %idx2
        %ptr = alloca i32               ; <i32*> [#uses=2]
        store i32 3, i32* %ptr
        %val = load i32* %ptr           ; <i32> [#uses=0]
        %sptr = alloca %struct          ; <%struct*> [#uses=1]
        %ubsptr = getelementptr %struct* %sptr, i64 0, i32 1, i32 1             ; <{ i8 }*> [#uses=1]
        %idx3 = getelementptr { i8 }* %ubsptr, i64 0, i32 0             ; <i8*> [#uses=1]
        store i8 4, i8* %idx3
        ret i32 3
}

