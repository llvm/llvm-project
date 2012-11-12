; RUN: llc < %s

define void @iterative_hash_host_wide_int() {
        %zero = alloca i32              ; <i32*> [#uses=2]
        %b = alloca i32         ; <i32*> [#uses=1]
        store i32 0, i32* %zero
        %tmp = load i32* %zero          ; <i32> [#uses=1]
        %tmp5 = bitcast i32 %tmp to i32         ; <i32> [#uses=1]
        %tmp6.u = add i32 %tmp5, 32             ; <i32> [#uses=1]
        %tmp6 = bitcast i32 %tmp6.u to i32              ; <i32> [#uses=1]
        %tmp7 = load i64* null          ; <i64> [#uses=1]
        %tmp6.upgrd.1 = trunc i32 %tmp6 to i8           ; <i8> [#uses=1]
        %shift.upgrd.2 = zext i8 %tmp6.upgrd.1 to i64           ; <i64> [#uses=1]
        %tmp8 = ashr i64 %tmp7, %shift.upgrd.2          ; <i64> [#uses=1]
        %tmp8.upgrd.3 = trunc i64 %tmp8 to i32          ; <i32> [#uses=1]
        store i32 %tmp8.upgrd.3, i32* %b
        unreachable
}

