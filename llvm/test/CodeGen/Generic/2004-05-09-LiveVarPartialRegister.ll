; RUN: llvm-as < %s | llc
@global_long_1 = linkonce global i64 7          ; <i64*> [#uses=1]
@global_long_2 = linkonce global i64 49         ; <i64*> [#uses=1]

define i32 @main() {
        %l1 = load i64* @global_long_1          ; <i64> [#uses=1]
        %l2 = load i64* @global_long_2          ; <i64> [#uses=1]
        %cond = icmp sle i64 %l1, %l2           ; <i1> [#uses=1]
        %cast2 = zext i1 %cond to i32           ; <i32> [#uses=1]
        %RV = sub i32 1, %cast2         ; <i32> [#uses=1]
        ret i32 %RV
}

