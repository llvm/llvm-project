; RUN: llvm-as < %s | llc -march=c

define i32 @foo() {
        ret i32 and (i32 123456, i32 ptrtoint (i32 ()* @foo to i32))
}
