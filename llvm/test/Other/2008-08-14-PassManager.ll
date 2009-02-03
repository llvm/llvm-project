; RUN:  llvm-as < %s |  opt -loop-deletion -loop-index-split -disable-output
; PR 2640
define i32 @test1() {
       ret i32 0;
}
