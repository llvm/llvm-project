; RUN: llvm-as < %s | llc -march=c | grep __builtin_stack_save
; RUN: llvm-as < %s | llc -march=c | grep __builtin_stack_restore
; PR1028

declare i8* @llvm.stacksave()
declare void @llvm.stackrestore(i8*)

define i8* @test() {
    %s = call i8* @llvm.stacksave()
    call void @llvm.stackrestore(i8* %s)
    ret i8* %s
}
