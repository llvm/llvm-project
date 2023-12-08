; RUN: llvm-as < %s | llvm-dis -disable-output
; RUN: verify-uselistorder < %s
; PR4373

@foo = weak global { i32 } zeroinitializer              
@bar = weak global i32 0                

define void @test() {
entry:
        store { i32 } zeroinitializer, { i32 }* @foo
        store i32 1, i32* @bar
        ret void
}
