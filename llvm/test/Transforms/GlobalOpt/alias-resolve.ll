; RUN: llvm-as < %s | opt -globalopt | llvm-dis > %t
; RUN: cat %t | grep foo1 | count 1
; RUN: cat %t | grep foo2 | count 4
; RUN: cat %t | grep bar1 | count 1
; RUN: cat %t | grep bar2 | count 4

@foo1 = alias void ()* @foo2
@foo2 = alias weak void()* @bar1
@bar1  = alias void ()* @bar2

declare void @bar2()

define void @baz() {
entry:
        call void @foo1()
        call void @foo2()
        call void @bar1()
        ret void
}
