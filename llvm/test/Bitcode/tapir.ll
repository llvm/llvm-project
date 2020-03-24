; Test that bitcode reader and writer can handle Tapir extensions.
;
; Thanks to @joshbohde for the code for this test case.
;;
; RUN: llvm-as %s -o=- | llvm-dis

declare token @llvm.syncregion.start() #3

declare void @print(i32)

define void @test (i32 %n) {
entry:
    %i = alloca i32
    store i32 0, i32* %i  ; store it back
    br label %loop.1.entry

loop.1.entry:
    %tmp = load i32, i32* %i
    %cmp = icmp ne i32 %tmp, %n
    br i1 %cmp, label %loop.1.body, label %loop.1.exit

loop.1.body:
    %syncreg = call token @llvm.syncregion.start()
    detach within %syncreg, label %loop.1.body.1, label %loop.1.body.2

loop.1.body.1:
    %tmp2 = load i32, i32* %i
    call void @print(i32 %tmp2)
    reattach within %syncreg, label %loop.1.body.2

loop.1.body.2:
    %tmp3 = load i32, i32* %i
    %tmp4 = add i32 %tmp3, 1   ; increment it
    store i32 %tmp4, i32* %i  ; store it back
    br label %loop.1.final

loop.1.final:
    sync within %syncreg, label %loop.1.entry

loop.1.exit:
    ret void
}
