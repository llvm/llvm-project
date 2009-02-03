; RUN: llvm-as < %s | llc -march=c

; This testcase breaks the C backend, because gcc doesn't like (...) functions
; with no arguments at all.

define void @test(i64 %Ptr) {
        %P = inttoptr i64 %Ptr to void (...)*           ; <void (...)*> [#uses=1]
        call void (...)* %P( i64 %Ptr )
        ret void
}

