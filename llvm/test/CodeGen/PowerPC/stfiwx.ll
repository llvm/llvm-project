; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=stfiwx -o %t1 -f
; RUN: grep stfiwx %t1
; RUN: not grep r1 %t1
; RUN: llvm-as < %s | \
; RUN:   llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=-stfiwx \
; RUN:   -o %t2 -f
; RUN: not grep stfiwx %t2
; RUN: grep r1 %t2

define void @test(float %a, i32* %b) {
        %tmp.2 = fptosi float %a to i32         ; <i32> [#uses=1]
        store i32 %tmp.2, i32* %b
        ret void
}

define void @test2(float %a, i32* %b, i32 %i) {
        %tmp.2 = getelementptr i32* %b, i32 1           ; <i32*> [#uses=1]
        %tmp.5 = getelementptr i32* %b, i32 %i          ; <i32*> [#uses=1]
        %tmp.7 = fptosi float %a to i32         ; <i32> [#uses=3]
        store i32 %tmp.7, i32* %tmp.5
        store i32 %tmp.7, i32* %tmp.2
        store i32 %tmp.7, i32* %b
        ret void
}

