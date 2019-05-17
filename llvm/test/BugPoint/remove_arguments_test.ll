; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -bugpoint-crashcalls -silence-passes
; RUN: llvm-dis %t-reduced-simplified.bc -o - | FileCheck %s
; REQUIRES: plugins

; Test to make sure that arguments are removed from the function if they are
; unnecessary. And clean up any types that frees up too.

; CHECK: ModuleID
; CHECK-NOT: struct.anon
%struct.anon = type { i32 }

declare i32 @test2()

; CHECK: define void @test() {
define i32 @test(i32 %A, %struct.anon* %B, float %C) {
	call i32 @test2()
	ret i32 %1
}
