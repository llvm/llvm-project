; Test that bugpoint can reduce the set of functions by replacing them with null.
;
; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%pluginext %s -output-prefix %t -replace-funcs-with-null -bugpoint-crash-decl-funcs -silence-passes -safe-run-llc
; REQUIRES: plugins

@foo2 = alias i32 (), ptr @foo

define i32 @foo() { ret i32 1 }

define i32 @test() {
	call i32 @test()
	ret i32 %1
}

define i32 @bar() { ret i32 2 }

@llvm.used = appending global [1 x ptr] [ptr @foo], section "llvm.metadata"
