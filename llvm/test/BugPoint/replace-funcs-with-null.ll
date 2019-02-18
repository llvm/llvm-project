; Test that bugpoint can reduce the set of functions by replacing them with null.
;
; RUN: bugpoint -load %llvmshlibdir/BugpointPasses%shlibext %s -output-prefix %t -replace-funcs-with-null -bugpoint-crash-decl-funcs -silence-passes -safe-run-llc
; REQUIRES: loadable_module

@foo2 = alias i32 (), i32 ()* @foo

define i32 @foo() { ret i32 1 }

define i32 @test() {
	call i32 @test()
	ret i32 %1
}

define i32 @bar() { ret i32 2 }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 ()* @foo to i8*)], section "llvm.metadata"
