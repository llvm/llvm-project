; Test various forms of calls.

; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   grep "bl " | count 1
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   grep "bctrl" | count 1
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   grep "bla " | count 1

declare void @foo()

define void @test_direct() {
        call void @foo( )
        ret void
}

define void @test_indirect(ptr %fp) {
        call void %fp( )
        ret void
}

define void @test_abs() {
        %fp = inttoptr i32 400 to ptr              ; <ptr> [#uses=1]
        call void %fp( )
        ret void
}

