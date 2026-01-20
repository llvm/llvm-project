; RUN: llvm-as %s -o /dev/null
; Test that null personality is silently accepted as "no personality" for
; backward compatibility with legacy IR.

define void @test() personality ptr null {
 ret void
}
