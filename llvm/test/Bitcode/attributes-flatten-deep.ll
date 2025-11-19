; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; Test that flatten_depth attribute with integer values is properly handled
; in both attribute groups (flatten_depth=N syntax) and inline (flatten_depth(N) syntax)

; Test inline syntax
; CHECK: define void @test_inline() #0
define void @test_inline() flatten_depth(5) {
  ret void
}

; Test attribute group alone
; CHECK: define void @test_group_alone() #1
define void @test_group_alone() #1 {
  ret void
}

; Test attribute group with other attributes
; CHECK: define void @test_group_combined() #2
define void @test_group_combined() #2 {
  ret void
}

; CHECK: attributes #0 = { flatten_depth=5 }
attributes #0 = { flatten_depth=5 }

; CHECK: attributes #1 = { flatten_depth=3 }
attributes #1 = { flatten_depth=3 }

; CHECK: attributes #2 = { noinline nounwind flatten_depth=7 }
attributes #2 = { noinline nounwind flatten_depth=7 }
