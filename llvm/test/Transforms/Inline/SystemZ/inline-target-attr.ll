; RUN: opt < %s -mtriple=s390x-linux-gnu -S -passes=inline | FileCheck %s
; RUN: opt < %s -mtriple=s390x-linux-gnu -S -passes='cgscc(inline)' | FileCheck %s
; Check that we only inline when we have equal target attributes.

define i32 @foo() #0 {
entry:
  %call = call i32 (...) @baz()
  ret i32 %call
; CHECK-LABEL: foo
; CHECK: call i32 (...) @baz()
}

declare i32 @baz(...) #0

define i32 @features_subset() #1 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: features_subset
; CHECK: call i32 (...) @baz()
}

define i32 @features_equal() #0 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: features_equal
; CHECK: call i32 (...) @baz()
}

define i32 @features_different() #2 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: features_different
; CHECK: call i32 @foo()
}


attributes #0 = { "target-cpu"="generic" "target-features"="+guarded-storage" }
attributes #1 = { "target-cpu"="generic" "target-features"="+guarded-storage,+enhanced-sort" }
attributes #2 = { "target-cpu"="generic" "target-features"="+concurrent-functions" }
