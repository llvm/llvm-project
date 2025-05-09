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

define i32 @bar() #1 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: bar
; CHECK: call i32 @foo()
}

define i32 @qux() #0 {
entry:
  %call = call i32 @foo()
  ret i32 %call
; CHECK-LABEL: qux
; CHECK: call i32 (...) @baz()
}

define i32 @quux() #2 {
entry:
  %call = call i32 @bar()
  ret i32 %call
; CHECK-LABEL: quux
; CHECK: call i32 @bar()
}


attributes #0 = { "target-cpu"="generic" "target-features"="+guarded-storage" }
attributes #1 = { "target-cpu"="generic" "target-features"="+guarded-storage,+enhanced-sort" }
attributes #2 = { "target-cpu"="generic" "target-features"="+concurrent-functions" }
