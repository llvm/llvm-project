; RUN: llc -march=hexagon -enable-save-restore-long -hexagon-initial-cfg-cleanup=0 < %s | FileCheck %s

; Check that the -long-calls feature is supported by the backend.

; CHECK: call ##foo
; CHECK: jump ##__restore
define i64 @test_longcall(i32 %x, i32 %y) #0 {
entry:
  %add = add nsw i32 %x, 5
  %call = tail call i64 @foo(i32 %add) #1
  %conv = sext i32 %y to i64
  %add1 = add nsw i64 %call, %conv
  ret i64 %add1
}

; CHECK: jump ##foo
define i64 @test_longtailcall(i32 %x, i32 %y) #1 {
entry:
  %add = add nsw i32 %x, 5
  %call = tail call i64 @foo(i32 %add) #1
  ret i64 %call
}

; CHECK: call ##bar
define i64 @test_longnoret(i32 %x, i32 %y) #2 {
entry:
  %add = add nsw i32 %x, 5
  %0 = tail call i64 @bar(i32 %add) #6
  unreachable
}

; CHECK: call foo
; CHECK: jump ##__restore
; The restore call will still be long because of the enable-save-restore-long
; option being used.
define i64 @test_shortcall(i32 %x, i32 %y) #3 {
entry:
  %add = add nsw i32 %x, 5
  %call = tail call i64 @foo(i32 %add) #1
  %conv = sext i32 %y to i64
  %add1 = add nsw i64 %call, %conv
  ret i64 %add1
}

; CHECK: jump foo
define i64 @test_shorttailcall(i32 %x, i32 %y) #4 {
entry:
  %add = add nsw i32 %x, 5
  %call = tail call i64 @foo(i32 %add) #1
  ret i64 %call
}

; CHECK: call bar
define i64 @test_shortnoret(i32 %x, i32 %y) #5 {
entry:
  %add = add nsw i32 %x, 5
  %0 = tail call i64 @bar(i32 %add) #6
  unreachable
}

declare i64 @foo(i32) #1
declare i64 @bar(i32) #6

attributes #0 = { minsize nounwind "target-cpu"="hexagonv60" "target-features"="+long-calls" }
attributes #1 = { nounwind "target-cpu"="hexagonv60" "target-features"="+long-calls" }
attributes #2 = { noreturn nounwind "target-cpu"="hexagonv60" "target-features"="+long-calls" }

attributes #3 = { minsize nounwind "target-cpu"="hexagonv60" "target-features"="-long-calls" }
attributes #4 = { nounwind "target-cpu"="hexagonv60" "target-features"="-long-calls" }
attributes #5 = { noreturn nounwind "target-cpu"="hexagonv60" "target-features"="-long-calls" }

attributes #6 = { noreturn nounwind "target-cpu"="hexagonv60" }
