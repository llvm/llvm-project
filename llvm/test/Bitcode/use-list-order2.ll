; RUN: verify-uselistorder %s

; Test 1
@g1 = global i8 0

declare void @llvm.donothing() nounwind readnone

define void @f.no_personality1() personality i8 0 {
  invoke void @llvm.donothing() to label %normal unwind label %exception
exception:
  %cleanup = landingpad i8 cleanup
  br label %normal
normal:
  ret void
}

; Test 2
@g2 = global i8 -1
@g3 = global i8 -1

define void @f.no_personality2() personality i8 -1 {
  invoke void @llvm.donothing() to label %normal unwind label %exception
exception:
  %cleanup = landingpad i8 cleanup
  br label %normal
normal:
  ret void
}

; Test 3
declare void @f1() prefix i32 1

define void @test1() {
  %t1 = alloca half  ; Implicit i32 1 used here.
  %t2 = alloca float
  ret void
}

; Test 4
declare void @f2() prefix i32 2

define void @test2(i32* %word) {
  %cmpxchg.0 = cmpxchg i32* %word, i32 0, i32 2 monotonic monotonic
  %cmpxchg.1 = cmpxchg i32* %word, i32 0, i32 2 acq_rel monotonic
  ret void
}

; Test 5
@g4 = global i32 3
@g5 = global i32 3
declare void @test3() prefix i32 3

; Test 6
@g6 = global i32 4
@g7 = global i32 4
declare void @c() prologue i32 4
