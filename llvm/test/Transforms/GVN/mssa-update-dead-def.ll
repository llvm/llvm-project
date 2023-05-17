; RUN: opt -passes='require<memoryssa>,gvn' -verify-memoryssa -S %s | FileCheck %s

; This is a regression test for a bug in MemorySSA updater.
; Make sure that we don't crash and end up with a valid MemorySSA.

; CHECK: @test()
define void @test() personality ptr null {
  invoke void @bar()
          to label %bar.normal unwind label %exceptional

bar.normal:
  ret void

dead.block:
  br label %baz.invoke

baz.invoke:
  invoke void @baz()
          to label %baz.normal unwind label %exceptional

baz.normal:
  ret void

exceptional:
  %tmp9 = landingpad { ptr, i32 }
          cleanup
  call void @foo()
  ret void
}

declare void @foo()
declare void @bar()
declare void @baz()
