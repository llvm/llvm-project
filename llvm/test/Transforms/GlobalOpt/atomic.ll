; RUN: opt -globalopt < %s -S -o - | FileCheck %s

@GV1 = internal global i64 1
; CHECK: @GV1 = internal unnamed_addr constant i64 1

define void @test1() {
entry:
  %0 = load atomic i8* bitcast (i64* @GV1 to i8*) acquire, align 8
  ret void
}
