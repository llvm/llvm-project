; RUN: opt -globalopt < %s -S -o - | FileCheck %s

@GV1 = internal global i64 1
@GV2 = internal global i32 0

; CHECK: @GV1 = internal unnamed_addr constant i64 1
; CHECK: @GV2 = internal unnamed_addr global i32 0

define void @test1() {
entry:
  %0 = load atomic i8, i8* bitcast (i64* @GV1 to i8*) acquire, align 8
  ret void
}

; PR17163
define void @test2a() {
entry:
  store atomic i32 10, i32* @GV2 seq_cst, align 4
  ret void
}
define i32 @test2b() {
entry:
  %atomic-load = load atomic i32, i32* @GV2 seq_cst, align 4
  ret i32 %atomic-load
}
