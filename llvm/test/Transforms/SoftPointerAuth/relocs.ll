; RUN: opt < %s -passes=soft-ptrauth -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK-NOT: @test1_reloc
; CHECK: @test1 = internal global { ptr, i32, i32, ptr } { ptr null, i32 1342177280, i32 0, ptr null }, align 8

@test1_reloc = private constant { ptr, i32, i64, i64 } { ptr @test1_function, i32 1, i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i32, i32, ptr }, ptr @test1, i32 0, i32 3) to i64), i64 0 }, section "llvm.ptrauth", align 8
@test1 = internal constant { ptr, i32, i32, ptr } { ptr null, i32 1342177280, i32 0, ptr @test1_reloc }, align 8

define internal void @test1_function(ptr %0) {
entry:
  ret void
}

; CHECK: define private void @ptrauth_soft_init() {
; CHECK: [[T0:%.*]] = call ptr @__ptrauth_sign(ptr @test1_function, i32 1, i64 ptrtoint (ptr getelementptr inbounds ({ ptr, i32, i32, ptr }, ptr @test1, i32 0, i32 3) to i64)) [[NOUNWIND:#[0-9]+]]
; CHECK: store ptr [[T0]], ptr getelementptr inbounds ({ ptr, i32, i32, ptr }, ptr @test1, i32 0, i32 3)

; CHECK: attributes [[NOUNWIND]] = { nounwind }
