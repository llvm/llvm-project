; RUN: opt < %s -passes=adce -S | FileCheck %s

; Verify that a call to instrument a constant is deleted.

@__profc_foo = private global [1 x i64] zeroinitializer, section "__llvm_prf_cnts", align 8
@__profd_foo = private global { i64, i64, ptr, ptr, ptr, i32, [1 x i16] } { i64 6699318081062747564, i64 0, ptr @__profc_foo, ptr @foo, ptr null, i32 1, [1 x i16] [i16 1] }, section "__llvm_prf_data", align 8

define i32 @foo() {
; CHECK-NOT: call void @__llvm_profile_instrument_target
entry:
  tail call void @__llvm_profile_instrument_target(i64 ptrtoint (ptr @bar to i64), ptr @__profd_foo, i32 0)
  %call = tail call i32 @bar(i32 21)
  ret i32 %call
}

declare i32 @bar(i32)

declare void @__llvm_profile_instrument_target(i64, ptr, i32)
