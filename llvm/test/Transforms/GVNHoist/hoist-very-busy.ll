; RUN: opt -S -passes=gvn-hoist < %s | FileCheck %s

%struct.__jmp_buf_tag = type { [8 x i64], i32 }

; Check that hoisting only happens when the expression is very busy.
; CHECK: store
; CHECK: store

@test_exit_buf = global %struct.__jmp_buf_tag zeroinitializer
@G = global i32 0

define void @test_command(i32 %c1) {
entry:
  switch i32 %c1, label %exit [
    i32 0, label %sw0
    i32 1, label %sw1
  ]

sw0:
  store i32 1, i32* @G
  br label %exit

sw1:
  store i32 1, i32* @G
  br label %exit

exit:
  call void @longjmp(%struct.__jmp_buf_tag* @test_exit_buf, i32 1) #0
  unreachable
}

declare void @longjmp(%struct.__jmp_buf_tag*, i32) #0

attributes #0 = { noreturn nounwind }

; Check that the store is hoisted.
; CHECK-LABEL: define void @fun(
; CHECK: store
; CHECK-NOT: store

define void @fun() {
entry:
  br label %if.then

if.then:                                          ; preds = %entry
  br i1 undef, label %sw0, label %sw1

sw0:
  store i32 1, i32* @G
  unreachable

sw1:
  store i32 1, i32* @G
  ret void
}
