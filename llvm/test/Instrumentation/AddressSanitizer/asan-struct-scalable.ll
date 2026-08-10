; RUN: opt -passes=asan -disable-output -S %s
; Check not crash.

define void @test() #0 {
entry:
  %t0 = alloca { <vscale x 2 x i32>, <vscale x 2 x i32> }, align 4
  call void null(ptr null, ptr %t0, i64 0)
  ret void
}

attributes #0 = { sanitize_address }
