; RUN: llc -mtriple=aarch64-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -disable-lsr -verify-machineinstrs -enable-misched=false -enable-post-misched=false -o - %s | FileCheck %s

define i64 @test(ptr %a) nounwind {
  ; CHECK: ldp	x{{[0-9]+}}, x{{[0-9]+}}
  ; CHECK-NOT: ldr
  %p1 = getelementptr inbounds i64, ptr %a, i32 64
  %tmp1 = load i64, ptr %p1, align 2
  %p2 = getelementptr inbounds i64, ptr %a, i32 63
  %tmp2 = load i64, ptr %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}
