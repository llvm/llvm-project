; RUN: llc -mtriple=thumbv7m-none-eabi %s -mcpu=cortex-m3 -o - | FileCheck %s
; RUN: llc -mtriple=thumbv7m-none-eabi %s -mcpu=cortex-m4 -o - | FileCheck %s
; RUN: llc -mtriple=thumbv8m-none-eabi %s -mcpu=cortex-m33 -o - | FileCheck %s

define void @test_loop_alignment(ptr %in, ptr  %out) optsize {
; CHECK-LABEL: test_loop_alignment:
; CHECK: mov{{.*}}, #0
; CHECK: .p2align 2

entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %in.addr = getelementptr inbounds i32, ptr %in, i32 %i
  %lhs = load i32, ptr %in.addr, align 4
  %res = mul nsw i32 %lhs, 5
  %out.addr = getelementptr inbounds i32, ptr %out, i32 %i
  store i32 %res, ptr %out.addr, align 4
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 1024
  br i1 %done, label %end, label %loop

end:
  ret void
}

define void @test_loop_alignment_minsize(ptr %in, ptr  %out) minsize {
; CHECK-LABEL: test_loop_alignment_minsize:
; CHECK: movs {{r[0-9]+}}, #0
; CHECK-NOT: .p2align

entry:
  br label %loop

loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %in.addr = getelementptr inbounds i32, ptr %in, i32 %i
  %lhs = load i32, ptr %in.addr, align 4
  %res = mul nsw i32 %lhs, 5
  %out.addr = getelementptr inbounds i32, ptr %out, i32 %i
  store i32 %res, ptr %out.addr, align 4
  %i.next = add i32 %i, 1
  %done = icmp eq i32 %i.next, 1024
  br i1 %done, label %end, label %loop

end:
  ret void
}
