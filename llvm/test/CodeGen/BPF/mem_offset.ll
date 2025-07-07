; RUN: llc -mtriple=bpfel -show-mc-encoding < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @bpf_prog1(ptr nocapture readnone) local_unnamed_addr #0 {
; CHECK: r1 += -1879113726 # encoding: [0x07,0x01,0x00,0x00,0x02,0x00,0xff,0x8f]
; CHECK: r0 = *(u64 *)(r1 + 0) # encoding: [0x79,0x10,0x00,0x00,0x00,0x00,0x00,0x00]
  %2 = alloca i64, align 8
  store volatile i64 590618314553, ptr %2, align 8
  %3 = load volatile i64, ptr %2, align 8
  %4 = add i64 %3, -1879113726
  %5 = inttoptr i64 %4 to ptr
  %6 = load i64, ptr %5, align 8
  %7 = trunc i64 %6 to i32
  ret i32 %7
}

