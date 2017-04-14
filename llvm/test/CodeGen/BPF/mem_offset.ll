; RUN: llc -march=bpfel -show-mc-encoding < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @bpf_prog1(i8* nocapture readnone) local_unnamed_addr #0 {
; CHECK: r1 += -1879113726 # encoding: [0x07,0x01,0x00,0x00,0x02,0x00,0xff,0x8f]
; CHECK: r0 = *(u64 *)(r1 + 0) # encoding: [0x79,0x10,0x00,0x00,0x00,0x00,0x00,0x00]
  %2 = alloca i64, align 8
  %3 = bitcast i64* %2 to i8*
  store volatile i64 590618314553, i64* %2, align 8
  %4 = load volatile i64, i64* %2, align 8
  %5 = add i64 %4, -1879113726
  %6 = inttoptr i64 %5 to i64*
  %7 = load i64, i64* %6, align 8
  %8 = trunc i64 %7 to i32
  ret i32 %8
}

