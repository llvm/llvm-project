; RUN: llc -march=bpfeb -show-mc-encoding < %s | FileCheck %s

; Function Attrs: nounwind
define i32 @bpf_prog1(i8* nocapture readnone) local_unnamed_addr #0 {
; CHECK: r1 = 590618314553 ll   # encoding: [0x18,0x10,0x00,0x00,0x83,0x98,0x47,0x39,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x89]
; CHECK: r1 += -1879113726     # encoding: [0x07,0x10,0x00,0x00,0x8f,0xff,0x00,0x02]
; CHECK: r0 = *(u64 *)(r1 + 0) # encoding: [0x79,0x01,0x00,0x00,0x00,0x00,0x00,0x00]
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

