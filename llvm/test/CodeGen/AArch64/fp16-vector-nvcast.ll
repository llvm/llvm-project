; RUN: llc < %s -asm-verbose=false -mtriple=aarch64-none-eabi | FileCheck %s

; Test pattern (v4f16 (AArch64NvCast (v2i32 FPR64:$src)))
define void @nvcast_v2i32(<4 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v2i32:
; CHECK-NEXT: movi v[[REG:[0-9]+]].2s, #171, lsl #16
; CHECK-NEXT: str d[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <4 x half> <half 0xH0000, half 0xH00AB, half 0xH0000, half 0xH00AB>, <4 x half>* %a
  ret void
}


; Test pattern (v4f16 (AArch64NvCast (v4i16 FPR64:$src)))
define void @nvcast_v4i16(<4 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v4i16:
; CHECK-NEXT: movi v[[REG:[0-9]+]].4h, #171
; CHECK-NEXT: str d[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <4 x half> <half 0xH00AB, half 0xH00AB, half 0xH00AB, half 0xH00AB>, <4 x half>* %a
  ret void
}


; Test pattern (v4f16 (AArch64NvCast (v8i8 FPR64:$src)))
define void @nvcast_v8i8(<4 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v8i8:
; CHECK-NEXT: movi v[[REG:[0-9]+]].8b, #171
; CHECK-NEXT: str d[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <4 x half> <half 0xHABAB, half 0xHABAB, half 0xHABAB, half 0xHABAB>, <4 x half>* %a
  ret void
}


; Test pattern (v4f16 (AArch64NvCast (f64 FPR64:$src)))
define void @nvcast_f64(<4 x half>* %a) #0 {
; CHECK-LABEL: nvcast_f64:
; CHECK-NEXT: movi d[[REG:[0-9]+]], #0000000000000000
; CHECK-NEXT: str d[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <4 x half> zeroinitializer, <4 x half>* %a
  ret void
}

; Test pattern (v8f16 (AArch64NvCast (v4i32 FPR128:$src)))
define void @nvcast_v4i32(<8 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v4i32:
; CHECK-NEXT: movi v[[REG:[0-9]+]].4s, #171, lsl #16
; CHECK-NEXT: str q[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <8 x half> <half 0xH0000, half 0xH00AB, half 0xH0000, half 0xH00AB, half 0xH0000, half 0xH00AB, half 0xH0000, half 0xH00AB>, <8 x half>* %a
  ret void
}


; Test pattern (v8f16 (AArch64NvCast (v8i16 FPR128:$src)))
define void @nvcast_v8i16(<8 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v8i16:
; CHECK-NEXT: movi v[[REG:[0-9]+]].8h, #171
; CHECK-NEXT: str q[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <8 x half> <half 0xH00AB, half 0xH00AB, half 0xH00AB, half 0xH00AB, half 0xH00AB, half 0xH00AB, half 0xH00AB, half 0xH00AB>, <8 x half>* %a
  ret void
}


; Test pattern (v8f16 (AArch64NvCast (v16i8 FPR128:$src)))
define void @nvcast_v16i8(<8 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v16i8:
; CHECK-NEXT: movi v[[REG:[0-9]+]].16b, #171
; CHECK-NEXT: str q[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <8 x half> <half 0xHABAB, half 0xHABAB, half 0xHABAB, half 0xHABAB, half 0xHABAB, half 0xHABAB, half 0xHABAB, half 0xHABAB>, <8 x half>* %a
  ret void
}


; Test pattern (v8f16 (AArch64NvCast (v2i64 FPR128:$src)))
define void @nvcast_v2i64(<8 x half>* %a) #0 {
; CHECK-LABEL: nvcast_v2i64:
; CHECK-NEXT: movi v[[REG:[0-9]+]].2d, #0000000000000000
; CHECK-NEXT: str q[[REG]], [x0]
; CHECK-NEXT: ret
  store volatile <8 x half> zeroinitializer, <8 x half>* %a
  ret void
}

attributes #0 = { nounwind }
