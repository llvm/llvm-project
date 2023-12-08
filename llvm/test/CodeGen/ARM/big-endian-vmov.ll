; RUN: llc < %s -mtriple armv7-eabi -o - | FileCheck %s --check-prefixes=CHECK,CHECK-LE
; RUN: llc < %s -mtriple armebv7-eabi -o - | FileCheck %s --check-prefixes=CHECK,CHECK-BE

; CHECK-LABEL: vmov_i8
; CHECK-LE: vmov.i64 d0, #0xff00000000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <8 x i8> @vmov_i8() {
  ret <8 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 -1>
}

; CHECK-LABEL: vmov_i16_a:
; CHECK-LE: vmov.i64 d0, #0xffff000000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xffff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <4 x i16> @vmov_i16_a() {
  ret <4 x i16> <i16 0, i16 0, i16 0, i16 -1>
}

; CHECK-LABEL: vmov_i16_b:
; CHECK-LE: vmov.i64 d0, #0xff000000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <4 x i16> @vmov_i16_b() {
  ret <4 x i16> <i16 0, i16 0, i16 0, i16 255>
}

; CHECK-LABEL: vmov_i16_c:
; CHECK-LE: vmov.i64 d0, #0xff00000000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff00{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <4 x i16> @vmov_i16_c() {
  ret <4 x i16> <i16 0, i16 0, i16 0, i16 65280>
}

; CHECK-LABEL: vmov_i32_a:
; CHECK-LE: vmov.i64 d0, #0xffffffff00000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xffffffff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <2 x i32> @vmov_i32_a() {
  ret <2 x i32> <i32 0, i32 -1>
}

; CHECK-LABEL: vmov_i32_b:
; CHECK-LE: vmov.i64 d0, #0xff00000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <2 x i32> @vmov_i32_b() {
  ret <2 x i32> <i32 0, i32 255>
}

; CHECK-LABEL: vmov_i32_c:
; CHECK-LE: vmov.i64 d0, #0xff0000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff00{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <2 x i32> @vmov_i32_c() {
  ret <2 x i32> <i32 0, i32 65280>
}

; CHECK-LABEL: vmov_i32_d:
; CHECK-LE: vmov.i64 d0, #0xff000000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff0000{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <2 x i32> @vmov_i32_d() {
  ret <2 x i32> <i32 0, i32 16711680>
}

; CHECK-LABEL: vmov_i32_e:
; CHECK-LE: vmov.i64 d0, #0xff00000000000000{{$}}
; CHECK-BE: vmov.i64 d0, #0xff000000{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <2 x i32> @vmov_i32_e() {
  ret <2 x i32> <i32 0, i32 4278190080>
}

; CHECK-LABEL: vmov_i64_a:
; CHECK: vmov.i8 d0, #0xff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <1 x i64> @vmov_i64_a() {
  ret <1 x i64> <i64 -1>
}

; CHECK-LABEL: vmov_i64_b:
; CHECK: vmov.i64 d0, #0xffff00ff0000ff{{$}}
; CHECK-NEXT: bx lr
define arm_aapcs_vfpcc <1 x i64> @vmov_i64_b() {
  ret <1 x i64> <i64 72056498804490495>
}
