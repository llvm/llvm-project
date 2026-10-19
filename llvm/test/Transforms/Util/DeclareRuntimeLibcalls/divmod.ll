; REQUIRES: arm-registered-target

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-none-eabi < %s | FileCheck -check-prefix=AEABI %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-windows-msvc < %s | FileCheck -check-prefix=WINDOWS %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-apple-ios5.0 < %s | FileCheck -check-prefix=DARWIN %s

; AEABI: declare arm_aapcscc inreg { i32, i32 } @__aeabi_idivmod(i32 signext, i32 signext) #0
; AEABI: declare arm_aapcscc inreg { i64, i64 } @__aeabi_ldivmod(i64 signext, i64 signext) #0
; AEABI: declare arm_aapcscc inreg { i32, i32 } @__aeabi_uidivmod(i32 zeroext, i32 zeroext) #0
; AEABI: declare arm_aapcscc inreg { i64, i64 } @__aeabi_uldivmod(i64 zeroext, i64 zeroext) #0
; AEABI: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }

; WINDOWS: declare arm_aapcscc inreg { i32, i32 } @__rt_sdiv(i32 signext, i32 signext) #0
; WINDOWS: declare arm_aapcscc inreg { i64, i64 } @__rt_sdiv64(i64 signext, i64 signext) #0
; WINDOWS: declare arm_aapcscc inreg { i32, i32 } @__rt_udiv(i32 zeroext, i32 zeroext) #0
; WINDOWS: declare arm_aapcscc inreg { i64, i64 } @__rt_udiv64(i64 zeroext, i64 zeroext) #0
; WINDOWS: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }

; DARWIN: declare void @__divmodsi4(...)
; DARWIN: declare void @__udivmodsi4(...)
; DARWIN-NOT: inreg { i32, i32 } @__divmodsi4

define void @f() {
  ret void
}
