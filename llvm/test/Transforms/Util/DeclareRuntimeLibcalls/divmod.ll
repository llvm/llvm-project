; REQUIRES: arm-registered-target

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-none-eabi < %s | FileCheck -check-prefix=AEABI %s
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=thumbv7-windows-msvc < %s | FileCheck -check-prefix=WINDOWS %s

; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=armv7-apple-ios5.0 < %s | FileCheck -check-prefix=DARWIN %s

; AEABI: declare arm_aapcscc inreg { i32, i32 } @__aeabi_idivmod(i32, i32) #0
; AEABI: declare arm_aapcscc inreg { i64, i64 } @__aeabi_ldivmod(i64, i64) #0
; AEABI: declare arm_aapcscc inreg { i32, i32 } @__aeabi_uidivmod(i32, i32) #0
; AEABI: declare arm_aapcscc inreg { i64, i64 } @__aeabi_uldivmod(i64, i64) #0
; AEABI: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }

; WINDOWS: declare arm_aapcscc inreg { i32, i32 } @__rt_sdiv(i32, i32) #0
; WINDOWS: declare arm_aapcscc inreg { i64, i64 } @__rt_sdiv64(i64, i64) #0
; WINDOWS: declare arm_aapcscc inreg { i32, i32 } @__rt_udiv(i32, i32) #0
; WINDOWS: declare arm_aapcscc inreg { i64, i64 } @__rt_udiv64(i64, i64) #0
; WINDOWS: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }

; DARWIN: declare i32 @__divmodsi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[DIVMOD_ATTRS:#[0-9]+]]
; DARWIN: declare i32 @__udivmodsi4(i32, i32, ptr noalias nonnull writeonly align 4 dereferenceable(4)) [[DIVMOD_ATTRS]]
; DARWIN-NOT: inreg { i32, i32 } @__divmodsi4
; DARWIN: attributes [[DIVMOD_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

define void @f() {
  ret void
}
