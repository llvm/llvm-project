; RUN: llc -mtriple=aarch64-none-eabi < %s | FileCheck --check-prefixes CHECK,CHECK-V8A %s
; RUN: llc -mtriple=aarch64-none-eabi -mattr=v8.3a < %s | FileCheck --check-prefixes CHECK,CHECK-V83A %s
; RUN: llc -mtriple=aarch64-none-eabi -filetype=obj -o - <%s | llvm-dwarfdump -v - | FileCheck --check-prefix=CHECK-DUMP %s

@.str = private unnamed_addr constant [15 x i8] c"some exception\00", align 1
@_ZTIPKc = external dso_local constant ptr

; CHECK: @_Z3fooi
; CHECK-V8A: hint #25
; CHECK-V83A: paciasp
; CHECK-NEXT: .cfi_negate_ra_state
; CHECK-NOT: .cfi_negate_ra_state
define dso_local i32 @_Z3fooi(i32 %x) #0 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %exception = call ptr @__cxa_allocate_exception(i64 8) #1
  store ptr @.str, ptr %exception, align 16
  call void @__cxa_throw(ptr %exception, ptr @_ZTIPKc, ptr null) #2
  unreachable

return:                                           ; No predecessors!
  %0 = load i32, ptr %retval, align 4
  ret i32 %0
}

declare dso_local ptr @__cxa_allocate_exception(i64)

declare dso_local void @__cxa_throw(ptr, ptr, ptr)

attributes #0 = { "sign-return-address"="all" }

;CHECK-DUMP: DW_CFA_AARCH64_negate_ra_state
