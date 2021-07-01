; RUN: llc -mtriple=aarch64-none-eabi < %s | FileCheck --check-prefixes CHECK,CHECK-V8A %s
; RUN: llc -mtriple=aarch64-none-eabi -mattr=v8.3a < %s | FileCheck --check-prefixes CHECK,CHECK-V83A %s
; RUN: llc -mtriple=aarch64-none-eabi -filetype=obj -o - <%s | llvm-dwarfdump -v - | FileCheck --check-prefix=CHECK-DUMP %s

@.str = private unnamed_addr constant [15 x i8] c"some exception\00", align 1
@_ZTIPKc = external dso_local constant i8*

; CHECK: @_Z3fooi
; CHECK-V8A: hint #25
; CHECK-V83A: pacia x30, sp
; CHECK-NEXT: .cfi_negate_ra_state
; CHECK-NOT: .cfi_negate_ra_state
define dso_local i32 @_Z3fooi(i32 %x) #0 {
entry:
  %retval = alloca i32, align 4
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  %exception = call i8* @__cxa_allocate_exception(i64 8) #1
  %0 = bitcast i8* %exception to i8**
  store i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i32 0, i32 0), i8** %0, align 16
  call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIPKc to i8*), i8* null) #2
  unreachable

return:                                           ; No predecessors!
  %1 = load i32, i32* %retval, align 4
  ret i32 %1
}

declare dso_local i8* @__cxa_allocate_exception(i64)

declare dso_local void @__cxa_throw(i8*, i8*, i8*)

attributes #0 = { "sign-return-address"="all" }

;CHECK-DUMP: DW_CFA_AARCH64_negate_ra_state
