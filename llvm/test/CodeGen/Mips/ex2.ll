; RUN: llc  -march=mipsel -mattr=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16

@.str = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@_ZTIPKc = external constant ptr

define i32 @main() {
; 16-LABEL: main:
; 16: 	.cfi_startproc
; 16: 	save	$16, $17, $ra, 32 # 16 bit inst
; 16:   .cfi_def_cfa_offset 32
; 16: 	.cfi_offset 31, -4
; 16: 	.cfi_offset 17, -8
; 16:   .cfi_offset 16, -12
; 16:   .cfi_endproc
entry:
  %retval = alloca i32, align 4
  store i32 0, ptr %retval
  %exception = call ptr @__cxa_allocate_exception(i32 4) nounwind
  store ptr @.str, ptr %exception
  call void @__cxa_throw(ptr %exception, ptr @_ZTIPKc, ptr null) noreturn
  unreachable

return:                                           ; No predecessors!
  %0 = load i32, ptr %retval
  ret i32 %0
}

declare ptr @__cxa_allocate_exception(i32)

declare void @__cxa_throw(ptr, ptr, ptr)
