; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s
; RUN: llc -enable-new-pm < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s
; Control Flow Guard is currently only available on Windows

; Test that longjmp targets have public labels and are included in the .gljmp section.
%struct._SETJMP_FLOAT128 = type { [2 x i64] }
@buf1 = internal global [16 x %struct._SETJMP_FLOAT128] zeroinitializer, align 16

define i32 @func_cf_setjmp() {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 -1, ptr %2, align 4
  %3 = call ptr @llvm.frameaddress(i32 0)
  %4 = call i32 @_setjmp(ptr @buf1, ptr %3) #0

  ; CHECK-LABEL: func_cf_setjmp
  ; CHECK:       callq _setjmp
  ; CHECK-NEXT:  $cfgsj_func_cf_setjmp0:

  %5 = call ptr @llvm.frameaddress(i32 0)
  %6 = call i32 @_setjmp(ptr @buf1, ptr %5) #0

  ; CHECK:       callq _setjmp
  ; CHECK-NEXT:  $cfgsj_func_cf_setjmp1:

  store i32 1, ptr %2, align 4
  %7 = load i32, ptr %2, align 4
  ret i32 %7

  ; CHECK:       .section .gljmp$y,"dr"
  ; CHECK-NEXT:  .symidx $cfgsj_func_cf_setjmp0
  ; CHECK-NEXT:  .symidx $cfgsj_func_cf_setjmp1
}

declare ptr @llvm.frameaddress(i32)

; Function Attrs: returns_twice
declare dso_local i32 @_setjmp(ptr, ptr) #0

attributes #0 = { returns_twice }

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
