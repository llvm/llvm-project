; RUN: llc < %s -mtriple=arm-pc-windows-msvc | FileCheck %s -check-prefixes=CHECK,MSVC
; RUN: llc < %s -mtriple=arm-w64-windows-gnu | FileCheck %s -check-prefixes=CHECK,MINGW
; Control Flow Guard is currently only available on Windows

; Test that Control Flow Guard checks are correctly added when required.


declare i32 @target_func()


; Test that Control Flow Guard checks are not added on calls with the "guard_nocf" attribute.
define i32 @func_guard_nocf() #0 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8
  %1 = call arm_aapcs_vfpcc i32 %0() #1
  ret i32 %1

  ; CHECK-LABEL: func_guard_nocf
  ; CHECK:       movw r0, :lower16:target_func
	; CHECK:       movt r0, :upper16:target_func
  ; CHECK-NOT:   __guard_check_icall_fptr
	; CHECK:       blx r0
}
attributes #0 = { "target-cpu"="cortex-a9" "target-features"="+armv7-a,+dsp,+fp16,+neon,+strict-align,+thumb-mode,+vfp3"}
attributes #1 = { "guard_nocf" }


; Test that Control Flow Guard checks are added even at -O0.
define i32 @func_optnone_cf() #2 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8
  %1 = call i32 %0()
  ret i32 %1

  ; The call to __guard_check_icall_fptr should come immediately before the call to the target function.
  ; CHECK-LABEL: func_optnone_cf
	; CHECK:       movw r0, :lower16:target_func
	; CHECK:       movt r0, :upper16:target_func
	; CHECK:       str r0, [sp]
	; CHECK:       ldr r4, [sp]
	; CHECK:       movw r0, :lower16:__guard_check_icall_fptr
	; CHECK:       movt r0, :upper16:__guard_check_icall_fptr
	; CHECK:       ldr r1, [r0]
	; CHECK:       mov r0, r4
	; CHECK:       blx r1
	; CHECK-NEXT:  blx r4
}
attributes #2 = { noinline optnone "target-cpu"="cortex-a9" "target-features"="+armv7-a,+dsp,+fp16,+neon,+strict-align,+thumb-mode,+vfp3"}


; Test that Control Flow Guard checks are correctly added in optimized code (common case).
define i32 @func_cf() #0 {
entry:
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %0 = load ptr, ptr %func_ptr, align 8
  %1 = call i32 %0()
  ret i32 %1

  ; The call to __guard_check_icall_fptr should come immediately before the call to the target function.
  ; CHECK-LABEL: func_cf
  ; CHECK:       movw r0, :lower16:__guard_check_icall_fptr
	; CHECK:       movt r0, :upper16:__guard_check_icall_fptr
	; CHECK:       ldr r1, [r0]
  ; CHECK:       movw r4, :lower16:target_func
	; CHECK:       movt r4, :upper16:target_func
	; CHECK:       mov r0, r4
	; CHECK:       blx r1
	; CHECK-NEXT:  blx r4
}


; Test that Control Flow Guard checks are correctly added on invoke instructions.
define i32 @func_cf_invoke() #0 personality ptr @h {
entry:
  %0 = alloca i32, align 4
  %func_ptr = alloca ptr, align 8
  store ptr @target_func, ptr %func_ptr, align 8
  %1 = load ptr, ptr %func_ptr, align 8
  %2 = invoke i32 %1()
          to label %invoke.cont unwind label %lpad
invoke.cont:                                      ; preds = %entry
  ret i32 %2

lpad:                                             ; preds = %entry
  %tmp = landingpad { ptr, i32 }
          catch ptr null
  ret i32 -1

  ; The call to __guard_check_icall_fptr should come immediately before the call to the target function.
  ; CHECK-LABEL: func_cf_invoke
  ; CHECK:       movw r0, :lower16:__guard_check_icall_fptr
	; CHECK:       movt r0, :upper16:__guard_check_icall_fptr
	; CHECK:       ldr r1, [r0]
  ; CHECK:       movw r4, :lower16:target_func
	; CHECK:       movt r4, :upper16:target_func
	; CHECK:       mov r0, r4
	; CHECK:       blx r1
  ; MSVC-NEXT:   $Mtmp0:
  ; MINGW-NEXT:  .Ltmp0:
	; CHECK-NEXT:  blx r4
  ; CHECK:       @ %common.ret
  ; CHECK:       @ %lpad
}

declare void @h()


; Test that longjmp targets have public labels and are included in the .gljmp section.
%struct._SETJMP_FLOAT128 = type { [2 x i64] }
@buf1 = internal global [16 x %struct._SETJMP_FLOAT128] zeroinitializer, align 16

define i32 @func_cf_setjmp() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, ptr %1, align 4
  store i32 -1, ptr %2, align 4
  %3 = call ptr @llvm.frameaddress(i32 0)
  %4 = call i32 @_setjmp(ptr @buf1, ptr %3) #3

  ; CHECK-LABEL: func_cf_setjmp
  ; CHECK:       bl _setjmp
  ; CHECK-NEXT:  $cfgsj_func_cf_setjmp0:

  %5 = call ptr @llvm.frameaddress(i32 0)
  %6 = call i32 @_setjmp(ptr @buf1, ptr %5) #3

  ; CHECK:       bl _setjmp
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
declare dso_local i32 @_setjmp(ptr, ptr) #3

attributes #3 = { returns_twice }


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
