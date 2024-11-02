; RUN: llc < %s -mtriple=aarch64-pc-windows-msvc | FileCheck %s
; RUN: llc < %s -mtriple=aarch64-w64-windows-gnu | FileCheck %s
; Control Flow Guard is currently only available on Windows

; Test that Control Flow Guard checks are correctly added when required.


declare i32 @target_func()


; Test that Control Flow Guard checks are not added on calls with the "guard_nocf" attribute.
define i32 @func_guard_nocf() {
entry:
  %func_ptr = alloca i32 ()*, align 8
  store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  %1 = call i32 %0() #0
  ret i32 %1

  ; CHECK-LABEL: func_guard_nocf
  ; CHECK:       adrp x8, target_func
	; CHECK:       add x8, x8, :lo12:target_func
  ; CHECK-NOT:   __guard_check_icall_fptr
	; CHECK:       blr x8
}
attributes #0 = { "guard_nocf" }


; Test that Control Flow Guard checks are added even at -O0.
define i32 @func_optnone_cf() #1 {
entry:
  %func_ptr = alloca i32 ()*, align 8
  store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  %1 = call i32 %0()
  ret i32 %1

  ; The call to __guard_check_icall_fptr should come immediately before the call to the target function.
  ; CHECK-LABEL: func_optnone_cf
	; CHECK:        adrp x8, target_func
	; CHECK:        add x8, x8, :lo12:target_func
	; CHECK:        adrp x9, __guard_check_icall_fptr
	; CHECK:        add x9, x9, :lo12:__guard_check_icall_fptr
	; CHECK:        ldr x9, [x9]
	; CHECK:        mov x15, x8
	; CHECK:        blr x9
	; CHECK-NEXT:   blr x8
}
attributes #1 = { noinline optnone }


; Test that Control Flow Guard checks are correctly added in optimized code (common case).
define i32 @func_cf() {
entry:
  %func_ptr = alloca i32 ()*, align 8
  store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  %1 = call i32 %0()
  ret i32 %1

  ; The call to __guard_check_icall_fptr should come immediately before the call to the target function.
  ; CHECK-LABEL: func_cf
  ; CHECK:        adrp x8, __guard_check_icall_fptr
	; CHECK:        ldr x9, [x8, :lo12:__guard_check_icall_fptr]
	; CHECK:        adrp x8, target_func
	; CHECK:        add x8, x8, :lo12:target_func
	; CHECK:        mov x15, x8
	; CHECK: 	     	blr x9
	; CHECK-NEXT:   blr x8
}


; Test that Control Flow Guard checks are correctly added on invoke instructions.
define i32 @func_cf_invoke() personality i8* bitcast (void ()* @h to i8*) {
entry:
  %0 = alloca i32, align 4
  %func_ptr = alloca i32 ()*, align 8
  store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  %1 = load i32 ()*, i32 ()** %func_ptr, align 8
  %2 = invoke i32 %1()
          to label %invoke.cont unwind label %lpad
invoke.cont:                                      ; preds = %entry
  ret i32 %2

lpad:                                             ; preds = %entry
  %tmp = landingpad { i8*, i32 }
          catch i8* null
  ret i32 -1

  ; The call to __guard_check_icall_fptr should come immediately before the call to the target function.
  ; CHECK-LABEL: func_cf_invoke
  ; CHECK:        adrp x8, __guard_check_icall_fptr
	; CHECK:        ldr x9, [x8, :lo12:__guard_check_icall_fptr]
	; CHECK:        adrp x8, target_func
	; CHECK:        add x8, x8, :lo12:target_func
	; CHECK:        mov x15, x8
	; CHECK:        blr x9
  ; CHECK-NEXT:   .Ltmp0:
	; CHECK-NEXT:   blr x8
  ; CHECK:       // %common.ret
  ; CHECK:       // %lpad
}

declare void @h()


; Test that longjmp targets have public labels and are included in the .gljmp section.
%struct._SETJMP_FLOAT128 = type { [2 x i64] }
@buf1 = internal global [16 x %struct._SETJMP_FLOAT128] zeroinitializer, align 16

define i32 @func_cf_setjmp() {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 -1, i32* %2, align 4
  %3 = call i8* @llvm.frameaddress(i32 0)
  %4 = call i32 @_setjmp(i8* bitcast ([16 x %struct._SETJMP_FLOAT128]* @buf1 to i8*), i8* %3) #2

  ; CHECK-LABEL: func_cf_setjmp
  ; CHECK:       bl _setjmp
  ; CHECK-NEXT:  $cfgsj_func_cf_setjmp0:

  %5 = call i8* @llvm.frameaddress(i32 0)
  %6 = call i32 @_setjmp(i8* bitcast ([16 x %struct._SETJMP_FLOAT128]* @buf1 to i8*), i8* %5) #3

  ; CHECK:       bl _setjmp
  ; CHECK-NEXT:  $cfgsj_func_cf_setjmp1:

  store i32 1, i32* %2, align 4
  %7 = load i32, i32* %2, align 4
  ret i32 %7

  ; CHECK:       .section .gljmp$y,"dr"
  ; CHECK-NEXT:  .symidx $cfgsj_func_cf_setjmp0
  ; CHECK-NEXT:  .symidx $cfgsj_func_cf_setjmp1
}

declare i8* @llvm.frameaddress(i32)

; Function Attrs: returns_twice
declare dso_local i32 @_setjmp(i8*, i8*) #2

attributes #2 = { returns_twice }
attributes #3 = { returns_twice }


!llvm.module.flags = !{!0}
!0 = !{i32 2, !"cfguard", i32 2}
