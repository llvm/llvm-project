; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-arm-none-eabi < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -enable-machine-outliner -mtriple=aarch64-arm-none-eabi -stop-after=machine-outliner < %s | FileCheck %s -check-prefix=TARGET_FEATURES

; Make sure that we haven't added nouwind.
; TARGET_FEATURES: define internal void @OUTLINED_FUNCTION_0()
; TARGET_FEATURES-SAME: #[[ATTR_NUM:[0-9]+]]
; TARGET_FEATURES: attributes #[[ATTR_NUM]] = { minsize optsize }

define dso_local i32 @_Z5func1i(i32 %x) #0 {
; CHECK-LABEL: _Z5func1i:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    stp x30, x19, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w19, -8
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    orr w8, wzr, #0x1
; CHECK-NEXT:    madd w19, w0, w0, w8
; CHECK-NEXT:    mov w0, #4
; CHECK-NEXT:    bl __cxa_allocate_exception
; CHECK-NEXT:    bl OUTLINED_FUNCTION_0
entry:
  %mul = mul nsw i32 %x, %x
  %add = add nuw nsw i32 %mul, 1
  %exception = tail call i8* @__cxa_allocate_exception(i64 4) #1
  %0 = bitcast i8* %exception to i32*
  store i32 %add, i32* %0, align 16
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #2
  unreachable
}

define dso_local i32 @_Z5func2c(i8 %x) #0 {
; CHECK-LABEL: _Z5func2c:
; CHECK:       // %bb.0: // %entry
; CHECK-NEXT:    stp x30, x19, [sp, #-16]! // 16-byte Folded Spill
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    .cfi_offset w19, -8
; CHECK-NEXT:    .cfi_offset w30, -16
; CHECK-NEXT:    and w8, w0, #0xff
; CHECK-NEXT:    mov w0, #4
; CHECK-NEXT:    orr w9, wzr, #0x1
; CHECK-NEXT:    madd w19, w8, w8, w9
; CHECK-NEXT:    bl __cxa_allocate_exception
; CHECK-NEXT:    bl OUTLINED_FUNCTION_0
entry:
  %conv = zext i8 %x to i32
  %mul = mul nuw nsw i32 %conv, %conv
  %add = add nuw nsw i32 %mul, 1
  %exception = tail call i8* @__cxa_allocate_exception(i64 4) #1
  %0 = bitcast i8* %exception to i32*
  store i32 %add, i32* %0, align 16
  tail call void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null) #2
  unreachable
}

; CHECK-LABEL: OUTLINED_FUNCTION_0:
; CHECK:      .cfi_startproc
; CHECK:        adrp    x1, _ZTIi
; CHECK-NEXT:   mov     x2, xzr
; CHECK-NEXT:   add     x1, x1, :lo12:_ZTIi
; CHECK-NEXT:   str     w19, [x0]
; CHECK-NEXT:   b       __cxa_throw
; CHECK:      .cfi_endproc


@_ZTIi = external dso_local constant i8*
declare dso_local i8* @__cxa_allocate_exception(i64) local_unnamed_addr
declare dso_local void @__cxa_throw(i8*, i8*, i8*) local_unnamed_addr

attributes #0 = { minsize noreturn optsize }
attributes #1 = { nounwind }
attributes #2 = { noreturn }
