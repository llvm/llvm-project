; RUN: llc < %s -mtriple=xcore | FileCheck %s

declare void @g()
declare i32 @__gxx_personality_v0(...)
declare i32 @llvm.eh.typeid.for(ptr) nounwind readnone
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare ptr @__cxa_allocate_exception(i32)
declare void @__cxa_throw(ptr, ptr, ptr)

@_ZTIi = external constant ptr
@_ZTId = external constant ptr

; CHECK-LABEL: fn_typeid:
; CHECK: .cfi_startproc
; CHECK: mkmsk r0, 1
; CHECK: retsp 0
; CHECK: .cfi_endproc
define i32 @fn_typeid() {
entry:
  %0 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi) nounwind
  ret i32 %0
}

; CHECK-LABEL: fn_throw
; CHECK: .cfi_startproc
; CHECK: entsp 1
; CHECK: .cfi_def_cfa_offset 4
; CHECK: .cfi_offset 15, 0
; CHECK: ldc r0, 4
; CHECK: bl __cxa_allocate_exception
; CHECK: ldaw r1, dp[_ZTIi]
; CHECK: ldc r2, 0
; CHECK: bl __cxa_throw
define void @fn_throw() {
entry:
  %0 = call ptr @__cxa_allocate_exception(i32 4) nounwind
  call void @__cxa_throw(ptr %0, ptr @_ZTIi, ptr null) noreturn
  unreachable
}

; CHECK-LABEL: fn_catch:
; CHECK-NEXT: [[START:.L[a-zA-Z0-9_]+]]
; CHECK: .cfi_startproc
; CHECK: .cfi_personality 0, __gxx_personality_v0
; CHECK: .cfi_lsda 0, [[LSDA:.L[a-zA-Z0-9_]+]]
; CHECK: entsp 4
; CHECK: .cfi_def_cfa_offset 16
; CHECK: .cfi_offset 15, 0
define void @fn_catch() personality ptr @__gxx_personality_v0 {
entry:

; N.B. we alloc no variables, hence force compiler to spill
; CHECK: stw r4, sp[3]
; CHECK: .cfi_offset 4, -4
; CHECK: stw r5, sp[2]
; CHECK: .cfi_offset 5, -8
; CHECK: stw r6, sp[1]
; CHECK: .cfi_offset 6, -12
; CHECK: [[PRE_G:.L[a-zA-Z0-9_]+]]
; CHECK: bl g
; CHECK: [[POST_G:.L[a-zA-Z0-9_]+]]
; CHECK: [[RETURN:.L[a-zA-Z0-9_]+]]
; CHECK: ldw r6, sp[1]
; CHECK: ldw r5, sp[2]
; CHECK: ldw r4, sp[3]
; CHECK: retsp 4
  invoke void @g() to label %cont unwind label %lpad
cont:
  ret void

; CHECK: {{.L[a-zA-Z0-9_]+}}
; CHECK: [[LANDING:.L[a-zA-Z0-9_]+]]
; CHECK: mov r5, r1
; CHECK: mov r4, r0
; CHECK: bl __cxa_begin_catch
; CHECK: ldw r6, r0[0]
; CHECK: bl __cxa_end_catch
lpad:
  %0 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
          catch ptr @_ZTId
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = extractvalue { ptr, i32 } %0, 1
  %3 = call ptr @__cxa_begin_catch(ptr %1) nounwind
  %4 = load i32, ptr %3
  call void @__cxa_end_catch() nounwind

; CHECK: eq r0, r6, r5
; CHECK: bf r0, [[RETURN]]
; CHECK: mov r0, r4
; CHECK: bl _Unwind_Resume
; CHECK: [[END:.L[a-zA-Z0-9_]+]]
; CHECK: .cfi_endproc
  %5 = icmp eq i32 %4, %2
  br i1 %5, label %Resume, label %Exit
Resume:
  resume { ptr, i32 } %0
Exit:
  ret void
}

; CHECK: [[LSDA]]:
; CHECK: .byte  255
; CHECK: .byte  0
; CHECK: .uleb128 [[TTBASE:.Lttbase[0-9]+]]-[[TTBASEREF:.Lttbaseref[0-9]+]]
; CHECK: [[TTBASEREF]]:
; CHECK: .byte  1
; CHECK: .uleb128 [[CST_END:.Lcst_end[0-9]+]]-[[CST_BEGIN:.Lcst_begin[0-9]+]]
; CHECK: [[CST_BEGIN]]:
; CHECK: .uleb128 [[PRE_G]]-[[START]]
; CHECK: .uleb128 [[POST_G]]-[[PRE_G]]
; CHECK: .uleb128 [[LANDING]]-[[START]]
; CHECK: .byte 5
; CHECK: .uleb128 [[POST_G]]-[[START]]
; CHECK: .uleb128 [[END]]-[[POST_G]]
; CHECK: .byte 0
; CHECK: .byte 0
; CHECK: [[CST_END]]:
; CHECK: .byte 0
; CHECK: .byte 0
; CHECK: .byte 1
; CHECK: .byte 125
; CHECK: .byte 2
; CHECK: .byte 125
; CHECK: .p2align 2
; CHECK: .long _ZTIi
; CHECK: .long _ZTId
; CHECK: [[TTBASE]]:
