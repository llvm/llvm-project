; RUN: llc  < %s -march=mipsel | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips   | FileCheck %s -check-prefix=CHECK-EB

@g1 = global double 0.000000e+00, align 8
@_ZTId = external constant ptr

define void @_Z1fd(double %i2) personality ptr @__gxx_personality_v0 {
entry:
; CHECK-EL:  addiu $sp, $sp
; CHECK-EL:  .cfi_def_cfa_offset
; CHECK-EL:  sdc1 $f20
; CHECK-EL:  sw  $ra
; CHECK-EL:  .cfi_offset 52, -8
; CHECK-EL:  .cfi_offset 53, -4
; CHECK-EB:  .cfi_offset 53, -8
; CHECK-EB:  .cfi_offset 52, -4
; CHECK-EL:  .cfi_offset 31, -12

  %exception = tail call ptr @__cxa_allocate_exception(i32 8) nounwind
  store double 3.200000e+00, ptr %exception, align 8
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTId, ptr null) noreturn
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
; CHECK-EL:  # %lpad
; CHECK-EL:  bne $5

  %exn.val = landingpad { ptr, i32 }
           cleanup
           catch ptr @_ZTId
  %exn = extractvalue { ptr, i32 } %exn.val, 0
  %sel = extractvalue { ptr, i32 } %exn.val, 1
  %0 = tail call i32 @llvm.eh.typeid.for(ptr @_ZTId) nounwind
  %1 = icmp eq i32 %sel, %0
  br i1 %1, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %2 = tail call ptr @__cxa_begin_catch(ptr %exn) nounwind
  %exn.scalar = load double, ptr %2, align 8
  %add = fadd double %exn.scalar, %i2
  store double %add, ptr @g1, align 8
  tail call void @__cxa_end_catch() nounwind
  ret void

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } %exn.val

unreachable:                                      ; preds = %entry
  unreachable
}

declare ptr @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(ptr) nounwind

declare void @__cxa_throw(ptr, ptr, ptr)

declare ptr @__cxa_begin_catch(ptr)

declare void @__cxa_end_catch()
