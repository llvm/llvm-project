; RUN: llc  < %s -march=mipsel | FileCheck %s -check-prefix=CHECK-EL
; RUN: llc  < %s -march=mips   | FileCheck %s -check-prefix=CHECK-EB

@g1 = global double 0.000000e+00, align 8
@_ZTId = external constant i8*

define void @_Z1fd(double %i2) personality i32 (...)* @__gxx_personality_v0 {
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

  %exception = tail call i8* @__cxa_allocate_exception(i32 8) nounwind
  %0 = bitcast i8* %exception to double*
  store double 3.200000e+00, double* %0, align 8
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTId to i8*), i8* null) noreturn
          to label %unreachable unwind label %lpad

lpad:                                             ; preds = %entry
; CHECK-EL:  # %lpad
; CHECK-EL:  beq $5

  %exn.val = landingpad { i8*, i32 }
           cleanup
           catch i8* bitcast (i8** @_ZTId to i8*)
  %exn = extractvalue { i8*, i32 } %exn.val, 0
  %sel = extractvalue { i8*, i32 } %exn.val, 1
  %1 = tail call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTId to i8*)) nounwind
  %2 = icmp eq i32 %sel, %1
  br i1 %2, label %catch, label %eh.resume

catch:                                            ; preds = %lpad
  %3 = tail call i8* @__cxa_begin_catch(i8* %exn) nounwind
  %4 = bitcast i8* %3 to double*
  %exn.scalar = load double, double* %4, align 8
  %add = fadd double %exn.scalar, %i2
  store double %add, double* @g1, align 8
  tail call void @__cxa_end_catch() nounwind
  ret void

eh.resume:                                        ; preds = %lpad
  resume { i8*, i32 } %exn.val

unreachable:                                      ; preds = %entry
  unreachable
}

declare i8* @__cxa_allocate_exception(i32)

declare i32 @__gxx_personality_v0(...)

declare i32 @llvm.eh.typeid.for(i8*) nounwind

declare void @__cxa_throw(i8*, i8*, i8*)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()
