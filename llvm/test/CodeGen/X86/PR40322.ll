; RUN: llc < %s -mtriple=i686-windows-gnu | FileCheck %s --check-prefix=CHECK-MINGW-X86

%struct.as = type { i32* }

@_ZZ2amiE2au = internal unnamed_addr global %struct.as zeroinitializer, align 4
@_ZGVZ2amiE2au = internal global i64 0, align 8
@_ZTIi = external constant i8*

define void @_Z2ami(i32) #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
; CHECK-MINGW-X86-LABEL: _Z2ami:
; CHECK-MINGW-X86:       # %bb.0: # %entry
; CHECK-MINGW-X86-NEXT:    pushl %edi
; CHECK-MINGW-X86-NEXT:    .cfi_def_cfa_offset 8
; CHECK-MINGW-X86-NEXT:    pushl %esi
; CHECK-MINGW-X86-NEXT:    .cfi_def_cfa_offset 12
; CHECK-MINGW-X86-NEXT:    .cfi_offset %esi, -12
; CHECK-MINGW-X86-NEXT:    .cfi_offset %edi, -8
; CHECK-MINGW-X86-NEXT:    movzbl __ZGVZ2amiE2au, %eax
; CHECK-MINGW-X86-NEXT:    testb %al, %al
; CHECK-MINGW-X86-NEXT:    jne LBB0_4
; CHECK-MINGW-X86-NEXT:  # %bb.1: # %init.check
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl $__ZGVZ2amiE2au
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll ___cxa_guard_acquire
; CHECK-MINGW-X86-NEXT:    addl $4, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -4
; CHECK-MINGW-X86-NEXT:    testl %eax, %eax
; CHECK-MINGW-X86-NEXT:    je LBB0_4
; CHECK-MINGW-X86-NEXT:  # %bb.2: # %init
; CHECK-MINGW-X86-NEXT:  Ltmp0:
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl $4
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll __Znwj
; CHECK-MINGW-X86-NEXT:    addl $4, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -4
; CHECK-MINGW-X86-NEXT:  Ltmp1:
; CHECK-MINGW-X86-NEXT:  # %bb.3: # %invoke.cont
; CHECK-MINGW-X86-NEXT:    movl %eax, __ZZ2amiE2au
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl $__ZGVZ2amiE2au
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll ___cxa_guard_release
; CHECK-MINGW-X86-NEXT:    addl $4, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -4
; CHECK-MINGW-X86-NEXT:  LBB0_4: # %init.end
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl $4
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll __Znwj
; CHECK-MINGW-X86-NEXT:    addl $4, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -4
; CHECK-MINGW-X86-NEXT:    movl %eax, %esi
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl $4
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll ___cxa_allocate_exception
; CHECK-MINGW-X86-NEXT:    addl $4, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -4
; CHECK-MINGW-X86-NEXT:    movl $0, (%eax)
; CHECK-MINGW-X86-NEXT:  Ltmp3:
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x0c
; CHECK-MINGW-X86-NEXT:    movl .refptr.__ZTIi, %ecx
; CHECK-MINGW-X86-NEXT:    pushl $0
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    pushl %ecx
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    pushl %eax
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll ___cxa_throw
; CHECK-MINGW-X86-NEXT:    addl $12, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -12
; CHECK-MINGW-X86-NEXT:  Ltmp4:
; CHECK-MINGW-X86-NEXT:  # %bb.8: # %unreachable
; CHECK-MINGW-X86-NEXT:  LBB0_5: # %lpad
; CHECK-MINGW-X86-NEXT:  Ltmp2:
; CHECK-MINGW-X86-NEXT:    movl %eax, %edi
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl $__ZGVZ2amiE2au
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll ___cxa_guard_abort
; CHECK-MINGW-X86-NEXT:    jmp LBB0_7
; CHECK-MINGW-X86-NEXT:  LBB0_6: # %lpad1
; CHECK-MINGW-X86-NEXT:    .cfi_def_cfa_offset 12
; CHECK-MINGW-X86-NEXT:  Ltmp5:
; CHECK-MINGW-X86-NEXT:    movl %eax, %edi
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl %esi
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll __ZdlPv
; CHECK-MINGW-X86-NEXT:  LBB0_7: # %eh.resume
; CHECK-MINGW-X86-NEXT:    addl $4, %esp
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset -4
; CHECK-MINGW-X86-NEXT:    .cfi_escape 0x2e, 0x04
; CHECK-MINGW-X86-NEXT:    pushl %edi
; CHECK-MINGW-X86-NEXT:    .cfi_adjust_cfa_offset 4
; CHECK-MINGW-X86-NEXT:    calll __Unwind_Resume
; CHECK-MINGW-X86-NEXT:  Lfunc_end0:
entry:
  %1 = load atomic i8, i8* bitcast (i64* @_ZGVZ2amiE2au to i8*) acquire, align 8
  %guard.uninitialized = icmp eq i8 %1, 0
  br i1 %guard.uninitialized, label %init.check, label %init.end

init.check:                                       ; preds = %entry
  %2 = tail call i32 @__cxa_guard_acquire(i64* nonnull @_ZGVZ2amiE2au)
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %init.end, label %init

init:                                             ; preds = %init.check
  %call.i3 = invoke i8* @_Znwj(i32 4)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %init
  store i8* %call.i3, i8** bitcast (%struct.as* @_ZZ2amiE2au to i8**), align 4
  tail call void @__cxa_guard_release(i64* nonnull @_ZGVZ2amiE2au)
  br label %init.end

init.end:                                         ; preds = %init.check, %invoke.cont, %entry
  %call.i = tail call i8* @_Znwj(i32 4)
  %exception = tail call i8* @__cxa_allocate_exception(i32 4)
  %3 = bitcast i8* %exception to i32*
  store i32 0, i32* %3, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
          to label %unreachable unwind label %lpad1

lpad:                                             ; preds = %init
  %4 = landingpad { i8*, i32 }
          cleanup
  %5 = extractvalue { i8*, i32 } %4, 0
  %6 = extractvalue { i8*, i32 } %4, 1
  tail call void @__cxa_guard_abort(i64* nonnull @_ZGVZ2amiE2au) #1
  br label %eh.resume

lpad1:                                            ; preds = %init.end
  %7 = landingpad { i8*, i32 }
          cleanup
  %8 = extractvalue { i8*, i32 } %7, 0
  %9 = extractvalue { i8*, i32 } %7, 1
  tail call void @_ZdlPv(i8* nonnull %call.i)
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %lpad
  %exn.slot.0 = phi i8* [ %8, %lpad1 ], [ %5, %lpad ]
  %ehselector.slot.0 = phi i32 [ %9, %lpad1 ], [ %6, %lpad ]
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val2

unreachable:                                      ; preds = %init.end
  unreachable
}

declare i32 @__cxa_guard_acquire(i64*)
declare i32 @__gxx_personality_v0(...)
declare void @__cxa_guard_abort(i64*)
declare void @__cxa_guard_release(i64*)
declare i8* @__cxa_allocate_exception(i32)
declare void @__cxa_throw(i8*, i8*, i8*)
declare noalias nonnull i8* @_Znwj(i32)
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare void @_ZdlPv(i8*)
