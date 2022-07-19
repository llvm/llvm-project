; RUN: llc < %s -mtriple=i686-windows-gnu | FileCheck %s --check-prefix=CHECK-MINGW-X86

%struct.as = type { ptr }

@_ZZ2amiE2au = internal unnamed_addr global %struct.as zeroinitializer, align 4
@_ZGVZ2amiE2au = internal global i64 0, align 8
@_ZTIi = external constant ptr

define void @_Z2ami(i32) #0 personality ptr @__gxx_personality_v0 {
; CHECK-MINGW-X86-LABEL: _Z2ami:
; CHECK-MINGW-X86:       # %bb.0: # %entry
; CHECK-MINGW-X86-NEXT:    pushl %edi
; CHECK-MINGW-X86-NEXT:    .cfi_def_cfa_offset 8
; CHECK-MINGW-X86-NEXT:    pushl %esi
; CHECK-MINGW-X86-NEXT:    .cfi_def_cfa_offset 12
; CHECK-MINGW-X86-NEXT:    .cfi_offset %esi, -12
; CHECK-MINGW-X86-NEXT:    .cfi_offset %edi, -8
; CHECK-MINGW-X86-NEXT:    movb __ZGVZ2amiE2au, %al
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
  %1 = load atomic i8, ptr @_ZGVZ2amiE2au acquire, align 8
  %guard.uninitialized = icmp eq i8 %1, 0
  br i1 %guard.uninitialized, label %init.check, label %init.end

init.check:                                       ; preds = %entry
  %2 = tail call i32 @__cxa_guard_acquire(ptr nonnull @_ZGVZ2amiE2au)
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %init.end, label %init

init:                                             ; preds = %init.check
  %call.i3 = invoke ptr @_Znwj(i32 4)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %init
  store ptr %call.i3, ptr @_ZZ2amiE2au, align 4
  tail call void @__cxa_guard_release(ptr nonnull @_ZGVZ2amiE2au)
  br label %init.end

init.end:                                         ; preds = %init.check, %invoke.cont, %entry
  %call.i = tail call ptr @_Znwj(i32 4)
  %exception = tail call ptr @__cxa_allocate_exception(i32 4)
  store i32 0, ptr %exception, align 16
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
          to label %unreachable unwind label %lpad1

lpad:                                             ; preds = %init
  %3 = landingpad { ptr, i32 }
          cleanup
  %4 = extractvalue { ptr, i32 } %3, 0
  %5 = extractvalue { ptr, i32 } %3, 1
  tail call void @__cxa_guard_abort(ptr nonnull @_ZGVZ2amiE2au) #1
  br label %eh.resume

lpad1:                                            ; preds = %init.end
  %6 = landingpad { ptr, i32 }
          cleanup
  %7 = extractvalue { ptr, i32 } %6, 0
  %8 = extractvalue { ptr, i32 } %6, 1
  tail call void @_ZdlPv(ptr nonnull %call.i)
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %lpad
  %exn.slot.0 = phi ptr [ %7, %lpad1 ], [ %4, %lpad ]
  %ehselector.slot.0 = phi i32 [ %8, %lpad1 ], [ %5, %lpad ]
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val2

unreachable:                                      ; preds = %init.end
  unreachable
}

declare i32 @__cxa_guard_acquire(ptr)
declare i32 @__gxx_personality_v0(...)
declare void @__cxa_guard_abort(ptr)
declare void @__cxa_guard_release(ptr)
declare ptr @__cxa_allocate_exception(i32)
declare void @__cxa_throw(ptr, ptr, ptr)
declare noalias nonnull ptr @_Znwj(i32)
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare void @_ZdlPv(ptr)
