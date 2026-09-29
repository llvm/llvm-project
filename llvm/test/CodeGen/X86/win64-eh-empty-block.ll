; RUN: llc -mtriple=x86_64-windows-gnu %s -o - | FileCheck %s

; Based on this C++ code:
; struct as {
;     as() { at = static_cast<int *>(operator new(sizeof(int))); }
;     ~as() { operator delete(at); }
;     int *at;
; };
; void am(int) {
;     static as au;
;     as av;
;     throw 0;
; }

; optnone was added to ensure that branch folding and block layout are not
; disturbed. The key thing about this test is that it ends in an empty
; unreachable block, which forces us to scan back across blocks.

; CHECK: _Z2ami:
; CHECK: callq   __cxa_throw
; CHECK: # %eh.resume
; CHECK: callq _Unwind_Resume
; CHECK-NEXT: # %unreachable
; CHECK-NEXT: int3
; CHECK-NEXT: .Lfunc_end0:

%struct.as = type { ptr }

@_ZZ2amiE2au = internal unnamed_addr global %struct.as zeroinitializer, align 8
@_ZGVZ2amiE2au = internal global i64 0, align 8
@_ZTIi = external constant ptr

define dso_local void @_Z2ami(i32 %0) noinline optnone personality ptr @__gxx_personality_seh0 {
entry:
  %1 = load atomic i8, ptr @_ZGVZ2amiE2au acquire, align 8
  %guard.uninitialized = icmp eq i8 %1, 0
  br i1 %guard.uninitialized, label %init.check, label %init.end

init.check:                                       ; preds = %entry
  %2 = tail call i32 @__cxa_guard_acquire(ptr nonnull @_ZGVZ2amiE2au)
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %init.end, label %init

init:                                             ; preds = %init.check
  %call.i3 = invoke ptr @_Znwy(i64 4)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %init
  store ptr %call.i3, ptr @_ZZ2amiE2au, align 8
  %3 = tail call i32 @atexit(ptr nonnull @__dtor__ZZ2amiE2au)
  tail call void @__cxa_guard_release(ptr nonnull @_ZGVZ2amiE2au)
  br label %init.end

init.end:                                         ; preds = %init.check, %invoke.cont, %entry
  %call.i = tail call ptr @_Znwy(i64 4)
  %exception = tail call ptr @__cxa_allocate_exception(i64 4)
  store i32 0, ptr %exception, align 16
  invoke void @__cxa_throw(ptr %exception, ptr @_ZTIi, ptr null)
          to label %unreachable unwind label %lpad1

lpad:                                             ; preds = %init
  %4 = landingpad { ptr, i32 }
          cleanup
  %5 = extractvalue { ptr, i32 } %4, 0
  %6 = extractvalue { ptr, i32 } %4, 1
  tail call void @__cxa_guard_abort(ptr nonnull @_ZGVZ2amiE2au)
  br label %eh.resume

lpad1:                                            ; preds = %init.end
  %7 = landingpad { ptr, i32 }
          cleanup
  %8 = extractvalue { ptr, i32 } %7, 0
  %9 = extractvalue { ptr, i32 } %7, 1
  tail call void @_ZdlPv(ptr %call.i)
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %lpad
  %exn.slot.0 = phi ptr [ %8, %lpad1 ], [ %5, %lpad ]
  %ehselector.slot.0 = phi i32 [ %9, %lpad1 ], [ %6, %lpad ]
  %lpad.val = insertvalue { ptr, i32 } undef, ptr %exn.slot.0, 0
  %lpad.val2 = insertvalue { ptr, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { ptr, i32 } %lpad.val2

unreachable:                                      ; preds = %init.end
  unreachable
}

declare dso_local i32 @__cxa_guard_acquire(ptr)

declare dso_local i32 @__gxx_personality_seh0(...)

declare dso_local void @__dtor__ZZ2amiE2au()

declare dso_local i32 @atexit(ptr)

declare dso_local void @__cxa_guard_abort(ptr)

declare dso_local void @__cxa_guard_release(ptr)

declare dso_local ptr @__cxa_allocate_exception(i64)

declare dso_local void @__cxa_throw(ptr, ptr, ptr)

declare dso_local noalias ptr @_Znwy(i64)

declare dso_local void @_ZdlPv(ptr)
