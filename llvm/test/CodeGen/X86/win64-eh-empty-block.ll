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
; CHECK-NEXT: int3
; CHECK-NEXT: # %unreachable
; CHECK-NEXT: .Lfunc_end0:

%struct.as = type { i32* }

@_ZZ2amiE2au = internal unnamed_addr global %struct.as zeroinitializer, align 8
@_ZGVZ2amiE2au = internal global i64 0, align 8
@_ZTIi = external constant i8*

define dso_local void @_Z2ami(i32 %0) noinline optnone personality i8* bitcast (i32 (...)* @__gxx_personality_seh0 to i8*) {
entry:
  %1 = load atomic i8, i8* bitcast (i64* @_ZGVZ2amiE2au to i8*) acquire, align 8
  %guard.uninitialized = icmp eq i8 %1, 0
  br i1 %guard.uninitialized, label %init.check, label %init.end

init.check:                                       ; preds = %entry
  %2 = tail call i32 @__cxa_guard_acquire(i64* nonnull @_ZGVZ2amiE2au)
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %init.end, label %init

init:                                             ; preds = %init.check
  %call.i3 = invoke i8* @_Znwy(i64 4)
          to label %invoke.cont unwind label %lpad

invoke.cont:                                      ; preds = %init
  store i8* %call.i3, i8** bitcast (%struct.as* @_ZZ2amiE2au to i8**), align 8
  %3 = tail call i32 @atexit(void ()* nonnull @__dtor__ZZ2amiE2au)
  tail call void @__cxa_guard_release(i64* nonnull @_ZGVZ2amiE2au)
  br label %init.end

init.end:                                         ; preds = %init.check, %invoke.cont, %entry
  %call.i = tail call i8* @_Znwy(i64 4)
  %exception = tail call i8* @__cxa_allocate_exception(i64 4)
  %4 = bitcast i8* %exception to i32*
  store i32 0, i32* %4, align 16
  invoke void @__cxa_throw(i8* %exception, i8* bitcast (i8** @_ZTIi to i8*), i8* null)
          to label %unreachable unwind label %lpad1

lpad:                                             ; preds = %init
  %5 = landingpad { i8*, i32 }
          cleanup
  %6 = extractvalue { i8*, i32 } %5, 0
  %7 = extractvalue { i8*, i32 } %5, 1
  tail call void @__cxa_guard_abort(i64* nonnull @_ZGVZ2amiE2au)
  br label %eh.resume

lpad1:                                            ; preds = %init.end
  %8 = landingpad { i8*, i32 }
          cleanup
  %9 = extractvalue { i8*, i32 } %8, 0
  %10 = extractvalue { i8*, i32 } %8, 1
  tail call void @_ZdlPv(i8* %call.i)
  br label %eh.resume

eh.resume:                                        ; preds = %lpad1, %lpad
  %exn.slot.0 = phi i8* [ %9, %lpad1 ], [ %6, %lpad ]
  %ehselector.slot.0 = phi i32 [ %10, %lpad1 ], [ %7, %lpad ]
  %lpad.val = insertvalue { i8*, i32 } undef, i8* %exn.slot.0, 0
  %lpad.val2 = insertvalue { i8*, i32 } %lpad.val, i32 %ehselector.slot.0, 1
  resume { i8*, i32 } %lpad.val2

unreachable:                                      ; preds = %init.end
  unreachable
}

declare dso_local i32 @__cxa_guard_acquire(i64*)

declare dso_local i32 @__gxx_personality_seh0(...)

declare dso_local void @__dtor__ZZ2amiE2au()

declare dso_local i32 @atexit(void ()*)

declare dso_local void @__cxa_guard_abort(i64*)

declare dso_local void @__cxa_guard_release(i64*)

declare dso_local i8* @__cxa_allocate_exception(i64)

declare dso_local void @__cxa_throw(i8*, i8*, i8*)

declare dso_local noalias i8* @_Znwy(i64)

declare dso_local void @_ZdlPv(i8*)
