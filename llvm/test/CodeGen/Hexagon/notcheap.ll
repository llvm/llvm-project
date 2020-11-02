; RUN: llc  -march=hexagon < %s | FileCheck %s
; Check if only one transfer immediate instruction is generated for init.end block.
; Since the transfer immediate of address operand is declared as not cheap, it
; should generate only one transfer immediate, rather than two of them.

; CHECK-LABEL: .LBB{{.*}}
; CHECK: r0 = ##_ZZ3foovE1x
; CHECK-NOT: r{{[1-9]*}} = ##_ZZ3foovE1x
; CHECK:  memw(r0+#0) += #1
; CHECK: r{{.*}} = dealloc_return

%struct.FooBaz = type { i32 }
@_ZZ3foovE1x = internal global %struct.FooBaz zeroinitializer, align 4
@_ZGVZ3foovE1x = internal global i64 0, section ".bss._ZGVZ3foovE1x", align 8
@__dso_handle = external dso_local global i8

define dso_local i32* @_Z3foov() local_unnamed_addr optsize {
entry:
  %0 = load atomic i8, i8* bitcast (i64* @_ZGVZ3foovE1x to i8*) acquire, align 8
  %guard.uninitialized = icmp eq i8 %0, 0
  br i1 %guard.uninitialized, label %init.check, label %init.end

init.check:                                       ; preds = %entry
  %1 = tail call i32 @__cxa_guard_acquire(i64* nonnull @_ZGVZ3foovE1x)
  %tobool = icmp eq i32 %1, 0
  br i1 %tobool, label %init.end, label %init

init:                                             ; preds = %init.check
  tail call void @_ZN6FooBazC1Ev(%struct.FooBaz* nonnull @_ZZ3foovE1x)
  %2 = tail call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.FooBaz*)* @_ZN6FooBazD1Ev to void (i8*)*), i8* bitcast (%struct.FooBaz* @_ZZ3foovE1x to i8*), i8* nonnull @__dso_handle)
  tail call void @__cxa_guard_release(i64* nonnull @_ZGVZ3foovE1x)
  br label %init.end

init.end:                                         ; preds = %init, %init.check, %entry
  %3 = load i32, i32* getelementptr inbounds (%struct.FooBaz, %struct.FooBaz* @_ZZ3foovE1x, i32 0, i32 0), align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, i32* getelementptr inbounds (%struct.FooBaz, %struct.FooBaz* @_ZZ3foovE1x, i32 0, i32 0), align 4
  ret i32* getelementptr inbounds (%struct.FooBaz, %struct.FooBaz* @_ZZ3foovE1x, i32 0, i32 0)
}

declare dso_local i32 @__cxa_guard_acquire(i64*) local_unnamed_addr
declare dso_local void @_ZN6FooBazC1Ev(%struct.FooBaz*) unnamed_addr
declare dso_local void @_ZN6FooBazD1Ev(%struct.FooBaz*) unnamed_addr
declare dso_local i32 @__cxa_atexit(void (i8*)*, i8*, i8*) local_unnamed_addr
declare dso_local void @__cxa_guard_release(i64*) local_unnamed_addr
