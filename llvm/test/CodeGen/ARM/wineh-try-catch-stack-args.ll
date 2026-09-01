; RUN: llc -mtriple thumbv7-windows-msvc -o - %s
; XFAIL: *

; FIXME: C++ EH is not supported on thumbv7-windows-msvc yet.

; FIXME: Windows SEH for armv7 does not preserve the frame register R11 in
; handlers. This may affect C++ EH when accessing arguments on the stack in
; functions with stack realignment.

; C++ source:
; struct X { int x[100]; };
; void f(X x, void (*a)(), void (*g)(int*)) {
;     alignas(64) int aligned;
;     try {
;         a();
;     } catch (...) {
;         g(&x.x[11]);
;     }
; }

%struct.X = type { [100 x i32] }

define void @"?f@@YAXUX@@P6AXXZP6AXPAH@Z@Z"(ptr byval(%struct.X) align 4 %x, ptr %a, ptr %g) personality ptr @__CxxFrameHandler3 {
entry:
  %g.addr = alloca ptr, align 4
  %a.addr = alloca ptr, align 4
  %aligned = alloca i32, align 64
  store ptr %g, ptr %g.addr, align 4
  store ptr %a, ptr %a.addr, align 4
  %0 = load ptr, ptr %a.addr, align 4
  invoke void %0()
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %1 = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %2 = catchpad within %1 [ptr null, i32 64, ptr null]
  %3 = load ptr, ptr %g.addr, align 4
  %x1 = getelementptr inbounds nuw %struct.X, ptr %x, i32 0, i32 0
  %arrayidx = getelementptr inbounds [100 x i32], ptr %x1, i32 0, i32 11
  call void %3(ptr %arrayidx) [ "funclet"(token %2) ]
  catchret from %2 to label %catchret.dest

catchret.dest:                                    ; preds = %catch
  br label %try.cont

try.cont:                                         ; preds = %catchret.dest, %invoke.cont
  ret void

invoke.cont:                                      ; preds = %entry
  br label %try.cont
}

declare dso_local arm_aapcs_vfpcc i32 @__CxxFrameHandler3(...)

attributes #0 = { uwtable }
