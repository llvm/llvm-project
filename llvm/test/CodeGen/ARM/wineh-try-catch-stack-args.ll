; RUN: llc -mtriple thumbv7-windows-msvc -o - %s | FileCheck %s
; RUN: llc -mtriple thumbv7-windows-msvc -frame-pointer=all -o - %s | FileCheck %s --check-prefix=FP

; The catch funclet reaches the parent's stack arguments through r6, the
; parent's frame base, and not through r11, which is the funclet's own when
; there is a frame pointer (second RUN line).
; FP-LABEL: "?f@@YAXUX@@P6AXXZP6AXPAH@Z@Z":
; FP:       sub sp, #[[#FARGS:]]
; FP:       push.w {r6, r7, r11, lr}
; FP:       sub sp, #[[#FSUB:]]
; FP-LABEL: "?catch$2@?0??f@@YAXUX@@P6AXXZP6AXPAH@Z@Z@4HA":
; FP:       .seh_endprologue
; FP-NEXT:  add.w r0, r6, #[[#FSUB + 16]]
; FP-NEXT:  ldr.w r1, [r6, #[[#FSUB + 16 + FARGS + 388]]]
; FP-NEXT:  adds r0, #44
; FP-NEXT:  blx r1
; CHECK-LABEL: "?f@@YAXUX@@P6AXXZP6AXPAH@Z@Z":
; CHECK:       sub sp, #[[#ARGS:]]
; CHECK:       push {r6, lr}
; CHECK:       sub sp, #[[#SUB:]]
; CHECK:       .seh_endprologue
; CHECK-NEXT:  mov r6, sp
; CHECK:       ldr.w r12, [r6, #[[#SUB + 8 + ARGS + 384]]]
; CHECK:       blx r12
; CHECK-LABEL: "?catch$2@?0??f@@YAXUX@@P6AXXZP6AXPAH@Z@Z@4HA":
; CHECK:       .seh_endprologue
; CHECK-NEXT:  add.w r0, r6, #[[#SUB + 8]]
; CHECK-NEXT:  ldr.w r1, [r6, #[[#SUB + 8 + ARGS + 388]]]
; CHECK-NEXT:  adds r0, #44
; CHECK-NEXT:  blx r1
; CHECK:       .long [[#%d,0 - SUB - 8 - ARGS]] @ UnwindHelp

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
