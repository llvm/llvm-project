; RUN: llc -O2 -mtriple=thumbv7-pc-windows-msvc < %s | FileCheck %s

; C++ source:
; struct X { int x[100]; };
; void g(int*, int*) noexcept;
; void f(X x, void (*a)()) {
;     alignas(64) int aligned;
;     struct A {
;         X &x;
;         int &aligned;
;         ~A() {g(&x.x[11], &aligned);}
;     } obj = {x, aligned};
;     a();
; }

; A C++ cleanup in a frame with an alignas(64) local. The frame is not realigned:
; the local is at a 64-byte aligned address in an over-sized block, and the
; cleanup funclet reaches it and the byval argument through r6.
; CHECK-LABEL: f:
; CHECK:       .seh_handler __CxxFrameHandler3, %unwind, %except
; CHECK:       sub sp, #[[#ARGS:]]
; CHECK:       push {r6, lr}
; CHECK:       sub sp, #[[#SUB:]]
; CHECK:       .seh_endprologue
; CHECK-NEXT:  mov r6, sp
; CHECK:       mvn [[T:r[0-9]+]], #1
; CHECK:       str.w [[T]], [r6]
; CHECK-NEXT:  str.w [[T]], [r6, #4]
; CHECK:       add.w [[P:r[0-9]+]], r6, #{{[0-9]+}}
; CHECK-NEXT:  adds [[P]], #56
; CHECK-NEXT:  bic [[P]], [[P]], #56
; CHECK:       str [[P]], [r6, #[[#SLOT:]]]
; CHECK:       blx
; CHECK-LABEL: "?dtor$2@?0?f@4HA":
; CHECK:       .seh_endprologue
; CHECK-NEXT:  add.w r0, r6, #[[#SUB + 8]]
; CHECK-NEXT:  ldr r1, [r6, #[[#SLOT]]]
; CHECK-NEXT:  adds r0, #44
; CHECK-NEXT:  bl g
; CHECK:       $cppxdata$f:
; CHECK:       .long [[#%d,0 - SUB - 8 - ARGS]] @ UnwindHelp
; CHECK-NEXT:  .long 0 @ ESTypeList
; CHECK-NEXT:  .long 1 @ EHFlags

%struct.X = type { [100 x i32] }

define dso_local arm_aapcs_vfpcc void @f(ptr noundef byval(%struct.X) align 4 %0, ptr noundef %1) personality ptr @__CxxFrameHandler3 {
entry:
  %aligned = alloca i32, align 64
  invoke arm_aapcs_vfpcc void %1()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %gep.x = getelementptr inbounds i8, ptr %0, i32 44
  call arm_aapcs_vfpcc void @g(ptr noundef nonnull %gep.x, ptr noundef nonnull %aligned)
  ret void

ehcleanup:                                        ; preds = %entry
  %cp = cleanuppad within none []
  %gep.unwind = getelementptr inbounds i8, ptr %0, i32 44
  call arm_aapcs_vfpcc void @g(ptr noundef nonnull %gep.unwind, ptr noundef nonnull %aligned) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

declare dso_local arm_aapcs_vfpcc i32 @__CxxFrameHandler3(...)

declare dso_local arm_aapcs_vfpcc void @g(ptr noundef, ptr noundef)
