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

; Check that the changes in PR #184953 forcing r6 indexing in SEH funclets does not affect C++ EH.

%struct.X = type { [100 x i32] }

define dso_local arm_aapcs_vfpcc void @f(ptr noundef byval(%struct.X) align 4 %0, ptr noundef %1) personality ptr @__CxxFrameHandler3 {
entry:
; CHECK-LABEL: f:
; CHECK: .seh_proc f
; CHECK: .seh_handler __CxxFrameHandler3, %unwind, %except
; CHECK: push.w {r11, lr}
; CHECK: mov r11, sp
; CHECK: .seh_save_sp r11
; CHECK: .seh_endprologue
; CHECK: bfc [[REG1:r[0-9]+]], #0, #6
; CHECK: mov sp, [[REG1]]

  %aligned = alloca i32, align 64
  invoke arm_aapcs_vfpcc void %1()
          to label %invoke.cont unwind label %ehcleanup

invoke.cont:                                      ; preds = %entry
  %gep.x = getelementptr inbounds i8, ptr %0, i32 44
  call arm_aapcs_vfpcc void @g(ptr noundef nonnull %gep.x, ptr noundef nonnull %aligned)
  ret void

ehcleanup:                                        ; preds = %entry
  %cp = cleanuppad within none []
; CHECK: .seh_proc {{.*}}dtor{{.*}}
; CHECK: push.w {r11, lr}
; CHECK: mov r11, sp
; CHECK: .seh_save_sp r11
; CHECK: .seh_endprologue
; CHECK: bfc [[REG2:r[0-9]+]], #0, #6
; CHECK: mov sp, [[REG2]]
; CHECK: add.w r0, r11, #{{[0-9]+}}
; CHECK: adds r0, #44
; CHECK: bl g

  %gep.unwind = getelementptr inbounds i8, ptr %0, i32 44
  call arm_aapcs_vfpcc void @g(ptr noundef nonnull %gep.unwind, ptr noundef nonnull %aligned) [ "funclet"(token %cp) ]
  cleanupret from %cp unwind to caller
}

declare dso_local arm_aapcs_vfpcc i32 @__CxxFrameHandler3(...)

declare dso_local arm_aapcs_vfpcc void @g(ptr noundef, ptr noundef)
