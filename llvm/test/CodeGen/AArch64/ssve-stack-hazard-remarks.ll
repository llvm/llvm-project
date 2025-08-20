; RUN: llc < %s -mtriple=aarch64 -mattr=+sve2 -pass-remarks-analysis=sme -aarch64-stack-hazard-remark-size=64 -o /dev/null < %s 2>&1 | FileCheck %s --check-prefixes=CHECK
; RUN: llc < %s -mtriple=aarch64 -mattr=+sve2 -pass-remarks-analysis=sme -aarch64-stack-hazard-size=1024 -o /dev/null < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-PADDING
; RUN: llc < %s -mtriple=aarch64 -mattr=+sve2 -pass-remarks-analysis=sme -aarch64-enable-zpr-predicate-spills -aarch64-stack-hazard-remark-size=64 -o /dev/null < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-ZPR-PRED-SPILLS
; RUN: llc < %s -mtriple=aarch64 -mattr=+sve2 -pass-remarks-analysis=sme -aarch64-enable-zpr-predicate-spills -aarch64-stack-hazard-size=1024 -o /dev/null < %s 2>&1 | FileCheck %s --check-prefixes=CHECK-ZPR-PRED-SPILLS-WITH-PADDING

; Don't emit remarks for non-streaming functions.
define float @csr_x20_stackargs_notsc(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i) {
; CHECK-NOT: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs_notsc':
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs_notsc':
entry:
  tail call void asm sideeffect "", "~{x20}"() #1
  ret float %i
}

; Don't emit remarks for functions that only access GPR stack objects.
define i64 @stackargs_gpr(i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h, i64 %i) #2 {
; CHECK-NOT: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs_gpr':
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs_gpr':
entry:
  ret i64 %i
}

; Don't emit remarks for functions that only access FPR stack objects.
define double @stackargs_fpr(double %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i) #2 {
; CHECK-NOT: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs_fpr':
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs_fpr':
entry:
  ret double %i
}

; As this case is handled by addition of stack hazard padding, only emit remarks when this is not switched on.
define i32 @csr_d8_alloci64(i64 %d) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'csr_d8_alloci64': FPR stack object at [SP-16] is too close to GPR stack object at [SP-8]
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'csr_d8_alloci64':
entry:
  %a = alloca i64
  tail call void asm sideeffect "", "~{d8}"() #1
  store i64 %d, ptr %a
  ret i32 0
}

; As this case is handled by addition of stack hazard padding, only emit remarks when this is not switched on.
define i32 @csr_d8_allocnxv4i32(i64 %d) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'csr_d8_allocnxv4i32': FPR stack object at [SP-16] is too close to GPR stack object at [SP-8]
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'csr_d8_allocnxv4i32':
entry:
  %a = alloca <vscale x 4 x i32>
  tail call void asm sideeffect "", "~{d8}"() #1
  store <vscale x 4 x i32> zeroinitializer, ptr %a
  ret i32 0
}

define float @csr_x20_stackargs(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs': GPR stack object at [SP-16] is too close to FPR stack object at [SP+0]
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'csr_x20_stackargs': GPR stack object at [SP-16] is too close to FPR stack object at [SP+0]
entry:
  tail call void asm sideeffect "", "~{x20}"() #1
  ret float %i
}

; In this case, addition of stack hazard padding triggers x29 (fp) spill, so we hazard occurs between FPR argument and GPR spill.
define float @csr_d8_stackargs(float %a, float %b, float %c, float %d, float %e, float %f, float %g, float %h, float %i) #2 {
; CHECK-NOT: remark: <unknown>:0:0: stack hazard in 'csr_d8_stackargs':
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'csr_d8_stackargs': GPR stack object at [SP-8] is too close to FPR stack object at [SP+0]
entry:
  tail call void asm sideeffect "", "~{d8}"() #1
  ret float %i
}

; SVE calling conventions
; Predicate register spills end up in FP region, currently. This can be
; mitigated with the -aarch64-enable-zpr-predicate-spills option.

define i32 @svecc_call(<4 x i16> %P0, ptr %P1, i32 %P2, <vscale x 16 x i8> %P3, i16 %P4) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'svecc_call': PPR stack object at [SP-64-258 * vscale] is too close to FPR stack object at [SP-64-256 * vscale]
; CHECK: remark: <unknown>:0:0: stack hazard in 'svecc_call': FPR stack object at [SP-64-16 * vscale] is too close to GPR stack object at [SP-64]
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'svecc_call': PPR stack object at [SP-1088-258 * vscale] is too close to FPR stack object at [SP-1088-256 * vscale]
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'svecc_call':
; CHECK-ZPR-PRED-SPILLS-NOT: <unknown>:0:0: stack hazard in 'svecc_call': PPR stack object at {{.*}} is too close to FPR stack object
; CHECK-ZPR-PRED-SPILLS: <unknown>:0:0: stack hazard in 'svecc_call': FPR stack object at [SP-64-16 * vscale] is too close to GPR stack object at [SP-64]
; CHECK-ZPR-PRED-SPILLS-WITH-PADDING-NOT: <unknown>:0:0: stack hazard in 'svecc_call': PPR stack object at {{.*}} is too close to FPR stack object
; CHECK-ZPR-PRED-SPILLS-WITH-PADDING-NOT: <unknown>:0:0: stack hazard in 'svecc_call': FPR stack object at {{.*}} is too close to GPR stack object
entry:
  tail call void asm sideeffect "", "~{x0},~{x28},~{x27},~{x3}"() #2
  %call = call ptr @memset(ptr noundef nonnull %P1, i32 noundef 45, i32 noundef 37)
  ret i32 -396142473
}

define i32 @svecc_alloca_call(<4 x i16> %P0, ptr %P1, i32 %P2, <vscale x 16 x i8> %P3, i16 %P4) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'svecc_alloca_call': PPR stack object at [SP-64-258 * vscale] is too close to FPR stack object at [SP-64-256 * vscale]
; CHECK: remark: <unknown>:0:0: stack hazard in 'svecc_alloca_call': FPR stack object at [SP-64-16 * vscale] is too close to GPR stack object at [SP-64]
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'svecc_alloca_call': PPR stack object at [SP-1088-258 * vscale] is too close to FPR stack object at [SP-1088-256 * vscale]
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'svecc_alloca_call':
; CHECK-ZPR-PRED-SPILLS-NOT: <unknown>:0:0: stack hazard in 'svecc_call': PPR stack object at {{.*}} is too close to FPR stack object
; CHECK-ZPR-PRED-SPILLS: <unknown>:0:0: stack hazard in 'svecc_alloca_call': FPR stack object at [SP-64-16 * vscale] is too close to GPR stack object at [SP-64]
; CHECK-ZPR-PRED-SPILLS-WITH-PADDING-NOT: <unknown>:0:0: stack hazard in 'svecc_alloca_call': PPR stack object at {{.*}} is too close to FPR stack object
; CHECK-ZPR-PRED-SPILLS-WITH-PADDING-NOT: <unknown>:0:0: stack hazard in 'svecc_alloca_call': FPR stack object at {{.*}} is too close to GPR stack object
entry:
  tail call void asm sideeffect "", "~{x0},~{x28},~{x27},~{x3}"() #2
  %0 = alloca [37 x i8], align 16
  %call = call ptr @memset(ptr noundef nonnull %0, i32 noundef 45, i32 noundef 37)
  ret i32 -396142473
}
declare ptr @memset(ptr, i32, i32)

%struct.mixed_struct = type { i32, float }

define i32 @mixed_stack_object(i32  %a, float %b) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'mixed_stack_object': Mixed stack object at [SP-8] accessed by both GP and FP instructions
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'mixed_stack_object': Mixed stack object at [SP-8] accessed by both GP and FP instructions
entry:
  %s = alloca %struct.mixed_struct
  %s.i = getelementptr %struct.mixed_struct, ptr %s, i32 0, i32 0
  %s.f = getelementptr %struct.mixed_struct, ptr %s, i32 0, i32 1
  store i32 %a, ptr %s.i
  store float %b, ptr %s.f
  ret i32 %a
}

define i32 @mixed_stack_objects(i32  %a, float %b) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'mixed_stack_objects': Mixed stack object at [SP-16] is too close to Mixed stack object at [SP-8]
; CHECK: remark: <unknown>:0:0: stack hazard in 'mixed_stack_objects': Mixed stack object at [SP-16] accessed by both GP and FP instructions
; CHECK: remark: <unknown>:0:0: stack hazard in 'mixed_stack_objects': Mixed stack object at [SP-8] accessed by both GP and FP instructions
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'mixed_stack_objects': Mixed stack object at [SP-16] is too close to Mixed stack object at [SP-8]
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'mixed_stack_objects': Mixed stack object at [SP-16] accessed by both GP and FP instructions
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'mixed_stack_objects': Mixed stack object at [SP-8] accessed by both GP and FP instructions
entry:
  %s0 = alloca %struct.mixed_struct
  %s0.i = getelementptr %struct.mixed_struct, ptr %s0, i32 0, i32 0
  %s0.f = getelementptr %struct.mixed_struct, ptr %s0, i32 0, i32 1
  store i32 %a, ptr %s0.i
  store float %b, ptr %s0.f

  %s1 = alloca %struct.mixed_struct
  %s1.i = getelementptr %struct.mixed_struct, ptr %s1, i32 0, i32 0
  %s1.f = getelementptr %struct.mixed_struct, ptr %s1, i32 0, i32 1
  store i32 %a, ptr %s1.i
  store float %b, ptr %s1.f

  ret i32 %a
}

; VLA-area stack objects are not separated.
define i32 @csr_d8_allocnxv4i32i32f64_vlai32f64(double %d, i32 %i) #2 {
; CHECK: remark: <unknown>:0:0: stack hazard in 'csr_d8_allocnxv4i32i32f64_vlai32f64': GPR stack object at [SP-48-16 * vscale] is too close to FPR stack object at [SP-48-16 * vscale]
; CHECK: remark: <unknown>:0:0: stack hazard in 'csr_d8_allocnxv4i32i32f64_vlai32f64': FPR stack object at [SP-32] is too close to GPR stack object at [SP-24]
; CHECK-PADDING: remark: <unknown>:0:0: stack hazard in 'csr_d8_allocnxv4i32i32f64_vlai32f64': GPR stack object at [SP-2096-16 * vscale] is too close to FPR stack object at [SP-2096-16 * vscale]
; CHECK-PADDING-NOT: remark: <unknown>:0:0: stack hazard in 'csr_d8_allocnxv4i32i32f64_vlai32f64':
entry:
  %a = alloca <vscale x 4 x i32>
  %0 = zext i32 %i to i64
  %vla0 = alloca i32, i64 %0
  %vla1 = alloca double, i64 %0
  %c = alloca double
  tail call void asm sideeffect "", "~{d8}"() #1
  store <vscale x 4 x i32> zeroinitializer, ptr %a
  store i32 zeroinitializer, ptr %vla0
  store double %d, ptr %vla1
  store double %d, ptr %c
  ret i32 0
}

attributes #2 = { "aarch64_pstate_sm_compatible" }
