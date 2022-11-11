; RUN: llc < %s -relocation-model=static -O1 -disable-ppc-sco=false -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s -check-prefix=CHECK-SCO
; RUN: llc < %s -relocation-model=static -O1 -disable-ppc-sco=false -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-SCO
; RUN: llc < %s -relocation-model=static -O1 -disable-ppc-sco=false -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-SCO
; RUN: llc < %s -relocation-model=static -O1 -disable-ppc-sco=false -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 -code-model=small | FileCheck %s -check-prefix=SCM

; No combination of "powerpc64le-unknown-linux-gnu" + "CHECK-SCO", because
; only Power8 (and later) fully support LE.

%S_56 = type { [13 x i32], i32 }
%S_64 = type { [15 x i32], i32 }
%S_32 = type { [7 x i32], i32 }

; Function Attrs: noinline nounwind
define dso_local void @callee_56_copy([7 x i64] %a, ptr %b) #0 { ret void }
define dso_local void @callee_64_copy([8 x i64] %a, ptr %b) #0 { ret void }

; Function Attrs: nounwind
define dso_local void @caller_56_reorder_copy(ptr %b, [7 x i64] %a) #1 {
  tail call void @callee_56_copy([7 x i64] %a, ptr %b)
  ret void

; CHECK-SCO-LABEL: caller_56_reorder_copy:
; CHECK-SCO-NOT: stdu 1
; CHECK-SCO: TC_RETURNd8 callee_56_copy
}

define dso_local void @caller_64_reorder_copy(ptr %b, [8 x i64] %a) #1 {
  tail call void @callee_64_copy([8 x i64] %a, ptr %b)
  ret void

; CHECK-SCO-LABEL: caller_64_reorder_copy:
; CHECK-SCO: bl callee_64_copy
}

define dso_local void @callee_64_64_copy([8 x i64] %a, [8 x i64] %b) #0 { ret void }
define dso_local void @caller_64_64_copy([8 x i64] %a, [8 x i64] %b) #1 {
  tail call void @callee_64_64_copy([8 x i64] %a, [8 x i64] %b)
  ret void

; CHECK-SCO-LABEL: caller_64_64_copy:
; CHECK-SCO: b callee_64_64_copy
}

define internal fastcc void @callee_64_64_copy_fastcc([8 x i64] %a, [8 x i64] %b) #0 { ret void }
define dso_local void @caller_64_64_copy_ccc([8 x i64] %a, [8 x i64] %b) #1 {
  tail call fastcc void @callee_64_64_copy_fastcc([8 x i64] %a, [8 x i64] %b)
  ret void
; If caller and callee use different calling convensions, we cannot apply TCO.
; CHECK-SCO-LABEL: caller_64_64_copy_ccc:
; CHECK-SCO: bl callee_64_64_copy_fastcc
}

define dso_local void @caller_64_64_reorder_copy([8 x i64] %a, [8 x i64] %b) #1 {
  tail call void @callee_64_64_copy([8 x i64] %b, [8 x i64] %a)
  ret void

; CHECK-SCO-LABEL: caller_64_64_reorder_copy:
; CHECK-SCO: bl callee_64_64_copy
}

define dso_local void @caller_64_64_undef_copy([8 x i64] %a, [8 x i64] %b) #1 {
  tail call void @callee_64_64_copy([8 x i64] %a, [8 x i64] undef)
  ret void

; CHECK-SCO-LABEL: caller_64_64_undef_copy:
; CHECK-SCO: b callee_64_64_copy
}

define dso_local void @arg8_callee(
  float %a, i32 signext %b, float %c, ptr %d,
  i8 zeroext %e, float %f, ptr %g, i32 signext %h)
{
  ret void
}

define dso_local void @arg8_caller(float %a, i32 signext %b, i8 zeroext %c, ptr %d) {
entry:
  tail call void @arg8_callee(float undef, i32 signext undef, float undef,
                              ptr %d, i8 zeroext undef, float undef,
                              ptr undef, i32 signext undef)
  ret void

; CHECK-SCO-LABEL: arg8_caller:
; CHECK-SCO: b arg8_callee
}

; Struct return test

; Function Attrs: noinline nounwind
define dso_local void @callee_sret_56(ptr noalias sret(%S_56) %agg.result) #0 { ret void }
define dso_local void @callee_sret_32(ptr noalias sret(%S_32) %agg.result) #0 { ret void }

; Function Attrs: nounwind
define dso_local void @caller_do_something_sret_32(ptr noalias sret(%S_32) %agg.result) #1 {
  %1 = alloca %S_56, align 4
  call void @callee_sret_56(ptr nonnull sret(%S_56) %1)
  tail call void @callee_sret_32(ptr sret(%S_32) %agg.result)
  ret void

; CHECK-SCO-LABEL: caller_do_something_sret_32:
; CHECK-SCO: stdu 1
; CHECK-SCO: bl callee_sret_56
; CHECK-SCO: addi 1
; CHECK-SCO: TC_RETURNd8 callee_sret_32
}

define dso_local void @caller_local_sret_32(ptr %a) #1 {
  %tmp = alloca %S_32, align 4
  tail call void @callee_sret_32(ptr nonnull sret(%S_32) %tmp)
  ret void

; CHECK-SCO-LABEL: caller_local_sret_32:
; CHECK-SCO: bl callee_sret_32
}

attributes #0 = { noinline nounwind  }
attributes #1 = { nounwind }

define dso_local void @f128_callee(ptr %ptr, ppc_fp128 %a, ppc_fp128 %b) { ret void }
define dso_local void @f128_caller(ptr %ptr, ppc_fp128 %a, ppc_fp128 %b) {
  tail call void @f128_callee(ptr %ptr, ppc_fp128 %a, ppc_fp128 %b)
  ret void

; CHECK-SCO-LABEL: f128_caller:
; CHECK-SCO: b f128_callee
}

; weak linkage test
%class.T = type { [2 x i8] }

define weak_odr hidden void @wo_hcallee(ptr %this, ptr %c) { ret void }
define dso_local void @wo_hcaller(ptr %this, ptr %c) {
  tail call void @wo_hcallee(ptr %this, ptr %c)
  ret void

; CHECK-SCO-LABEL: wo_hcaller:
; CHECK-SCO: bl wo_hcallee

; SCM-LABEL: wo_hcaller:
; SCM:       bl wo_hcallee
}

define weak_odr protected void @wo_pcallee(ptr %this, ptr %c) { ret void }
define dso_local void @wo_pcaller(ptr %this, ptr %c) {
  tail call void @wo_pcallee(ptr %this, ptr %c)
  ret void

; CHECK-SCO-LABEL: wo_pcaller:
; CHECK-SCO: bl wo_pcallee

; SCM-LABEL: wo_pcaller:
; SCM:       bl wo_pcallee
}

define weak_odr void @wo_callee(ptr %this, ptr %c) { ret void }
define dso_local void @wo_caller(ptr %this, ptr %c) {
  tail call void @wo_callee(ptr %this, ptr %c)
  ret void

; CHECK-SCO-LABEL: wo_caller:
; CHECK-SCO: bl wo_callee

; SCM-LABEL: wo_caller:
; SCM:       bl wo_callee
}

define weak protected void @w_pcallee(ptr %ptr) { ret void }
define dso_local void @w_pcaller(ptr %ptr) {
  tail call void @w_pcallee(ptr %ptr)
  ret void

; CHECK-SCO-LABEL: w_pcaller:
; CHECK-SCO: bl w_pcallee

; SCM-LABEL: w_pcaller:
; SCM:       bl w_pcallee
}

define weak hidden void @w_hcallee(ptr %ptr) { ret void }
define dso_local void @w_hcaller(ptr %ptr) {
  tail call void @w_hcallee(ptr %ptr)
  ret void

; CHECK-SCO-LABEL: w_hcaller:
; CHECK-SCO: bl w_hcallee

; SCM-LABEL: w_hcaller:
; SCM:       bl w_hcallee
}

define weak void @w_callee(ptr %ptr) { ret void }
define dso_local void @w_caller(ptr %ptr) {
  tail call void @w_callee(ptr %ptr)
  ret void

; CHECK-SCO-LABEL: w_caller:
; CHECK-SCO: bl w_callee

; SCM-LABEL: w_caller:
; SCM:       bl w_callee
}

%struct.byvalTest = type { [8 x i8] }
@byval = common global %struct.byvalTest zeroinitializer

define dso_local void @byval_callee(ptr byval(%struct.byvalTest) %ptr) { ret void }
define dso_local void @byval_caller() {
  tail call void @byval_callee(ptr byval(%struct.byvalTest) @byval)
  ret void

; CHECK-SCO-LABEL: bl byval_callee
; CHECK-SCO: bl byval_callee
}
