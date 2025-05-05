; template<typename Function>
; [[intel::prefer_dsp]]
; [[intel::propagate_dsp_preference]]
; void math_prefer_dsp_propagate(Function f)
; {
;   f();
; }

; int main() {
;   math_prefer_dsp_propagate([]() {
;     int a = 0;
;     a += 1;
;   });

; return 0;
; }

; RUN: llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_INTEL_fpga_dsp_control %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV


; CHECK-SPIRV: OpCapability FPGADSPControlINTEL
; CHECK-SPIRV: OpExtension "SPV_INTEL_fpga_dsp_control"
; CHECK-SPIRV: OpName %[[#FuncNameId:]] "_Z25math_prefer_dsp_propagateIZ4mainE3$_0EvT_"
; CHECK-SPIRV: OpDecorate %[[#FuncNameId]] MathOpDSPModeINTEL 1 1


; ModuleID = 'prefer_dsp_propagate.cpp'
source_filename = "prefer_dsp_propagate.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

; Function Attrs: noinline norecurse optnone mustprogress
define dso_local i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %agg.tmp = alloca %class.anon, align 1
  store i32 0, ptr %retval, align 4
  call spir_func void @"_Z25math_prefer_dsp_propagateIZ4mainE3$_0EvT_"(ptr byval(%class.anon) align 1 %agg.tmp)
  ret i32 0
}

; Function Attrs: noinline optnone mustprogress
define internal spir_func void @"_Z25math_prefer_dsp_propagateIZ4mainE3$_0EvT_"(ptr byval(%class.anon) align 1 %f) #1 !prefer_dsp !3 !propagate_dsp_preference !3 {
entry:
  call spir_func void @"_ZZ4mainENK3$_0clEv"(ptr nonnull dereferenceable(1) %f)
  ret void
}

; Function Attrs: noinline nounwind optnone mustprogress
define internal spir_func void @"_ZZ4mainENK3$_0clEv"(ptr nonnull dereferenceable(1) %this) #2 align 2 {
entry:
  %this.addr = alloca ptr, align 8
  %a = alloca i32, align 4
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  store i32 0, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  ret void
}

attributes #0 = { noinline norecurse optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noinline nounwind optnone mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.used.extensions = !{!1}
!opencl.used.optional.core.features = !{!1}
!opencl.compiler.options = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{}
!2 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 7d09e1d7cf27ce781e83f9d388a7a3e1e6487ead)"}
!3 = !{i32 1}
