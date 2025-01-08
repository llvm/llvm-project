; Adapted from https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/main/test/extensions/INTEL/SPV_INTEL_float_controls2

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls,+SPV_INTEL_float_controls2 %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_float_controls,+SPV_INTEL_float_controls2 %s -o - -filetype=obj | spirv-val %}

; CHECK-NOT: {{ExecutionMode.*(DenormPreserve|DenormFlushToZero|SignedZeroInfNanPreserve|RoundingModeRTE|RoundingModeRTZ|RoundingModeRTPINTEL|RoundingModeRTNINTEL|FloatingPointModeALTINTEL|FloatingPointModeIEEEINTEL)}}
define dso_local dllexport spir_kernel void @k_no_fc(i32 %ibuf, i32 %obuf) local_unnamed_addr #16 {
entry:
  ret void
}

attributes #16 = { noinline norecurse nounwind readnone "VCMain" "VCFunction" "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 8.0.1"}
