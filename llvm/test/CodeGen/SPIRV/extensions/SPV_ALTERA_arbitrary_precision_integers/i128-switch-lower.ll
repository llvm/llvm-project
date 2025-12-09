; RUN: not llc -O0 -mtriple=spirv64-unknown-unknown %s -o %t.spvt 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_ALTERA_arbitrary_precision_integers %s -o - -filetype=obj | spirv-val %}

; CHECK-ERROR: LLVM ERROR: OpTypeInt type with a width other than 8, 16, 32 or 64 bits requires the following SPIR-V extension: SPV_ALTERA_arbitrary_precision_integers

; CHECK: OpCapability ArbitraryPrecisionIntegersALTERA
; CHECK: OpExtension "SPV_ALTERA_arbitrary_precision_integers"
; CHECK: OpName %[[#Test:]] "test"
; CHECK: OpName %[[#Exit:]] "exit"
; CHECK: %[[#Int128Ty:]] = OpTypeInt 128 0
; CHECK: %[[#UndefInt128:]] = OpUndef %[[#Int128Ty]]

; CHECK: %[[#Test]] = OpFunction
define void @test() {
entry:
; CHECK: OpSwitch %[[#UndefInt128]] %[[#Exit]] 0 0 3 0 %[[#Exit]] 0 0 5 0 %[[#Exit]] 0 0 4 0 %[[#Exit]] 0 0 8 0 %[[#Exit]]
  switch i128 poison, label %exit [
    i128 55340232221128654848, label %exit
    i128 92233720368547758080, label %exit
    i128 73786976294838206464, label %exit
    i128 147573952589676412928, label %exit
  ]
exit:
  unreachable
}
