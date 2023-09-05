// RUN: mlir-opt %s --nvvm-attach-target='module=nvvm.* O=3 chip=sm_90' --rocdl-attach-target='module=rocdl.* O=3 chip=gfx90a' | FileCheck %s
// RUN: mlir-opt %s --nvvm-attach-target='module=options.* O=1 chip=sm_70 fast=true ftz=true' --rocdl-attach-target='module=options.* l=file1.bc,file2.bc wave64=false finite-only=true' | FileCheck %s --check-prefix=CHECK_OPTS

module attributes {gpu.container_module} {
// Verify the target is appended.
// CHECK: @nvvm_module_1 [#nvvm.target<O = 3, chip = "sm_90">] {
gpu.module @nvvm_module_1 {
}
// Verify the target is appended.
// CHECK: @nvvm_module_2 [#nvvm.target<chip = "sm_60">, #nvvm.target<O = 3, chip = "sm_90">] {
gpu.module @nvvm_module_2 [#nvvm.target<chip = "sm_60">] {
}
// Verify the target is not added multiple times.
// CHECK: @nvvm_module_3 [#nvvm.target<O = 3, chip = "sm_90">] {
gpu.module @nvvm_module_3 [#nvvm.target<O = 3, chip = "sm_90">] {
}
// Verify the NVVM target is not added as it fails to match the regex, but the ROCDL does get appended.
// CHECK: @rocdl_module [#rocdl.target<O = 3, chip = "gfx90a">] {
gpu.module @rocdl_module {
}
// Check the options were added.
// CHECK_OPTS: @options_module_1 [#nvvm.target<O = 1, chip = "sm_70", flags = {fast, ftz}>, #rocdl.target<flags = {finite_only, no_wave64}, link = ["file1.bc", "file2.bc"]>]  {
gpu.module @options_module_1 {
}
// Check the options were added and that the first target was preserved.
// CHECK_OPTS: @options_module_2 [#nvvm.target<O = 3, chip = "sm_90">, #nvvm.target<O = 1, chip = "sm_70", flags = {fast, ftz}>, #rocdl.target<flags = {finite_only, no_wave64}, link = ["file1.bc", "file2.bc"]>]  {
gpu.module @options_module_2 [#nvvm.target<O = 3, chip = "sm_90">] {
}
}
