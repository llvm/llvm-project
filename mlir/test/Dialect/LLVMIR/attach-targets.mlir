// RUN: mlir-opt %s \
// RUN:   --nvvm-attach-target='module=nvvm.* O=3 chip=sm_90' \
// RUN:   --rocdl-attach-target='module=rocdl.* O=3 chip=gfx90a' \
// RUN:   --xevm-attach-target='module=xevm.* O=3 chip=pvc' \
// RUN: | FileCheck %s
// RUN: mlir-opt %s \
// RUN:   --nvvm-attach-target='module=options.* O=1 chip=sm_70 fast=true ftz=true' \
// RUN:   --rocdl-attach-target='module=options.* l=file1.bc,file2.bc wave64=false finite-only=true' \
// RUN:   --xevm-attach-target='module=options.* O=1 chip=pvc' \
// RUN: | FileCheck %s --check-prefix=CHECK-OPTIONS

module attributes {gpu.container_module} {

// CHECK-LABEL: @nvvm_module_1
// CHECK-SAME: [#nvvm.target<O = 3, chip = "sm_90">]
gpu.module @nvvm_module_1 {
}

// CHECK-LABEL: @nvvm_module_2
// CHECK-SAME: [#nvvm.target<chip = "sm_60">, #nvvm.target<O = 3, chip = "sm_90">]
gpu.module @nvvm_module_2 [#nvvm.target<chip = "sm_60">] {
}

// Verify the target is not added multiple times.
// CHECK-LABEL: @nvvm_module_3
// CHECK-SAME: [#nvvm.target<O = 3, chip = "sm_90">]
gpu.module @nvvm_module_3 [#nvvm.target<O = 3, chip = "sm_90">] {
}

// Verify only the ROCDL target is added.
// CHECK-LABEL: @rocdl_module
// CHECK-SAME: [#rocdl.target<O = 3, chip = "gfx90a">]
gpu.module @rocdl_module {
}

// Verify only the XeVM target is added.
// CHECK-LABEL: @xevm_module
// CHECK-SAME: [#xevm.target<O = 3, chip = "pvc">]
gpu.module @xevm_module {
}

// CHECK-OPTIONS-LABEL: @options_module_1
// CHECK-OPTIONS-SAME: [#nvvm.target<O = 1, chip = "sm_70", flags = {fast, ftz}>,
// CHECK-OPTIONS-SAME: #rocdl.target<flags = {finite_only, no_wave64}, link = ["file1.bc", "file2.bc"]>,
// CHECK-OPTIONS-SAME: #xevm.target<O = 1, chip = "pvc">]
gpu.module @options_module_1 {
}

// CHECK-OPTIONS-LABEL: @options_module_2
// CHECK-OPTIONS-SAME: [#nvvm.target<O = 3, chip = "sm_90">,
// CHECK-OPTIONS-SAME: #nvvm.target<O = 1, chip = "sm_70", flags = {fast, ftz}>,
// CHECK-OPTIONS-SAME: #rocdl.target<flags = {finite_only, no_wave64}, link = ["file1.bc", "file2.bc"]>,
// CHECK-OPTIONS-SAME: #xevm.target<O = 1, chip = "pvc">]
gpu.module @options_module_2 [#nvvm.target<O = 3, chip = "sm_90">] {
}
}
