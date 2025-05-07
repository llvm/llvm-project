; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_EXT_optnone %s -o - | FileCheck %s --check-prefixes=CHECK-EXTENSION
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-NO-EXTENSION

; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_optnone %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_EXT_optnone,+SPV_INTEL_optnone %s -o - | FileCheck %s --check-prefixes=CHECK-TWO-EXTENSIONS
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=all %s -o - | FileCheck %s --check-prefixes=CHECK-ALL-EXTENSIONS

; CHECK-EXTENSION: OpCapability OptNoneEXT
; CHECK-EXTENSION: OpExtension "SPV_EXT_optnone"
; CHECK-NO-EXTENSION-NOT: OpCapability OptNoneINTEL
; CHECK-NO-EXTENSION-NOT: OpCapability OptNoneEXT
; CHECK-NO-EXTENSION-NOT: OpExtension "SPV_INTEL_optnone"
; CHECK-NO-EXTENSION-NOT: OpExtension "SPV_EXT_optnone"

; CHECK-TWO-EXTENSIONS: OpExtension "SPV_INTEL_optnone"
; CHECK-ALL-EXTENSIONS: OpExtension "SPV_INTEL_optnone"

define spir_func void @foo() #0 {
; CHECK-EXTENSION: %[[#]] = OpFunction %[[#]] DontInline|OptNoneEXT %[[#]]
entry:
  ret void
}

attributes #0 = { nounwind optnone noinline }
