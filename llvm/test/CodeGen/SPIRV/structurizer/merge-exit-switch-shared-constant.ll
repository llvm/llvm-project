; RUN: llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute -spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute -spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val %}

; The structurizer builds the switch that merges the three exits. The long
; vector type emits the constant 1, so one case operand reaches the
; pre-legalizer as a SPIR-V constant and the other as a G_CONSTANT.

target triple = "spirv1.6-unknown-vulkan1.3-compute"

; CHECK: OpSwitch %[[#]] %[[#]] 1 %[[#]] 2 %[[#]]

define <1 x float> @csmain() {
entry:
  br i1 false, label %if.then, label %if.end

if.then:
  br i1 false, label %crit.edge, label %if.else

crit.edge:
  br label %if.end

if.else:
  br i1 false, label %if.then47, label %if.else67

if.then47:
  ret <1 x float> zeroinitializer

if.else67:
  ret <1 x float> zeroinitializer

if.end:
  ret <1 x float> zeroinitializer
}
