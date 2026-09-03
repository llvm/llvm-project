; RUN: split-file %s %t
; RUN: not llc -mtriple=aarch64-linux-gnu %t/negative.ll -o - 2>&1 | FileCheck %s
; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve %t/negative.ll -o - 2>&1 | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve %t/positive.ll -o - | FileCheck %s --check-prefix=POS

; CHECK-COUNT-7: error: could not allocate input reg for constraint '{z0}'
; CHECK-COUNT-14: error: could not allocate output register for constraint '{z0}'

; Test representative non-scalable types for each selected bit size.

;--- negative.ll

define void @input_8bit(<1 x i8> %value) {
  call void asm sideeffect "", "{z0}"(<1 x i8> %value)
  ret void
}

define void @input_16bit(<2 x i8> %value) {
  call void asm sideeffect "", "{z0}"(<2 x i8> %value)
  ret void
}

define void @input_32bit(<4 x i8> %value) {
  call void asm sideeffect "", "{z0}"(<4 x i8> %value)
  ret void
}

define void @input_64bit(<8 x i8> %value) {
  call void asm sideeffect "", "{z0}"(<8 x i8> %value)
  ret void
}

define void @input_128bit(<16 x i8> %value) {
  call void asm sideeffect "", "{z0}"(<16 x i8> %value)
  ret void
}

define void @input_256bit(<32 x i8> %value) {
  call void asm sideeffect "", "{z0}"(<32 x i8> %value)
  ret void
}

define void @input_scalar_i32(i32 %value) {
  call void asm sideeffect "", "{z0}"(i32 %value)
  ret void
}

define <1 x i8> @output_8bit() {
  %value = call <1 x i8> asm sideeffect "", "={z0}"()
  ret <1 x i8> %value
}

define <2 x i8> @output_16bit() {
  %value = call <2 x i8> asm sideeffect "", "={z0}"()
  ret <2 x i8> %value
}

define <4 x i8> @output_32bit() {
  %value = call <4 x i8> asm sideeffect "", "={z0}"()
  ret <4 x i8> %value
}

define <8 x i8> @output_64bit() {
  %value = call <8 x i8> asm sideeffect "", "={z0}"()
  ret <8 x i8> %value
}

define <16 x i8> @output_128bit() {
  %value = call <16 x i8> asm sideeffect "", "={z0}"()
  ret <16 x i8> %value
}

define <32 x i8> @output_256bit() {
  %value = call <32 x i8> asm sideeffect "", "={z0}"()
  ret <32 x i8> %value
}

define i32 @output_scalar_i32() {
  %value = call i32 asm sideeffect "", "={z0}"()
  ret i32 %value
}

define <1 x i8> @inout_8bit(<1 x i8> %value) {
  %result = call <1 x i8> asm sideeffect "", "={z0},0"(<1 x i8> %value)
  ret <1 x i8> %result
}

define <2 x i8> @inout_16bit(<2 x i8> %value) {
  %result = call <2 x i8> asm sideeffect "", "={z0},0"(<2 x i8> %value)
  ret <2 x i8> %result
}

define <4 x i8> @inout_32bit(<4 x i8> %value) {
  %result = call <4 x i8> asm sideeffect "", "={z0},0"(<4 x i8> %value)
  ret <4 x i8> %result
}

define <8 x i8> @inout_64bit(<8 x i8> %value) {
  %result = call <8 x i8> asm sideeffect "", "={z0},0"(<8 x i8> %value)
  ret <8 x i8> %result
}

define <16 x i8> @inout_128bit(<16 x i8> %value) {
  %result = call <16 x i8> asm sideeffect "", "={z0},0"(<16 x i8> %value)
  ret <16 x i8> %result
}

define <32 x i8> @inout_256bit(<32 x i8> %value) {
  %result = call <32 x i8> asm sideeffect "", "={z0},0"(<32 x i8> %value)
  ret <32 x i8> %result
}

define i32 @inout_scalar_i32(i32 %value) {
  %result = call i32 asm sideeffect "", "={z0},0"(i32 %value)
  ret i32 %result
}

;--- positive.ll

define void @input_nxv16i8(<vscale x 16 x i8> %value) {
; POS-LABEL: input_nxv16i8:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 16 x i8> %value)
  ret void
}

define void @input_nxv8i16(<vscale x 8 x i16> %value) {
; POS-LABEL: input_nxv8i16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 8 x i16> %value)
  ret void
}

define void @input_nxv4i32(<vscale x 4 x i32> %value) {
; POS-LABEL: input_nxv4i32:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 4 x i32> %value)
  ret void
}

define void @input_nxv2i64(<vscale x 2 x i64> %value) {
; POS-LABEL: input_nxv2i64:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 2 x i64> %value)
  ret void
}

define void @input_nxv8f16(<vscale x 8 x half> %value) {
; POS-LABEL: input_nxv8f16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 8 x half> %value)
  ret void
}

define void @input_nxv4f32(<vscale x 4 x float> %value) {
; POS-LABEL: input_nxv4f32:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 4 x float> %value)
  ret void
}

define void @input_nxv2f64(<vscale x 2 x double> %value) {
; POS-LABEL: input_nxv2f64:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 2 x double> %value)
  ret void
}

define void @input_nxv8bf16(<vscale x 8 x bfloat> %value) {
; POS-LABEL: input_nxv8bf16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  call void asm sideeffect "mov $0.d, $0.d", "{z0}"(<vscale x 8 x bfloat> %value)
  ret void
}

define <vscale x 16 x i8> @output_nxv16i8() {
; POS-LABEL: output_nxv16i8:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.b, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 16 x i8> asm sideeffect "dup $0.b, #0", "={z0}"()
  ret <vscale x 16 x i8> %value
}

define <vscale x 8 x i16> @output_nxv8i16() {
; POS-LABEL: output_nxv8i16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.h, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 8 x i16> asm sideeffect "dup $0.h, #0", "={z0}"()
  ret <vscale x 8 x i16> %value
}

define <vscale x 4 x i32> @output_nxv4i32() {
; POS-LABEL: output_nxv4i32:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.s, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 4 x i32> asm sideeffect "dup $0.s, #0", "={z0}"()
  ret <vscale x 4 x i32> %value
}

define <vscale x 2 x i64> @output_nxv2i64() {
; POS-LABEL: output_nxv2i64:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 2 x i64> asm sideeffect "dup $0.d, #0", "={z0}"()
  ret <vscale x 2 x i64> %value
}

define <vscale x 8 x half> @output_nxv8f16() {
; POS-LABEL: output_nxv8f16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.h, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 8 x half> asm sideeffect "dup $0.h, #0", "={z0}"()
  ret <vscale x 8 x half> %value
}

define <vscale x 4 x float> @output_nxv4f32() {
; POS-LABEL: output_nxv4f32:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.s, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 4 x float> asm sideeffect "dup $0.s, #0", "={z0}"()
  ret <vscale x 4 x float> %value
}

define <vscale x 2 x double> @output_nxv2f64() {
; POS-LABEL: output_nxv2f64:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 2 x double> asm sideeffect "dup $0.d, #0", "={z0}"()
  ret <vscale x 2 x double> %value
}

define <vscale x 8 x bfloat> @output_nxv8bf16() {
; POS-LABEL: output_nxv8bf16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.h, #0 // =0x0
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %value = call <vscale x 8 x bfloat> asm sideeffect "dup $0.h, #0", "={z0}"()
  ret <vscale x 8 x bfloat> %value
}

define <vscale x 16 x i8> @inout_nxv16i8(<vscale x 16 x i8> %value) {
; POS-LABEL: inout_nxv16i8:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 16 x i8> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 16 x i8> %value)
  ret <vscale x 16 x i8> %result
}

define <vscale x 8 x i16> @inout_nxv8i16(<vscale x 8 x i16> %value) {
; POS-LABEL: inout_nxv8i16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 8 x i16> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 8 x i16> %value)
  ret <vscale x 8 x i16> %result
}

define <vscale x 4 x i32> @inout_nxv4i32(<vscale x 4 x i32> %value) {
; POS-LABEL: inout_nxv4i32:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 4 x i32> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 4 x i32> %value)
  ret <vscale x 4 x i32> %result
}

define <vscale x 2 x i64> @inout_nxv2i64(<vscale x 2 x i64> %value) {
; POS-LABEL: inout_nxv2i64:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 2 x i64> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 2 x i64> %value)
  ret <vscale x 2 x i64> %result
}

define <vscale x 8 x half> @inout_nxv8f16(<vscale x 8 x half> %value) {
; POS-LABEL: inout_nxv8f16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 8 x half> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 8 x half> %value)
  ret <vscale x 8 x half> %result
}

define <vscale x 4 x float> @inout_nxv4f32(<vscale x 4 x float> %value) {
; POS-LABEL: inout_nxv4f32:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 4 x float> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 4 x float> %value)
  ret <vscale x 4 x float> %result
}

define <vscale x 2 x double> @inout_nxv2f64(<vscale x 2 x double> %value) {
; POS-LABEL: inout_nxv2f64:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 2 x double> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 2 x double> %value)
  ret <vscale x 2 x double> %result
}

define <vscale x 8 x bfloat> @inout_nxv8bf16(<vscale x 8 x bfloat> %value) {
; POS-LABEL: inout_nxv8bf16:
; POS:       // %bb.0:
; POS-NEXT:    //APP
; POS-NEXT:    mov z0.d, z0.d
; POS-NEXT:    //NO_APP
; POS-NEXT:    ret
  %result = call <vscale x 8 x bfloat> asm sideeffect "mov $0.d, $0.d", "={z0},0"(<vscale x 8 x bfloat> %value)
  ret <vscale x 8 x bfloat> %result
}