; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIRV

; SPIRV: OpCapability Linkage
; SPIRV: OpEntryPoint Kernel %[[#kern:]] "kern"

@ae = available_externally addrspace(1) global i32 79, align 4
; SPIRV-DAG: OpName %[[#ae:]] "ae"

@i1 = addrspace(1) global i32 1, align 4
; SPIRV-DAG: OpName %[[#i1:]] "i1"

@i2 = internal addrspace(1) global i32 2, align 4
; SPIRV-DAG: OpName %[[#i2:]] "i2"

@i3 = addrspace(1) global i32 3, align 4
; SPIRV-DAG: OpName %[[#i3:]] "i3"

@i4 = common addrspace(1) global i32 0, align 4
; SPIRV-DAG: OpName %[[#i4:]] "i4"

@i5 = internal addrspace(1) global i32 0, align 4
; SPIRV-DAG: OpName %[[#i5:]] "i5"

@color_table = addrspace(2) constant [2 x i32] [i32 0, i32 1], align 4
; SPIRV-DAG: OpName %[[#color_table:]] "color_table"

@noise_table = external addrspace(2) constant [256 x i32]
; SPIRV-DAG: OpName %[[#noise_table:]] "noise_table"

@w = addrspace(1) constant i32 0, align 4
; SPIRV-DAG: OpName %[[#w:]] "w"

@f.color_table = internal addrspace(2) constant [2 x i32] [i32 2, i32 3], align 4
; SPIRV-DAG: OpName %[[#f_color_table:]] "f.color_table"

@e = external addrspace(1) global i32
; SPIRV-DAG: OpName %[[#e:]] "e"

@f.t = internal addrspace(1) global i32 5, align 4
; SPIRV-DAG: OpName %[[#f_t:]] "f.t"

@f.stint = internal addrspace(1) global i32 0, align 4
; SPIRV-DAG: OpName %[[#f_stint:]] "f.stint"

@f.inside = internal addrspace(1) global i32 0, align 4
; SPIRV-DAG: OpName %[[#f_inside:]] "f.inside"

@f.b = internal addrspace(2) constant float 1.000000e+00, align 4
; SPIRV-DAG: OpName %[[#f_b:]] "f.b"

; SPIRV-DAG: OpName %[[#foo:]] "foo"
; SPIRV-DAG: OpName %[[#f:]] "f"
; SPIRV-DAG: OpName %[[#g:]] "g"
; SPIRV-DAG: OpName %[[#inline_fun:]] "inline_fun"

; SPIRV-DAG: OpDecorate %[[#ae]] LinkageAttributes "ae" Import
; SPIRV-DAG: OpDecorate %[[#e]] LinkageAttributes "e" Import
; SPIRV-DAG: OpDecorate %[[#f]] LinkageAttributes "f" Export
; SPIRV-DAG: OpDecorate %[[#w]] LinkageAttributes "w" Export
; SPIRV-DAG: OpDecorate %[[#i1]] LinkageAttributes "i1" Export
; SPIRV-DAG: OpDecorate %[[#i3]] LinkageAttributes "i3" Export
; SPIRV-DAG: OpDecorate %[[#i4]] LinkageAttributes "i4" Export
; SPIRV-DAG: OpDecorate %[[#foo]] LinkageAttributes "foo" Import
; SPIRV-DAG: OpDecorate %[[#inline_fun]] LinkageAttributes "inline_fun" Export
; SPIRV-DAG: OpDecorate %[[#color_table]] LinkageAttributes "color_table" Export
; SPIRV-DAG: OpDecorate %[[#noise_table]] LinkageAttributes "noise_table" Import

; SPIRV: %[[#foo]] = OpFunction %[[#]]
declare spir_func void @foo()

; SPIRV: %[[#f]] = OpFunction %[[#]]
define spir_func void @f() {
entry:
  %q = alloca i32, align 4
  %r = alloca i32, align 4
  %0 = load i32, i32 addrspace(1)* @i2, align 4
  store i32 %0, i32* %q, align 4
  %1 = load i32, i32 addrspace(1)* @i3, align 4
  store i32 %1, i32 addrspace(1)* @i5, align 4
  %2 = load i32, i32 addrspace(1)* @e, align 4
  store i32 %2, i32* %r, align 4
  %3 = load i32, i32 addrspace(2)* getelementptr inbounds ([256 x i32], [256 x i32] addrspace(2)* @noise_table, i32 0, i32 0), align 4
  store i32 %3, i32* %r, align 4
  %4 = load i32, i32 addrspace(2)* getelementptr inbounds ([2 x i32], [2 x i32] addrspace(2)* @f.color_table, i32 0, i32 0), align 4
  store i32 %4, i32* %r, align 4
  %call = call spir_func i32 @g()
  call spir_func void @inline_fun()
  ret void
}

; SPIRV: %[[#g]] = OpFunction %[[#]]
define internal spir_func i32 @g() {
entry:
  call spir_func void @foo()
  ret i32 25
}

; SPIRV: %[[#inline_fun]] = OpFunction %[[#]]
;; "linkonce_odr" is lost in translation !
define linkonce_odr spir_func void @inline_fun() {
entry:
  %t = alloca i32 addrspace(1)*, align 4
  store i32 addrspace(1)* @i1, i32 addrspace(1)** %t, align 4
  ret void
}

; SPIRV: %[[#kern]] = OpFunction %[[#]]
define spir_kernel void @kern() {
entry:
  call spir_func void @f()
  ret void
}
