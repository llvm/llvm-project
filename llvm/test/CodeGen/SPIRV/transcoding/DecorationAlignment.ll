; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; CHECK-SPIRV: OpDecorate %[[#ALIGNMENT:]] Alignment 16
; CHECK-SPIRV: %[[#ALIGNMENT]] = OpFunctionParameter %[[#]]

%struct._ZTS6Struct.Struct = type { %struct._ZTS11floatStruct.floatStruct, %struct._ZTS11floatStruct.floatStruct }
%struct._ZTS11floatStruct.floatStruct = type { float, float, float, float }

define spir_func void @_ZN3FooC2Ev(%struct._ZTS6Struct.Struct addrspace(4)* align 16 %0) {
  ret void
}
