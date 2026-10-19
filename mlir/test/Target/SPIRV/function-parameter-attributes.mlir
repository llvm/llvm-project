// RUN: mlir-translate --no-implicit-module --test-spirv-roundtrip --split-input-file %s | FileCheck %s
// RUN: %if spirv-tools %{ rm -rf %t %}
// RUN: %if spirv-tools %{ mkdir %t %}
// RUN: %if spirv-tools %{ mlir-translate --no-implicit-module --serialize-spirv --split-input-file --spirv-save-validation-files-with-prefix=%t/module %s %}
// RUN: %if spirv-tools %{ spirv-val %t %}

// All core FunctionParameterAttribute enumerants, including several decorations
// on one parameter. Alignment and existing decorations must survive together.
spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Kernel, Linkage, Int8], []> {
  // CHECK-LABEL: spirv.func @parameters(
  // CHECK-SAME: !spirv.ptr<!spirv.struct<(i32)>, CrossWorkgroup> {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sret>]}
  // CHECK-SAME: i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]}
  // CHECK-SAME: i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}
  // CHECK-SAME: !spirv.ptr<f32, CrossWorkgroup> {spirv.alignment = 16 : i32, spirv.decoration = #spirv.decoration<Restrict>, spirv.func_param_attr = [#spirv.function_parameter_attribute<ByVal>, #spirv.function_parameter_attribute<NoAlias>, #spirv.function_parameter_attribute<NoCapture>, #spirv.function_parameter_attribute<NoWrite>]}
  // CHECK-SAME: !spirv.ptr<f32, CrossWorkgroup> {spirv.func_param_attr = [#spirv.function_parameter_attribute<NoReadWrite>]}
  spirv.func @parameters(
      %result: !spirv.ptr<!spirv.struct<(i32)>, CrossWorkgroup> {
        spirv.func_param_attr = [#spirv.function_parameter_attribute<Sret>]},
      %unsigned: i8 {
        spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]},
      %signed: i8 {
        spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]},
      %input: !spirv.ptr<f32, CrossWorkgroup> {
        spirv.alignment = 16 : i32,
        spirv.decoration = #spirv.decoration<Restrict>,
        spirv.func_param_attr = [#spirv.function_parameter_attribute<ByVal>,
                                #spirv.function_parameter_attribute<NoAlias>,
                                #spirv.function_parameter_attribute<NoCapture>,
                                #spirv.function_parameter_attribute<NoWrite>]},
      %unused: !spirv.ptr<f32, CrossWorkgroup> {
        spirv.func_param_attr = [#spirv.function_parameter_attribute<NoReadWrite>]}
  ) "None" {
    spirv.Return
  }
}

// -----

// Return attributes decorate the OpFunction result id, not a parameter id.
// Preserve them on both definitions and declarations.
spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Kernel, Linkage, Int8, Int16], []> {
  // CHECK-LABEL: spirv.func @return_zext(
  // CHECK-SAME: i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}
  // CHECK-SAME: -> (i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]})
  spirv.func @return_zext(%v: i8 {
      spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}) ->
      (i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]}) "None" {
    spirv.ReturnValue %v : i8
  }

  // CHECK-LABEL: spirv.func @return_sext(
  // CHECK-SAME: -> (i16 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]})
  spirv.func @return_sext(%v: i16) ->
      (i16 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}) "None" {
    spirv.ReturnValue %v : i16
  }
}

// -----

spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Kernel, Linkage, Int8, Int16], []> {
  // CHECK-DAG: spirv.func @external_zext() -> (i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]})
  spirv.func @external_zext() ->
      (i8 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]}) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "external_zext", linkage_type = <Import>>
  }

  // CHECK-DAG: spirv.func @external_sext() -> (i16 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]})
  spirv.func @external_sext() ->
      (i16 {spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "external_sext", linkage_type = <Import>>
  }
}

// -----

// Alignment alone must not be rejected as an unsupported argument decoration.
// Its literal is unsigned: the high bit must not be lost in either direction.
// The signless i32 attribute prints the same bits using a signed spelling.
spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Kernel, Linkage], []> {
  // CHECK-LABEL: spirv.func @alignment(
  // CHECK-SAME: !spirv.ptr<f32, CrossWorkgroup> {spirv.alignment = 4 : i32}
  // CHECK-SAME: !spirv.ptr<f32, CrossWorkgroup> {spirv.alignment = -2147483648 : i32}
  spirv.func @alignment(
      %a: !spirv.ptr<f32, CrossWorkgroup> {spirv.alignment = 4 : i32},
      %b: !spirv.ptr<f32, CrossWorkgroup> {spirv.alignment = 2147483648 : i32}
  ) "None" {
    spirv.Return
  }
}

// -----

// Imported function declarations carry the same parameter attributes as bodies.
spirv.module Physical64 OpenCL requires #spirv.vce<v1.0, [Addresses, Kernel, Linkage], []> {
  // CHECK-LABEL: spirv.func @external(
  // CHECK-SAME: !spirv.ptr<f32, CrossWorkgroup> {spirv.alignment = 8 : i32, spirv.func_param_attr = [#spirv.function_parameter_attribute<NoWrite>, #spirv.function_parameter_attribute<NoCapture>]}
  spirv.func @external(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.alignment = 8 : i32,
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoWrite>,
                            #spirv.function_parameter_attribute<NoCapture>]
  }) "None" attributes {
    linkage_attributes = #spirv.linkage_attributes<linkage_name = "external", linkage_type = <Import>>
  }
}
