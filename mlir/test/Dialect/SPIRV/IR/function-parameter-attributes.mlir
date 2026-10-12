// RUN: mlir-opt --split-input-file --verify-diagnostics %s

spirv.func @valid(%result: !spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Sret>]},
    %p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.alignment = 16 : i32,
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoAlias>,
                            #spirv.function_parameter_attribute<NoCapture>]}
) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.alignment' must be a 32-bit signless integer attribute}}
spirv.func @alignment_type(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.alignment = 16 : i64}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.alignment' must be a 32-bit signless integer attribute}}
spirv.func @alignment_not_integer(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.alignment = "16"}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.alignment' requires a pointer parameter}}
spirv.func @alignment_not_pointer(%p: i32 {spirv.alignment = 16 : i32}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @not_an_array(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.func_param_attr = #spirv.function_parameter_attribute<NoAlias>}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @empty_array(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.func_param_attr = []}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @integer_element(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.func_param_attr = [4 : i32]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @mixed_elements(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoAlias>, 5 : i32]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{duplicate function parameter attribute 'NoAlias'}}
spirv.func @duplicate(%p: !spirv.ptr<f32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoAlias>,
                            #spirv.function_parameter_attribute<NoAlias>]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{function parameter attributes 'Zext' and 'Sext' are mutually exclusive}}
spirv.func @conflicting_extensions(%p: i32 {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>,
                            #spirv.function_parameter_attribute<Sext>]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{function parameter attribute 'Zext' requires an integer parameter}}
spirv.func @zext_pointer(%p: !spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{function parameter attribute 'Sext' requires an integer parameter}}
spirv.func @sext_float(%p: f32 {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{function parameter attribute 'NoAlias' requires a pointer parameter}}
spirv.func @not_pointer(%p: i32 {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoAlias>]}) "None" {
  spirv.Return
}

// -----

// expected-error @+1 {{function parameter attribute 'Sret' requires the first parameter}}
spirv.func @sret_not_first(%unused: i32, %result: !spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Sret>]}) "None" {
  spirv.Return
}

// -----

spirv.func @unknown_enum(%p: i32 {
    // expected-error @+2 {{expected ::mlir::spirv::FunctionParameterAttribute to be one of}}
    // expected-error @+1 {{failed to parse SPIRV_FunctionParameterAttributeAttr parameter 'value'}}
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Unknown>]}) "None" {
  spirv.Return
}
