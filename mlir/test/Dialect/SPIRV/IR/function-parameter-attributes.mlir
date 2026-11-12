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

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @return_not_an_array(%v: i32) -> (i32 {
    spirv.func_param_attr = #spirv.function_parameter_attribute<Zext>}) "None" {
  spirv.ReturnValue %v : i32
}

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @return_empty_array(%v: i32) -> (i32 {
    spirv.func_param_attr = []}) "None" {
  spirv.ReturnValue %v : i32
}

// -----

// expected-error @+1 {{'spirv.func_param_attr' must be a non-empty array of spirv::FunctionParameterAttributeAttr}}
spirv.func @return_integer_element(%v: i32) -> (i32 {
    spirv.func_param_attr = [0 : i32]}) "None" {
  spirv.ReturnValue %v : i32
}

// -----

// expected-error @+1 {{duplicate function parameter attribute 'Zext'}}
spirv.func @return_duplicate(%v: i32) -> (i32 {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>,
                            #spirv.function_parameter_attribute<Zext>]}) "None" {
  spirv.ReturnValue %v : i32
}

// -----

// expected-error @+1 {{function parameter attributes 'Zext' and 'Sext' are mutually exclusive}}
spirv.func @return_conflicting_extensions(%v: i32) -> (i32 {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>,
                            #spirv.function_parameter_attribute<Sext>]}) "None" {
  spirv.ReturnValue %v : i32
}

// -----

// expected-error @+1 {{function parameter attribute 'Zext' requires an integer result}}
spirv.func @return_zext_pointer(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Zext>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{function parameter attribute 'Sext' requires an integer result}}
spirv.func @return_sext_float(%v: f32) -> (f32 {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Sext>]}) "None" {
  spirv.ReturnValue %v : f32
}

// -----

// expected-error @+1 {{function parameter attribute 'ByVal' is not valid on function results}}
spirv.func @return_byval(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<ByVal>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{function parameter attribute 'Sret' is not valid on function results}}
spirv.func @return_sret(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<Sret>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{function parameter attribute 'NoAlias' is not valid on function results}}
spirv.func @return_noalias(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoAlias>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{function parameter attribute 'NoCapture' is not valid on function results}}
spirv.func @return_nocapture(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoCapture>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{function parameter attribute 'NoWrite' is not valid on function results}}
spirv.func @return_nowrite(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoWrite>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{function parameter attribute 'NoReadWrite' is not valid on function results}}
spirv.func @return_noreadwrite(%v: !spirv.ptr<i32, CrossWorkgroup>) -> (!spirv.ptr<i32, CrossWorkgroup> {
    spirv.func_param_attr = [#spirv.function_parameter_attribute<NoReadWrite>]}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}

// -----

// expected-error @+1 {{found unsupported 'spirv.alignment' attribute on function result}}
spirv.func @return_alignment(%v: !spirv.ptr<i32, CrossWorkgroup>) ->
    (!spirv.ptr<i32, CrossWorkgroup> {spirv.alignment = 4 : i32}) "None" {
  spirv.ReturnValue %v : !spirv.ptr<i32, CrossWorkgroup>
}
