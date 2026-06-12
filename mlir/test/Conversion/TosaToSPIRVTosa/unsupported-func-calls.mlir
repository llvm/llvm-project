// RUN: mlir-opt --split-input-file --tosa-to-spirv-tosa --verify-diagnostics %s

func.func @direct_call(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  // expected-error@below {{'func.call' op is not supported in TOSA to SPIR-V Graph conversion; inline calls before running this pass}}
  %0 = func.call @direct_call(%arg0) : (tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

func.func @indirect_call(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  %callee = func.constant @indirect_call : (tensor<1xi8>) -> tensor<1xi8>
  // expected-error@below {{'func.call_indirect' op is not supported in TOSA to SPIR-V Graph conversion; inline calls before running this pass}}
  %0 = func.call_indirect %callee(%arg0) : (tensor<1xi8>) -> tensor<1xi8>
  return %0 : tensor<1xi8>
}

// -----

func.func @nested_func(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  builtin.module {
    // expected-error@below {{'func.func' op nesting is not supported in TOSA to SPIR-V Graph conversion}}
    func.func @nested(%arg1: tensor<1xi8>) -> tensor<1xi8> {
      return %arg1 : tensor<1xi8>
    }
  }
  return %arg0 : tensor<1xi8>
}
