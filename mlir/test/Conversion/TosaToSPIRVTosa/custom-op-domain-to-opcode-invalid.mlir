// RUN: not mlir-opt --split-input-file --pass-pipeline='builtin.module(tosa-to-spirv-tosa{custom-op-domain-to-opcode=test:7})' %s 2>&1 | FileCheck %s

func.func @zero_results(%arg0: tensor<1x16xf32>) {
  // CHECK: 'tosa.custom' op with mapped domain requires at least one result
  tosa.custom %arg0 {domain_name = "test", implementation_attrs = "{}", operator_name = "NoResult"} : (tensor<1x16xf32>) -> ()
  return
}
