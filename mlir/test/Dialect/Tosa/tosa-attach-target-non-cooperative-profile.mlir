// RUN: mlir-opt %s -verify-diagnostics -tosa-attach-target="profiles=pro_fp extensions=int16"

// expected-error@below {{use of extension 'int16' requires any of profiles: [pro_int] to be enabled in the target}}
module {
    func.func @test_simple(%arg0 : tensor<1x1x1x1xf32>, %arg1 : tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32> {
        %1 = tosa.add %arg0, %arg1 : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>) -> tensor<1x1x1x1xf32>
        return %1 : tensor<1x1x1x1xf32>
    }
}
