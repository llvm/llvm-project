// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  irdl.dialect @testd {
    irdl.type @singleton
  
    irdl.operation @foo {
      %0 = irdl.any
      irdl.operands(in1: %0, in2: %0)
      irdl.results(out1: %0)
    }
  }
}
