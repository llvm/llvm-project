// RUN: mlir-opt %s | mlir-opt | FileCheck %s

module {
  irdl.dialect @testd {
    irdl.type @singleton
  
    irdl.operation @any {
      %0 = irdl.any
      irdl.results(%0)
    }
  }
}
