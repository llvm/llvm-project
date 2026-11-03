#ifndef POLYGEIST_BASE
#define POLYGEIST_BASE

include "mlir/IR/OpBase.td"

def Polygeist_Dialect : Dialect {
  let name = "polygeist";
  let cppNamespace = "::mlir::polygeist";
  let summary = "The Polygeist dialect.";
  let description = [{
    The Polygeist dialect contains operations for raising low-level code to higher-level forms, and performing parallel and device transformations (including polyhedral).
  }];
}

#endif // POLYGEIST_BASE
