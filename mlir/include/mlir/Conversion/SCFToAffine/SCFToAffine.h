#ifndef __MLIR_CONVERSION_SCFTOAFFINE_H
#define __MLIR_CONVERSION_SCFTOAFFINE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {

#define GEN_PASS_DECL_RAISESCFTOAFFINEPASS
#define GEN_PASS_DECL_AFFINECFGPASS
#include "mlir/Conversion/Passes.h.inc"

} // namespace mlir

#endif // __MLIR_CONVERSION_SCFTOAFFINE_H