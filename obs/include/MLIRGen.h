
#ifndef OBS_MLIRGEN_H
#define OBS_MLIRGEN_H

#include <memory>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>

namespace mlir{
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
} // namespace mlir

namespace obs {
class ModuleAST;
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} //namespace obs

#endif //OBS_MLIRGEN_H