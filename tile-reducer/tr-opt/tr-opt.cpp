//===- tr-opt.cpp - TileReducer optimizer driver ----------------*- C++ -*-===//
//
// mlir-opt-like driver that registers the TileReducer dialect plus the
// dialects that appear in the canonical source (func, arith).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "TileReducer/TileReducerDialect.h"
#include "TileReducer/TileReducerPasses.h"
#include "TileReducer/TileReducerTransformOps.h"

int main(int argc, char **argv) {
  mlir::tr::registerPasses();
  mlir::transform::registerTransformPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::tr::TileReducerDialect, mlir::affine::AffineDialect,
                  mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::transform::TransformDialect>();
  mlir::func::registerInlinerExtension(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::tr::registerTransformDialectExtension(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TileReducer optimizer driver\n", registry));
}
