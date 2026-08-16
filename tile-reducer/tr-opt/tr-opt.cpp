//===- tr-opt.cpp - TileReducer optimizer driver ----------------*- C++ -*-===//
//
// mlir-opt-like driver that registers the TileReducer dialect plus the
// dialects that appear in the canonical source (func, arith).
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "TileReducer/TileReducerDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::tr::TileReducerDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "TileReducer optimizer driver\n", registry));
}
