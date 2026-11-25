// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "CVNSS/CVNSS.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::cvnss::CVNSSDialect>();
  mlir::registerAllPasses();
  mlir::cvnss::registerCVNSSPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CVNSS MLIR optimizer\n", registry));
}
