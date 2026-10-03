//===- TileReducerPasses.h - TileReducer passes -----------------*- C++ -*-===//

#ifndef TILE_REDUCER_TILEREDUCERPASSES_H
#define TILE_REDUCER_TILEREDUCERPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class ModuleOp;
class RewritePatternSet;

namespace func {
class FuncOp;
} // namespace func

namespace tr {
#define GEN_PASS_DECL
#include "TileReducer/TileReducerPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "TileReducer/TileReducerPasses.h.inc"

void populateFoldAddZeroPatterns(RewritePatternSet &patterns);
void populateRecognizeLoadReducePatterns(RewritePatternSet &patterns);
} // namespace tr
} // namespace mlir

#endif // TILE_REDUCER_TILEREDUCERPASSES_H
