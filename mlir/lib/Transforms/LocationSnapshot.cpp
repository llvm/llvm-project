//===- LocationSnapshot.cpp - Location Snapshot Utilities -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/LocationSnapshot.h"
#include "PassDetail.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// If a 'tag' is non-empty, the generated locations are represented as a
/// NameLoc with the given tag as the name, and then fused with the existing
/// locations. Otherwise, the existing locations are replaced.
static void generateLocationsFromIR(raw_ostream &os, StringRef fileName,
                                    Operation *op, const OpPrintingFlags &flags,
                                    StringRef tag) {
  // Print the IR to the stream, and collect the raw line+column information.
  AsmState::LocationMap opToLineCol;
  AsmState state(op, flags, &opToLineCol);
  op->print(os, state);

  Builder builder(op->getContext());
  Optional<StringAttr> tagIdentifier;
  if (!tag.empty())
    tagIdentifier = builder.getStringAttr(tag);

  // Walk and generate new locations for each of the operations.
  StringAttr file = builder.getStringAttr(fileName);
  op->walk([&](Operation *opIt) {
    // Check to see if this operation has a mapped location. Some operations may
    // be elided from the printed form, e.g. the body terminators of some region
    // operations.
    auto it = opToLineCol.find(opIt);
    if (it == opToLineCol.end())
      return;
    const std::pair<unsigned, unsigned> &lineCol = it->second;
    auto newLoc = FileLineColLoc::get(file, lineCol.first, lineCol.second);

    // If we don't have a tag, set the location directly
    if (!tagIdentifier) {
      opIt->setLoc(newLoc);
      return;
    }

    // Otherwise, build a fused location with the existing op loc.
    opIt->setLoc(builder.getFusedLoc(
        {opIt->getLoc(), NameLoc::get(*tagIdentifier, newLoc)}));
  });
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
static LogicalResult generateLocationsFromIR(StringRef fileName, Operation *op,
                                             OpPrintingFlags flags,
                                             StringRef tag) {
  // If a filename wasn't provided, then generate one.
  SmallString<32> filepath(fileName);
  if (filepath.empty()) {
    if (std::error_code error = llvm::sys::fs::createTemporaryFile(
            "mlir_snapshot", "tmp.mlir", filepath)) {
      return op->emitError()
             << "failed to generate temporary file for location snapshot: "
             << error.message();
    }
  }

  // Open the output file for emission.
  std::string error;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(filepath, &error);
  if (!outputFile)
    return op->emitError() << error;

  // Generate the intermediate locations.
  generateLocationsFromIR(outputFile->os(), filepath, op, flags, tag);
  outputFile->keep();
  return success();
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// The generated locations replace the current operation locations.
void mlir::generateLocationsFromIR(raw_ostream &os, StringRef fileName,
                                   Operation *op, OpPrintingFlags flags) {
  ::generateLocationsFromIR(os, fileName, op, flags, /*tag=*/StringRef());
}
/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
LogicalResult mlir::generateLocationsFromIR(StringRef fileName, Operation *op,
                                            OpPrintingFlags flags) {
  return ::generateLocationsFromIR(fileName, op, flags, /*tag=*/StringRef());
}

/// This function generates new locations from the given IR by snapshotting the
/// IR to the given stream, and using the printed locations within that stream.
/// The generated locations are represented as a NameLoc with the given tag as
/// the name, and then fused with the existing locations.
void mlir::generateLocationsFromIR(raw_ostream &os, StringRef fileName,
                                   StringRef tag, Operation *op,
                                   OpPrintingFlags flags) {
  ::generateLocationsFromIR(os, fileName, op, flags, tag);
}
/// This function generates new locations from the given IR by snapshotting the
/// IR to the given file, and using the printed locations within that file. If
/// `filename` is empty, a temporary file is generated instead.
LogicalResult mlir::generateLocationsFromIR(StringRef fileName, StringRef tag,
                                            Operation *op,
                                            OpPrintingFlags flags) {
  return ::generateLocationsFromIR(fileName, op, flags, tag);
}

namespace {
struct LocationSnapshotPass
    : public LocationSnapshotBase<LocationSnapshotPass> {
  LocationSnapshotPass() = default;
  LocationSnapshotPass(OpPrintingFlags flags, StringRef fileName, StringRef tag)
      : flags(flags) {
    this->fileName = fileName.str();
    this->tag = tag.str();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    if (failed(generateLocationsFromIR(fileName, op, OpPrintingFlags(), tag)))
      return signalPassFailure();
  }

  /// The printing flags to use when creating the snapshot.
  OpPrintingFlags flags;
};
} // namespace

std::unique_ptr<Pass> mlir::createLocationSnapshotPass(OpPrintingFlags flags,
                                                       StringRef fileName,
                                                       StringRef tag) {
  return std::make_unique<LocationSnapshotPass>(flags, fileName, tag);
}
std::unique_ptr<Pass> mlir::createLocationSnapshotPass() {
  return std::make_unique<LocationSnapshotPass>();
}
