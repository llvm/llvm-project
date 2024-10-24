//===- Location.cpp - MLIR Location Classes -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
/// Tablegen Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/IR/BuiltinLocationAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// BuiltinDialect
//===----------------------------------------------------------------------===//

void BuiltinDialect::registerLocationAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/IR/BuiltinLocationAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// LocationAttr
//===----------------------------------------------------------------------===//

WalkResult LocationAttr::walk(function_ref<WalkResult(Location)> walkFn) {
  AttrTypeWalker walker;
  // Walk locations, but skip any other attribute.
  walker.addWalk([&](Attribute attr) {
    if (auto loc = llvm::dyn_cast<LocationAttr>(attr))
      return walkFn(loc);

    return WalkResult::skip();
  });
  return walker.walk<WalkOrder::PreOrder>(*this);
}

/// Methods for support type inquiry through isa, cast, and dyn_cast.
bool LocationAttr::classof(Attribute attr) {
  return attr.hasTrait<AttributeTrait::IsLocation>();
}

//===----------------------------------------------------------------------===//
// CallSiteLoc
//===----------------------------------------------------------------------===//

CallSiteLoc CallSiteLoc::get(Location name, ArrayRef<Location> frames) {
  assert(!frames.empty() && "required at least 1 call frame");
  Location caller = frames.back();
  for (auto frame : llvm::reverse(frames.drop_back()))
    caller = CallSiteLoc::get(frame, caller);
  return CallSiteLoc::get(name, caller);
}

//===----------------------------------------------------------------------===//
// FusedLoc
//===----------------------------------------------------------------------===//

Location FusedLoc::get(ArrayRef<Location> locs, Attribute metadata,
                       MLIRContext *context) {
  // Unique the set of locations to be fused.
  llvm::SmallSetVector<Location, 4> decomposedLocs;
  for (auto loc : locs) {
    // If the location is a fused location we decompose it if it has no
    // metadata or the metadata is the same as the top level metadata.
    if (auto fusedLoc = llvm::dyn_cast<FusedLoc>(loc)) {
      if (fusedLoc.getMetadata() == metadata) {
        // UnknownLoc's have already been removed from FusedLocs so we can
        // simply add all of the internal locations.
        decomposedLocs.insert(fusedLoc.getLocations().begin(),
                              fusedLoc.getLocations().end());
        continue;
      }
    }
    // Otherwise, only add known locations to the set.
    if (!llvm::isa<UnknownLoc>(loc))
      decomposedLocs.insert(loc);
  }
  locs = decomposedLocs.getArrayRef();

  // Handle the simple cases of less than two locations. Ensure the metadata (if
  // provided) is not dropped.
  if (locs.empty()) {
    if (!metadata)
      return UnknownLoc::get(context);
    // TODO: Investigate ASAN failure when using implicit conversion from
    // Location to ArrayRef<Location> below.
    return Base::get(context, ArrayRef<Location>{UnknownLoc::get(context)},
                     metadata);
  }
  if (locs.size() == 1 && !metadata)
    return locs.front();

  return Base::get(context, locs, metadata);
}
