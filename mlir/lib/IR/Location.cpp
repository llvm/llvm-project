//===- Location.cpp - MLIR Location Classes -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Location.h"
#include "LocationDetail.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;
using namespace mlir::detail;

//===----------------------------------------------------------------------===//
// LocationAttr
//===----------------------------------------------------------------------===//

/// Methods for support type inquiry through isa, cast, and dyn_cast.
bool LocationAttr::classof(Attribute attr) {
  return attr.isa<CallSiteLoc, FileLineColLoc, FusedLoc, NameLoc, OpaqueLoc,
                  UnknownLoc>();
}

//===----------------------------------------------------------------------===//
// CallSiteLoc
//===----------------------------------------------------------------------===//

Location CallSiteLoc::get(Location callee, Location caller) {
  return Base::get(callee->getContext(), callee, caller);
}

Location CallSiteLoc::get(Location name, ArrayRef<Location> frames) {
  assert(!frames.empty() && "required at least 1 call frame");
  Location caller = frames.back();
  for (auto frame : llvm::reverse(frames.drop_back()))
    caller = CallSiteLoc::get(frame, caller);
  return CallSiteLoc::get(name, caller);
}

Location CallSiteLoc::getCallee() const { return getImpl()->callee; }

Location CallSiteLoc::getCaller() const { return getImpl()->caller; }

//===----------------------------------------------------------------------===//
// FileLineColLoc
//===----------------------------------------------------------------------===//

Location FileLineColLoc::get(Identifier filename, unsigned line,
                             unsigned column, MLIRContext *context) {
  return Base::get(context, filename, line, column);
}

Location FileLineColLoc::get(StringRef filename, unsigned line, unsigned column,
                             MLIRContext *context) {
  return get(Identifier::get(filename.empty() ? "-" : filename, context), line,
             column, context);
}

StringRef FileLineColLoc::getFilename() const { return getImpl()->filename; }
unsigned FileLineColLoc::getLine() const { return getImpl()->line; }
unsigned FileLineColLoc::getColumn() const { return getImpl()->column; }

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
    if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      if (fusedLoc.getMetadata() == metadata) {
        // UnknownLoc's have already been removed from FusedLocs so we can
        // simply add all of the internal locations.
        decomposedLocs.insert(fusedLoc.getLocations().begin(),
                              fusedLoc.getLocations().end());
        continue;
      }
    }
    // Otherwise, only add known locations to the set.
    if (!loc.isa<UnknownLoc>())
      decomposedLocs.insert(loc);
  }
  locs = decomposedLocs.getArrayRef();

  // Handle the simple cases of less than two locations.
  if (locs.empty())
    return UnknownLoc::get(context);
  if (locs.size() == 1)
    return locs.front();
  return Base::get(context, locs, metadata);
}

ArrayRef<Location> FusedLoc::getLocations() const {
  return getImpl()->getLocations();
}

Attribute FusedLoc::getMetadata() const { return getImpl()->metadata; }

//===----------------------------------------------------------------------===//
// NameLoc
//===----------------------------------------------------------------------===//

Location NameLoc::get(Identifier name, Location child) {
  assert(!child.isa<NameLoc>() &&
         "a NameLoc cannot be used as a child of another NameLoc");
  return Base::get(child->getContext(), name, child);
}

Location NameLoc::get(Identifier name, MLIRContext *context) {
  return get(name, UnknownLoc::get(context));
}

/// Return the name identifier.
Identifier NameLoc::getName() const { return getImpl()->name; }

/// Return the child location.
Location NameLoc::getChildLoc() const { return getImpl()->child; }

//===----------------------------------------------------------------------===//
// OpaqueLoc
//===----------------------------------------------------------------------===//

Location OpaqueLoc::get(uintptr_t underlyingLocation, TypeID typeID,
                        Location fallbackLocation) {
  return Base::get(fallbackLocation->getContext(), underlyingLocation, typeID,
                   fallbackLocation);
}

uintptr_t OpaqueLoc::getUnderlyingLocation() const {
  return Base::getImpl()->underlyingLocation;
}

TypeID OpaqueLoc::getUnderlyingTypeID() const { return getImpl()->typeID; }

Location OpaqueLoc::getFallbackLocation() const {
  return Base::getImpl()->fallbackLocation;
}
