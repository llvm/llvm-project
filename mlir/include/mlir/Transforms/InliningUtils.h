//===- InliningUtils.h - Inliner utilities ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines interfaces for various inlining utility methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_INLININGUTILS_H
#define MLIR_TRANSFORMS_INLININGUTILS_H

#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InlinerInterface.h"
#include <optional>

namespace mlir {

class CallableOpInterface;
class CallOpInterface;

//===----------------------------------------------------------------------===//
// Inline Methods.
//===----------------------------------------------------------------------===//

/// This function inlines a region, 'src', into another. This function returns
/// failure if it is not possible to inline this function. If the function
/// returned failure, then no changes to the module have been made.
///
/// The provided 'inlinePoint' must be within a region, and corresponds to the
/// location where the 'src' region should be inlined. 'mapping' contains any
/// remapped operands that are used within the region, and *must* include
/// remappings for the entry arguments to the region. 'resultsToReplace'
/// corresponds to any results that should be replaced by terminators within the
/// inlined region. 'regionResultTypes' specifies the expected return types of
/// the terminators in the region. 'inlineLoc' is an optional Location that, if
/// provided, will be used to update the inlined operations' location
/// information. 'shouldCloneInlinedRegion' corresponds to whether the source
/// region should be cloned into the 'inlinePoint' or spliced directly.
LogicalResult inlineRegion(InlinerInterface &interface, Region *src,
                           Operation *inlinePoint, IRMapping &mapper,
                           ValueRange resultsToReplace,
                           TypeRange regionResultTypes,
                           std::optional<Location> inlineLoc = std::nullopt,
                           bool shouldCloneInlinedRegion = true);
LogicalResult inlineRegion(InlinerInterface &interface, Region *src,
                           Block *inlineBlock, Block::iterator inlinePoint,
                           IRMapping &mapper, ValueRange resultsToReplace,
                           TypeRange regionResultTypes,
                           std::optional<Location> inlineLoc = std::nullopt,
                           bool shouldCloneInlinedRegion = true);

/// This function is an overload of the above 'inlineRegion' that allows for
/// providing the set of operands ('inlinedOperands') that should be used
/// in-favor of the region arguments when inlining.
LogicalResult inlineRegion(InlinerInterface &interface, Region *src,
                           Operation *inlinePoint, ValueRange inlinedOperands,
                           ValueRange resultsToReplace,
                           std::optional<Location> inlineLoc = std::nullopt,
                           bool shouldCloneInlinedRegion = true);
LogicalResult inlineRegion(InlinerInterface &interface, Region *src,
                           Block *inlineBlock, Block::iterator inlinePoint,
                           ValueRange inlinedOperands,
                           ValueRange resultsToReplace,
                           std::optional<Location> inlineLoc = std::nullopt,
                           bool shouldCloneInlinedRegion = true);

/// This function inlines a given region, 'src', of a callable operation,
/// 'callable', into the location defined by the given call operation. This
/// function returns failure if inlining is not possible, success otherwise. On
/// failure, no changes are made to the module. 'shouldCloneInlinedRegion'
/// corresponds to whether the source region should be cloned into the 'call' or
/// spliced directly.
LogicalResult inlineCall(InlinerInterface &interface, CallOpInterface call,
                         CallableOpInterface callable, Region *src,
                         bool shouldCloneInlinedRegion = true);

} // namespace mlir

#endif // MLIR_TRANSFORMS_INLININGUTILS_H
