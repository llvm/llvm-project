//===- DynamicDebugging.h - Dynamic Debugging utils ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_DYNAMIC_DEBUGGING_H
#define LLVM_TRANSFORMS_UTILS_DYNAMIC_DEBUGGING_H

#include "llvm/ADT/StringRef.h"
#include <memory>

namespace llvm {
class Module;

/// Modify \p M to prepare it for dynamic debugging before running
/// optimizations. Return a clone of the module which references global
/// values in \p M. The names of the cloned function definitions in this
//  module are prefixed with “__dyndbg.”.
///
/// \p M requires debug info. Note any instrumentation in \p M will be
/// cloned into the returned module.
///
/// Aliases with external linkage are created for GlobalValues in \p M that
/// have local (non-weak) linkage and are not in a COMDAT. These names of these
/// aliases are appended with \p PromotionSuffix which should be unique to the
/// translation unit. If it's not unique then linking may result in multiple
/// definition errors.
///
/// Input pseudo-code:
/// +----------------------------------------------+
/// | internal global g                            |
/// | internal function loc() { return g }         |
/// | exported function ext() { return loc() }     |
/// +----------------------------------------------+
///
/// M becomes:
/// +----------------------------------------------+
/// | internal global g                            |
/// | internal function loc() { return g }         |
/// | exported function ext() { return loc() }     |<---+
/// | exported loc.promo = alias of loc            |<--+|
/// | exported g.promo = alias of g                |<-+||
/// +----------------------------------------------+  |||
///                                                   |||
/// Returned Module:                                  |||
/// +----------------------------------------------+  |||
/// | external global g.promo                      |--+||
/// | external function loc.promo()                |---+|
/// | external function ext()                      |----+
/// | exported function __dyndbg.loc.promo() {     |
/// |   return g.promo                             |
/// | }                                            |
/// | exported function __dyndbg.ext() {           |
/// |   return return loc.promo()                  |
/// | }                                            |
/// +----------------------------------------------+
std::unique_ptr<Module> prepareForDynamicDebugging(Module *M,
                                                   StringRef PromotionSuffix);
} // namespace llvm

#endif
