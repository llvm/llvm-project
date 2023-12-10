//=== DWARFLinker.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerImpl.h"
#include "DependencyTracker.h"

using namespace llvm;
using namespace dwarflinker;
using namespace dwarflinker_parallel;

std::unique_ptr<DWARFLinker>
llvm::dwarflinker::dwarflinker_parallel::DWARFLinker::createLinker(
    MessageHandlerTy ErrorHandler, MessageHandlerTy WarningHandler,
    TranslatorFuncTy StringsTranslator) {
  return std::make_unique<DWARFLinkerImpl>(ErrorHandler, WarningHandler,
                                           StringsTranslator);
}
