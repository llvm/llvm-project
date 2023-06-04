//=== DWARFLinker.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFLinkerImpl.h"

std::unique_ptr<llvm::dwarflinker_parallel::DWARFLinker>
llvm::dwarflinker_parallel::DWARFLinker::createLinker(
    MessageHandlerTy ErrorHandler, MessageHandlerTy WarningHandler,
    TranslatorFuncTy StringsTranslator) {
  return std::make_unique<DWARFLinkerImpl>(ErrorHandler, WarningHandler,
                                           StringsTranslator);
}
