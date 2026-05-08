//===--- ObjectAnalyzer.cpp - LLVM Advisor -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Binary/ObjectAnalyzer.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
ObjectAnalyzer::run(const CapabilityContext &Context) {
  if (Context.ObjectPath.empty())
    return makeUnavailableResult(getCapabilityID(), Context.Unit.ID,
                                 "missing object artifact");

  Expected<object::OwningBinary<object::ObjectFile>> Binary =
      object::ObjectFile::createObjectFile(Context.ObjectPath);
  if (!Binary)
    return Binary.takeError();

  const object::ObjectFile *Obj = Binary->getBinary();
  if (!Obj)
    return createStringError(inconvertibleErrorCode(),
                             "object file is unavailable");
  uint64_t Sections = 0;
  uint64_t Symbols = 0;
  for (object::SectionRef Section : Obj->sections()) {
    (void)Section;
    ++Sections;
  }
  for (object::SymbolRef Symbol : Obj->symbols()) {
    (void)Symbol;
    ++Symbols;
  }

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"format", Obj->getFileFormatName()},
      {"arch", Triple::getArchTypeName(Obj->getArch())},
      {"sections", static_cast<int64_t>(Sections)},
      {"symbols", static_cast<int64_t>(Symbols)}});
}
