//===------------------- AsmViewAnalyzer.cpp - LLVM Advisor ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Analysis/Inspection/AsmViewAnalyzer.h"

#include "llvm/Object/ObjectFile.h"

using namespace llvm;
using namespace llvm::advisor;

Expected<std::unique_ptr<CapabilityResult>>
AsmViewAnalyzer::run(const CapabilityContext &Context) {
  if (Context.ObjectPath.empty())
    return makeUnavailableResult(getCapabilityID(), Context.Unit.ID,
                                 "missing object artifact");

  Expected<object::OwningBinary<object::ObjectFile>> Obj =
      object::ObjectFile::createObjectFile(Context.ObjectPath);
  if (!Obj)
    return Obj.takeError();

  json::Array Sections;
  for (object::SectionRef Sec : Obj->getBinary()->sections()) {
    json::Object Item;
    if (Expected<StringRef> Name = Sec.getName())
      Item["name"] = *Name;
    else
      Item["name"] = "";
    Item["size"] = static_cast<int64_t>(Sec.getSize());
    Item["is_text"] = Sec.isText();
    Sections.push_back(std::move(Item));
  }

  return makeJSONResult(getCapabilityID(), Context.Unit.ID, json::Object{
      {"format", Obj->getBinary()->getFileFormatName()},
      {"sections", std::move(Sections)},
  });
}
