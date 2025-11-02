//===-- InstrumentorStubPrinter.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/Instrumentor.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/StringSaver.h"

#include <string>

namespace llvm {
namespace instrumentor {

void writeConfigToJSON(InstrumentationConfig &IConf, StringRef OutputFile,
                       LLVMContext &Ctx) {
  if (OutputFile.empty())
    return;

  std::error_code EC;
  raw_fd_stream OS(OutputFile, EC);
  if (EC) {
    Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("failed to open instrumentor configuration file for writing: ") +
            EC.message(),
        DS_Warning));
    return;
  }

  json::OStream J(OS, 2);
  J.objectBegin();

  J.attributeBegin("configuration");
  J.objectBegin();
  for (auto *BaseCO : IConf.BaseConfigurationOptions) {
    switch (BaseCO->Kind) {
    case BaseConfigurationOption::STRING:
      J.attribute(BaseCO->Name, BaseCO->getString());
      break;
    case BaseConfigurationOption::BOOLEAN:
      J.attribute(BaseCO->Name, BaseCO->getBool());
      break;
    }
    if (!BaseCO->Description.empty())
      J.attribute(std::string(BaseCO->Name) + ".description",
                  BaseCO->Description);
  }
  J.objectEnd();
  J.attributeEnd();

  for (unsigned KindVal = 0; KindVal <= InstrumentationLocation::Last;
       ++KindVal) {
    auto Kind = InstrumentationLocation::KindTy(KindVal);

    auto &KindChoices = IConf.IChoices[Kind];
    if (KindChoices.empty())
      continue;

    J.attributeBegin(InstrumentationLocation::getKindStr(Kind));
    J.objectBegin();
    for (auto &ChoiceIt : KindChoices) {
      J.attributeBegin(ChoiceIt.getKey());
      J.objectBegin();
      J.attribute("enabled", ChoiceIt.second->Enabled);
      for (auto &ArgIt : ChoiceIt.second->IRTArgs) {
        J.attribute(ArgIt.Name, ArgIt.Enabled);
        if ((ArgIt.Flags & IRTArg::REPLACABLE) ||
            (ArgIt.Flags & IRTArg::REPLACABLE_CUSTOM))
          J.attribute(std::string(ArgIt.Name) + ".replace", true);
        if (!ArgIt.Description.empty())
          J.attribute(std::string(ArgIt.Name) + ".description",
                      ArgIt.Description);
      }
      J.objectEnd();
      J.attributeEnd();
    }
    J.objectEnd();
    J.attributeEnd();
  }

  J.objectEnd();
}

bool readConfigFromJSON(InstrumentationConfig &IConf, StringRef InputFile,
                        LLVMContext &Ctx) {
  if (InputFile.empty())
    return true;

  std::error_code EC;
  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(InputFile);
  if (std::error_code EC = BufferOrErr.getError()) {
    Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("failed to open instrumentor configuration file for reading: ") +
            EC.message(),
        DS_Warning));
    return false;
  }
  auto Buffer = std::move(BufferOrErr.get());
  json::Path::Root NullRoot;
  auto Parsed = json::parse(Buffer->getBuffer());
  if (!Parsed) {
    Ctx.diagnose(DiagnosticInfoInstrumentation(
        Twine("failed to parse instrumentor configuration file: ") +
            toString(Parsed.takeError()),
        DS_Warning));
    return false;
  }
  auto *Config = Parsed->getAsObject();
  if (!Config) {
    Ctx.diagnose(DiagnosticInfoInstrumentation(
        "failed to parse instrumentor configuration file, expected an object "
        "'{ ... }'",
        DS_Warning));
    return false;
  }

  StringMap<BaseConfigurationOption *> BCOMap;
  for (auto *BO : IConf.BaseConfigurationOptions)
    BCOMap[BO->Name] = BO;

  SmallPtrSet<InstrumentationOpportunity *, 32> SeenIOs;
  for (auto &It : *Config) {
    auto *Obj = It.second.getAsObject();
    if (!Obj) {
      Ctx.diagnose(DiagnosticInfoInstrumentation(
          "malformed JSON configuration, expected an object", DS_Warning));
      continue;
    }
    if (It.first == "configuration") {
      for (auto &ObjIt : *Obj) {
        if (auto *BO = BCOMap.lookup(ObjIt.first)) {
          switch (BO->Kind) {
          case BaseConfigurationOption::STRING:
            if (auto V = ObjIt.second.getAsString()) {
              BO->setString(IConf.SS.save(*V));
            } else {
              Ctx.diagnose(DiagnosticInfoInstrumentation(
                  Twine("configuration key '") + ObjIt.first.str() +
                      Twine("' expects a string, value ignored"),
                  DS_Warning));
            }
            break;
          case BaseConfigurationOption::BOOLEAN:
            if (auto V = ObjIt.second.getAsBoolean())
              BO->setBool(*V);
            else {
              Ctx.diagnose(DiagnosticInfoInstrumentation(
                  Twine("configuration key '") + ObjIt.first.str() +
                      Twine("' expects a boolean, value ignored"),
                  DS_Warning));
            }
            break;
          }
        } else if (!StringRef(ObjIt.first).ends_with(".description")) {
          Ctx.diagnose(DiagnosticInfoInstrumentation(
              Twine("configuration key '") + ObjIt.first.str() +
                  Twine("' not found and ignored"),
              DS_Warning));
        }
      }
      continue;
    }

    auto &IChoiceMap =
        IConf.IChoices[InstrumentationLocation::getKindFromStr(It.first)];
    for (auto &ObjIt : *Obj) {
      auto *InnerObj = ObjIt.second.getAsObject();
      if (!InnerObj) {
        Ctx.diagnose(DiagnosticInfoInstrumentation(
            "malformed JSON configuration, expected an object", DS_Warning));
        continue;
      }
      auto *IO = IChoiceMap.lookup(ObjIt.first);
      if (!IO) {
        Ctx.diagnose(DiagnosticInfoInstrumentation(
            Twine("malformed JSON configuration, expected an object matching "
                  "an instrumentor choice, got ") +
                ObjIt.first.str(),
            DS_Warning));
        continue;
      }
      SeenIOs.insert(IO);
      StringMap<bool> ValueMap, ReplaceMap;
      for (auto &InnerObjIt : *InnerObj) {
        auto Name = StringRef(InnerObjIt.first);
        if (Name.consume_back(".replace"))
          ReplaceMap[Name] = InnerObjIt.second.getAsBoolean().value_or(false);
        else
          ValueMap[Name] = InnerObjIt.second.getAsBoolean().value_or(false);
      }
      IO->Enabled = ValueMap["enabled"];
      for (auto &IRArg : IO->IRTArgs) {
        IRArg.Enabled = ValueMap[IRArg.Name];
        if (!ReplaceMap.lookup(IRArg.Name)) {
          IRArg.Flags &= ~IRTArg::REPLACABLE;
          IRArg.Flags &= ~IRTArg::REPLACABLE_CUSTOM;
        }
      }
    }
  }

  for (auto &IChoiceMap : IConf.IChoices)
    for (auto &It : IChoiceMap)
      if (!SeenIOs.count(It.second))
        It.second->Enabled = false;

  return true;
}

} // end namespace instrumentor
} // end namespace llvm
