//===- WasmObjcopy.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WasmObjcopy.h"
#include "Buffer.h"
#include "CopyConfig.h"
#include "Object.h"
#include "Reader.h"
#include "Writer.h"
#include "llvm-objcopy.h"
#include "llvm/Support/Errc.h"

namespace llvm {
namespace objcopy {
namespace wasm {

using namespace object;

static Error handleArgs(const CopyConfig &Config, Object &Obj) {
  if (!Config.AddGnuDebugLink.empty() || !Config.BuildIdLinkDir.empty() ||
      Config.BuildIdLinkInput || Config.BuildIdLinkOutput ||
      Config.ExtractPartition || !Config.SplitDWO.empty() ||
      !Config.SymbolsPrefix.empty() || !Config.AllocSectionsPrefix.empty() ||
      Config.DiscardMode != DiscardType::None || Config.NewSymbolVisibility ||
      !Config.SymbolsToAdd.empty() || !Config.RPathToAdd.empty() ||
      !Config.OnlySection.empty() || !Config.SymbolsToGlobalize.empty() ||
      !Config.SymbolsToKeep.empty() || !Config.SymbolsToLocalize.empty() ||
      !Config.SymbolsToRemove.empty() ||
      !Config.UnneededSymbolsToRemove.empty() ||
      !Config.SymbolsToWeaken.empty() || !Config.SymbolsToKeepGlobal.empty() ||
      !Config.SectionsToRename.empty() || !Config.SetSectionAlignment.empty() ||
      !Config.SetSectionFlags.empty() || !Config.SymbolsToRename.empty() ||
      !Config.ToRemove.empty() || !Config.DumpSection.empty() ||
      !Config.AddSection.empty()) {
    return createStringError(
        llvm::errc::invalid_argument,
        "no flags are supported yet, only basic copying is allowed");
  }
  return Error::success();
}

Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::WasmObjectFile &In, Buffer &Out) {
  Reader TheReader(In);
  Expected<std::unique_ptr<Object>> ObjOrErr = TheReader.create();
  if (!ObjOrErr)
    return createFileError(Config.InputFilename, ObjOrErr.takeError());
  Object *Obj = ObjOrErr->get();
  assert(Obj && "Unable to deserialize Wasm object");
  if (Error E = handleArgs(Config, *Obj))
    return createFileError(Config.InputFilename, std::move(E));
  Writer TheWriter(*Obj, Out);
  if (Error E = TheWriter.write())
    return createFileError(Config.OutputFilename, std::move(E));
  return Error::success();
}

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm
