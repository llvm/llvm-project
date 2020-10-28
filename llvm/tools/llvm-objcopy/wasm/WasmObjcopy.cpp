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
#include "llvm/Support/Errc.h"

namespace llvm {
namespace objcopy {
namespace wasm {

using namespace object;

static Error dumpSectionToFile(StringRef SecName, StringRef Filename,
                               Object &Obj) {
  for (const Section &Sec : Obj.Sections) {
    if (Sec.Name == SecName) {
      ArrayRef<uint8_t> Contents = Sec.Contents;
      Expected<std::unique_ptr<FileOutputBuffer>> BufferOrErr =
          FileOutputBuffer::create(Filename, Contents.size());
      if (!BufferOrErr)
        return BufferOrErr.takeError();
      std::unique_ptr<FileOutputBuffer> Buf = std::move(*BufferOrErr);
      std::copy(Contents.begin(), Contents.end(), Buf->getBufferStart());
      if (Error E = Buf->commit())
        return E;
      return Error::success();
    }
  }
  return createStringError(errc::invalid_argument, "section '%s' not found",
                           SecName.str().c_str());
}
static Error handleArgs(const CopyConfig &Config, Object &Obj) {
  // Only support AddSection, DumpSection, RemoveSection for now.
  for (StringRef Flag : Config.DumpSection) {
    StringRef SecName;
    StringRef FileName;
    std::tie(SecName, FileName) = Flag.split("=");
    if (Error E = dumpSectionToFile(SecName, FileName, Obj))
      return createFileError(FileName, std::move(E));
  }

  Obj.removeSections([&Config](const Section &Sec) {
    if (Config.ToRemove.matches(Sec.Name))
      return true;
    return false;
  });

  for (StringRef Flag : Config.AddSection) {
    StringRef SecName, FileName;
    std::tie(SecName, FileName) = Flag.split("=");
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
        MemoryBuffer::getFile(FileName);
    if (!BufOrErr)
      return createFileError(FileName, errorCodeToError(BufOrErr.getError()));
    Section Sec;
    Sec.SectionType = llvm::wasm::WASM_SEC_CUSTOM;
    Sec.Name = SecName;
    std::unique_ptr<MemoryBuffer> Buf = std::move(*BufOrErr);
    Sec.Contents = makeArrayRef<uint8_t>(
        reinterpret_cast<const uint8_t *>(Buf->getBufferStart()),
        Buf->getBufferSize());
    Obj.addSectionWithOwnedContents(Sec, std::move(Buf));
  }

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
      !Config.SetSectionFlags.empty() || !Config.SymbolsToRename.empty()) {
    return createStringError(
        llvm::errc::invalid_argument,
        "only add-section, dump-section, and remove-section are supported");
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
    return E;
  Writer TheWriter(*Obj, Out);
  if (Error E = TheWriter.write())
    return createFileError(Config.OutputFilename, std::move(E));
  return Error::success();
}

} // end namespace wasm
} // end namespace objcopy
} // end namespace llvm
