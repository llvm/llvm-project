//===- WasmObjcopy.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WasmObjcopy.h"
#include "CommonConfig.h"
#include "Object.h"
#include "Reader.h"
#include "Writer.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileOutputBuffer.h"

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
static Error handleArgs(const CommonConfig &Config, Object &Obj) {
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

  return Error::success();
}

Error executeObjcopyOnBinary(const CommonConfig &Config, const WasmConfig &,
                             object::WasmObjectFile &In, raw_ostream &Out) {
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
