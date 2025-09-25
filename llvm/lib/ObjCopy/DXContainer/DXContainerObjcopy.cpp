//===- DXContainerObjcopy.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjCopy/DXContainer/DXContainerObjcopy.h"
#include "DXContainerReader.h"
#include "DXContainerWriter.h"
#include "llvm/ObjCopy/CommonConfig.h"
#include "llvm/ObjCopy/DXContainer/DXContainerConfig.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace objcopy {
namespace dxbc {

using namespace object;

static Error extractPartAsObject(StringRef PartName, StringRef OutFilename,
                                 StringRef InputFilename, const Object &Obj) {
  for (const Part &P : Obj.Parts)
    if (P.Name == PartName) {
      Object PartObj;
      PartObj.Header = Obj.Header;
      PartObj.Parts.push_back({P.Name, P.Data});
      PartObj.recomputeHeader();

      auto Write = [&OutFilename, &PartObj](raw_ostream &Out) -> Error {
        DXContainerWriter Writer(PartObj, Out);
        if (Error E = Writer.write())
          return createFileError(OutFilename, std::move(E));
        return Error::success();
      };

      return writeToOutput(OutFilename, Write);
    }

  return createFileError(InputFilename, object_error::parse_failed,
                         "part '%s' not found", PartName.str().c_str());
}

static Error handleArgs(const CommonConfig &Config, Object &Obj) {
  // Extract all sections before any modifications.
  for (StringRef Flag : Config.ExtractSection) {
    StringRef SectionName;
    StringRef FileName;
    std::tie(SectionName, FileName) = Flag.split('=');
    if (Error E = extractPartAsObject(SectionName, FileName,
                                      Config.InputFilename, Obj))
      return E;
  }

  std::function<bool(const Part &)> RemovePred = [](const Part &) {
    return false;
  };

  if (!Config.ToRemove.empty())
    RemovePred = [&Config](const Part &P) {
      return Config.ToRemove.matches(P.Name);
    };

  if (!Config.OnlySection.empty())
    RemovePred = [&Config](const Part &P) {
      // Explicitly keep these sections regardless of previous removes and
      // remove everything else.
      return !Config.OnlySection.matches(P.Name);
    };

  if (auto E = Obj.removeParts(RemovePred))
    return E;

  Obj.recomputeHeader();
  return Error::success();
}

Error executeObjcopyOnBinary(const CommonConfig &Config,
                             const DXContainerConfig &,
                             DXContainerObjectFile &In, raw_ostream &Out) {
  DXContainerReader Reader(In);
  Expected<std::unique_ptr<Object>> ObjOrErr = Reader.create();
  if (!ObjOrErr)
    return createFileError(Config.InputFilename, ObjOrErr.takeError());
  Object *Obj = ObjOrErr->get();
  assert(Obj && "Unable to deserialize DXContainer object");

  if (Error E = handleArgs(Config, *Obj))
    return E;

  DXContainerWriter Writer(*Obj, Out);
  if (Error E = Writer.write())
    return createFileError(Config.OutputFilename, std::move(E));
  return Error::success();
}

} // end namespace dxbc
} // end namespace objcopy
} // end namespace llvm
