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

static Error splitPartAsObject(StringRef PartName, StringRef OutFilename,
                               StringRef InputFilename, const Object &Obj) {
  for (const Part &P : Obj.Parts)
    if (P.Name == PartName) {
      auto PartObj = std::make_unique<Object>();
      PartObj->Header = Obj.Header;
      PartObj->Parts.push_back({P.Name, P.Data});
      PartObj->recomputeHeader();

      auto Write = [&OutFilename, &PartObj](raw_ostream &Out) -> Error {
        DXContainerWriter Writer(*PartObj, Out);
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
  std::function<bool(const Part &)> RemovePred = [](const Part &) {
    return false;
  };

  if (!Config.ToRemove.empty())
    RemovePred = [&Config](const Part &P) {
      return Config.ToRemove.matches(P.Name);
    };

  if (!Config.SplitSection.empty()) {
    for (StringRef Flag : Config.SplitSection) {
      StringRef SectionName;
      StringRef FileName;
      std::tie(SectionName, FileName) = Flag.split('=');

      if (Error E = splitPartAsObject(SectionName, FileName,
                                      Config.InputFilename, Obj))
        return E;
    }

    RemovePred = [&Config, RemovePred](const Part &P) {
      if (RemovePred(P))
        return true;

      for (StringRef Flag : Config.SplitSection) {
        bool CanContain = Flag.size() > P.Name.size();
        if (CanContain && Flag.starts_with(P.Name) &&
            Flag[P.Name.size()] == '=')
          return true;
      }

      return false;
    };
  }

  if (auto E = Obj.removeParts(RemovePred))
    return E;

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
