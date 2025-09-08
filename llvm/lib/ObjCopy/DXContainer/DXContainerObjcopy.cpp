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

namespace llvm {
namespace objcopy {
namespace dxbc {

using namespace object;

static Error handleArgs(const CommonConfig &Config, Object &Obj) {
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
