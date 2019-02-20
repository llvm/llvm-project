//===- MachOObjcopy.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MachOObjcopy.h"
#include "../CopyConfig.h"
#include "../llvm-objcopy.h"
#include "MachOReader.h"
#include "MachOWriter.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace objcopy {
namespace macho {

Error executeObjcopyOnBinary(const CopyConfig &Config,
                             object::MachOObjectFile &In, Buffer &Out) {
  MachOReader Reader(In);
  std::unique_ptr<Object> O = Reader.create();
  assert(O && "Unable to deserialize MachO object");
  MachOWriter Writer(*O, In.is64Bit(), In.isLittleEndian(), Out);
  return Writer.write();
}

} // end namespace macho
} // end namespace objcopy
} // end namespace llvm
