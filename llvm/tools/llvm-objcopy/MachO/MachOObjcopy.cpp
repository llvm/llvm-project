//===- MachOObjcopy.cpp -----------------------------------------*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
