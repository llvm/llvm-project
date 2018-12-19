//===- COFFObjcopy.cpp ----------------------------------------------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "COFFObjcopy.h"
#include "Buffer.h"
#include "CopyConfig.h"
#include "Object.h"
#include "Reader.h"
#include "Writer.h"
#include "llvm-objcopy.h"

#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include <cassert>

namespace llvm {
namespace objcopy {
namespace coff {

using namespace object;
using namespace COFF;

void executeObjcopyOnBinary(const CopyConfig &Config,
                            object::COFFObjectFile &In, Buffer &Out) {
  COFFReader Reader(In);
  std::unique_ptr<Object> Obj = Reader.create();
  assert(Obj && "Unable to deserialize COFF object");
  COFFWriter Writer(*Obj, Out);
  Writer.write();
}

} // end namespace coff
} // end namespace objcopy
} // end namespace llvm
