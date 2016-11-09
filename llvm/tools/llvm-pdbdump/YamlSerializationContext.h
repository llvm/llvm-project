//===- YamlSerializationContext.h ----------------------------- *- C++ --*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVMPDBDUMP_YAMLSERIALIZATIONCONTEXT_H
#define LLVM_TOOLS_LLVMPDBDUMP_YAMLSERIALIZATIONCONTEXT_H

#include "PdbYaml.h"
#include "YamlTypeDumper.h"
#include "llvm/Support/Allocator.h"

namespace llvm {
namespace codeview {
class TypeSerializer;
}
namespace yaml {
class IO;
}

namespace pdb {
namespace yaml {
struct SerializationContext {
  explicit SerializationContext(llvm::yaml::IO &IO, BumpPtrAllocator &Allocator)
      : Dumper(IO, *this), Allocator(Allocator) {}

  codeview::yaml::YamlTypeDumperCallbacks Dumper;
  BumpPtrAllocator &Allocator;
  codeview::TypeSerializer *ActiveSerializer = nullptr;
};
}
}
}

#endif