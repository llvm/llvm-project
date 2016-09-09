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

#include "CodeViewYaml.h"
#include "PdbYaml.h"
#include "llvm/DebugInfo/CodeView/FieldListRecordBuilder.h"
#include "llvm/DebugInfo/CodeView/MemoryTypeTableBuilder.h"

namespace llvm {
namespace yaml {
class IO;
}

namespace pdb {
namespace yaml {
struct SerializationContext {
  explicit SerializationContext(llvm::yaml::IO &IO, BumpPtrAllocator &Allocator)
      : Dumper(IO, *this), TypeTableBuilder(Allocator) {}
  codeview::yaml::YamlTypeDumperCallbacks Dumper;
  codeview::MemoryTypeTableBuilder TypeTableBuilder;
  codeview::FieldListRecordBuilder FieldListBuilder;
};
}
}
}

#endif