//===- MachOYAML.cpp - MachO YAMLIO implementation ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines classes for handling the YAML representation of MachO.
//
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/MachOYAML.h"
#include "llvm/Support/Casting.h"

namespace llvm {

namespace yaml {

void MappingTraits<MachOYAML::FileHeader>::mapping(
    IO &IO, MachOYAML::FileHeader &FileHdr) {
  IO.mapRequired("magic", FileHdr.magic);
  IO.mapRequired("cputype", FileHdr.cputype);
  IO.mapRequired("cpusubtype", FileHdr.cpusubtype);
  IO.mapRequired("filetype", FileHdr.filetype);
  IO.mapRequired("ncmds", FileHdr.ncmds);
  IO.mapRequired("sizeofcmds", FileHdr.sizeofcmds);
  IO.mapRequired("flags", FileHdr.flags);
  IO.mapOptional("reserved", FileHdr.reserved,
                 static_cast<llvm::yaml::Hex32>(0xDEADBEEFu));
}

void MappingTraits<MachOYAML::Object>::mapping(IO &IO,
                                               MachOYAML::Object &Object) {
  // If the context isn't already set, tag the document as !mach-o.
  // For Fat files there will be a different tag so they can be differentiated.
  if (!IO.getContext()) {
    IO.setContext(&Object);
    IO.mapTag("!mach-o", true);
  }
  IO.mapRequired("FileHeader", Object.Header);
  IO.setContext(nullptr);
}

} // namespace llvm::yaml

} // namespace llvm
