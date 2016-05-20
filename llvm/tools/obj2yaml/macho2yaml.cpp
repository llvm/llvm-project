//===------ macho2yaml.cpp - obj2yaml conversion tool -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "obj2yaml.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/ObjectYAML/MachOYAML.h"
#include "llvm/Support/ErrorHandling.h"

#include <string.h> // for memcpy

using namespace llvm;

class MachODumper {

  template <typename StructType>
  const char *processLoadCommandData(
      MachOYAML::LoadCommand &LC,
      const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd);

  const object::MachOObjectFile &Obj;

public:
  MachODumper(const object::MachOObjectFile &O) : Obj(O) {}
  Expected<std::unique_ptr<MachOYAML::Object>> dump();
};

#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  case MachO::LCName:                                                          \
    memcpy((void *) & (LC.Data.LCStruct##_data), LoadCmd.Ptr,                  \
           sizeof(MachO::LCStruct));                                           \
    if (Obj.isLittleEndian() != sys::IsLittleEndianHost)                       \
      MachO::swapStruct(LC.Data.LCStruct##_data);                              \
    EndPtr = processLoadCommandData<MachO::LCStruct>(LC, LoadCmd);             \
    break;

template <typename SectionType>
MachOYAML::Section constructSectionCommon(SectionType Sec) {
  MachOYAML::Section TempSec;
  memcpy(reinterpret_cast<void *>(&TempSec.sectname[0]), &Sec.sectname[0], 16);
  memcpy(reinterpret_cast<void *>(&TempSec.segname[0]), &Sec.segname[0], 16);
  TempSec.addr = Sec.addr;
  TempSec.size = Sec.size;
  TempSec.offset = Sec.offset;
  TempSec.align = Sec.align;
  TempSec.reloff = Sec.reloff;
  TempSec.nreloc = Sec.nreloc;
  TempSec.flags = Sec.flags;
  TempSec.reserved1 = Sec.reserved1;
  TempSec.reserved2 = Sec.reserved2;
  TempSec.reserved3 = 0;
  return TempSec;
}

template <typename SectionType>
MachOYAML::Section constructSection(SectionType Sec);

template <> MachOYAML::Section constructSection(MachO::section Sec) {
  MachOYAML::Section TempSec = constructSectionCommon(Sec);
  TempSec.reserved3 = 0;
  return TempSec;
}

template <> MachOYAML::Section constructSection(MachO::section_64 Sec) {
  MachOYAML::Section TempSec = constructSectionCommon(Sec);
  TempSec.reserved3 = Sec.reserved3;
  return TempSec;
}

template <typename SectionType, typename SegmentType>
const char *
extractSections(const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd,
                std::vector<MachOYAML::Section> &Sections,
                bool IsLittleEndian) {
  auto End = LoadCmd.Ptr + LoadCmd.C.cmdsize;
  const SectionType *Curr =
      reinterpret_cast<const SectionType *>(LoadCmd.Ptr + sizeof(SegmentType));
  for (; reinterpret_cast<const void *>(Curr) < End; Curr++) {
    if (IsLittleEndian != sys::IsLittleEndianHost) {
      SectionType Sec;
      memcpy((void *)&Sec, Curr, sizeof(SectionType));
      MachO::swapStruct(Sec);
      Sections.push_back(constructSection(Sec));
    } else {
      Sections.push_back(constructSection(*Curr));
    }
  }
  return reinterpret_cast<const char *>(Curr);
}

template <typename StructType>
const char *MachODumper::processLoadCommandData(
    MachOYAML::LoadCommand &LC,
    const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd) {
  return LoadCmd.Ptr + sizeof(StructType);
}

template <>
const char *MachODumper::processLoadCommandData<MachO::segment_command>(
    MachOYAML::LoadCommand &LC,
    const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd) {
  return extractSections<MachO::section, MachO::segment_command>(
      LoadCmd, LC.Sections, Obj.isLittleEndian());
}

template <>
const char *MachODumper::processLoadCommandData<MachO::segment_command_64>(
    MachOYAML::LoadCommand &LC,
    const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd) {
  return extractSections<MachO::section_64, MachO::segment_command_64>(
      LoadCmd, LC.Sections, Obj.isLittleEndian());
}

template <typename StructType>
const char *
readString(MachOYAML::LoadCommand &LC,
           const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd) {
  auto Start = LoadCmd.Ptr + sizeof(StructType);
  auto MaxSize = LoadCmd.C.cmdsize - sizeof(StructType);
  auto Size = strnlen(Start, MaxSize);
  LC.PayloadString = StringRef(Start, Size).str();
  return Start + Size;
}

template <>
const char *MachODumper::processLoadCommandData<MachO::dylib_command>(
    MachOYAML::LoadCommand &LC,
    const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd) {
  return readString<MachO::dylib_command>(LC, LoadCmd);
}

template <>
const char *MachODumper::processLoadCommandData<MachO::dylinker_command>(
    MachOYAML::LoadCommand &LC,
    const llvm::object::MachOObjectFile::LoadCommandInfo &LoadCmd) {
  return readString<MachO::dylinker_command>(LC, LoadCmd);
}

Expected<std::unique_ptr<MachOYAML::Object>> MachODumper::dump() {
  auto Y = make_unique<MachOYAML::Object>();
  Y->Header.magic = Obj.getHeader().magic;
  Y->Header.cputype = Obj.getHeader().cputype;
  Y->Header.cpusubtype = Obj.getHeader().cpusubtype;
  Y->Header.filetype = Obj.getHeader().filetype;
  Y->Header.ncmds = Obj.getHeader().ncmds;
  Y->Header.sizeofcmds = Obj.getHeader().sizeofcmds;
  Y->Header.flags = Obj.getHeader().flags;
  Y->Header.reserved = 0;

  for (auto LoadCmd : Obj.load_commands()) {
    MachOYAML::LoadCommand LC;
    const char *EndPtr = LoadCmd.Ptr;
    switch (LoadCmd.C.cmd) {
    default:
      memcpy((void *)&(LC.Data.load_command_data), LoadCmd.Ptr,
             sizeof(MachO::load_command));
      if (Obj.isLittleEndian() != sys::IsLittleEndianHost)
        MachO::swapStruct(LC.Data.load_command_data);
      EndPtr = processLoadCommandData<MachO::load_command>(LC, LoadCmd);
      break;
#include "llvm/Support/MachO.def"
    }
    auto RemainingBytes = LoadCmd.C.cmdsize - (EndPtr - LoadCmd.Ptr);
    if (!std::all_of(EndPtr, &EndPtr[RemainingBytes],
                     [](const char C) { return C == 0; })) {
      LC.PayloadBytes.insert(LC.PayloadBytes.end(), EndPtr,
                             &EndPtr[RemainingBytes]);
      RemainingBytes = 0;
    }
    LC.ZeroPadBytes = RemainingBytes;
    Y->LoadCommands.push_back(std::move(LC));
  }

  return std::move(Y);
}

Error macho2yaml(raw_ostream &Out, const object::MachOObjectFile &Obj) {
  MachODumper Dumper(Obj);
  Expected<std::unique_ptr<MachOYAML::Object>> YAML = Dumper.dump();
  if (!YAML)
    return YAML.takeError();

  yaml::Output Yout(Out);
  Yout << *(YAML.get());
  return Error::success();
}

Error macho2yaml(raw_ostream &Out, const object::MachOUniversalBinary &Obj) {
  return make_error<Obj2YamlError>(obj2yaml_error::not_implemented);
}

std::error_code macho2yaml(raw_ostream &Out, const object::ObjectFile &Obj) {
  if (const auto *MachOObj = dyn_cast<object::MachOUniversalBinary>(&Obj)) {
    if (auto Err = macho2yaml(Out, *MachOObj)) {
      return errorToErrorCode(std::move(Err));
    }
    return obj2yaml_error::success;
  }

  if (const auto *MachOObj = dyn_cast<object::MachOObjectFile>(&Obj)) {
    if (auto Err = macho2yaml(Out, *MachOObj)) {
      return errorToErrorCode(std::move(Err));
    }
    return obj2yaml_error::success;
  }

  return obj2yaml_error::unsupported_obj_file_format;
}
