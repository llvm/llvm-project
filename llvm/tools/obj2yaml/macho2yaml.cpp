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
    break;

template <typename SectionType>
MachOYAML::Section constructSection(SectionType Sec) {
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
    switch (LoadCmd.C.cmd) {
    default:
      memcpy((void *)&(LC.Data.load_command_data), LoadCmd.Ptr,
             sizeof(MachO::load_command));
      if (Obj.isLittleEndian() != sys::IsLittleEndianHost)
        MachO::swapStruct(LC.Data.load_command_data);
      break;
#include "llvm/Support/MachO.def"
    }
    if (LoadCmd.C.cmd == MachO::LC_SEGMENT) {
      auto End = LoadCmd.Ptr + LoadCmd.C.cmdsize;
      const MachO::section *Curr = reinterpret_cast<const MachO::section *>(
          LoadCmd.Ptr + sizeof(MachO::segment_command));
      for (; reinterpret_cast<const void *>(Curr) < End; Curr++) {
        if (Obj.isLittleEndian() != sys::IsLittleEndianHost) {
          MachO::section Sec;
          memcpy((void *)&Sec, Curr, sizeof(MachO::section));
          MachO::swapStruct(Sec);
          LC.Sections.push_back(constructSection(Sec));
        } else {
          LC.Sections.push_back(constructSection(*Curr));
        }
      }
    } else if (LoadCmd.C.cmd == MachO::LC_SEGMENT_64) {
      auto End = LoadCmd.Ptr + LoadCmd.C.cmdsize;
      const MachO::section_64 *Curr =
          reinterpret_cast<const MachO::section_64 *>(
              LoadCmd.Ptr + sizeof(MachO::segment_command_64));
      for (; reinterpret_cast<const void *>(Curr) < End; Curr++) {
        MachOYAML::Section TempSec;
        if (Obj.isLittleEndian() != sys::IsLittleEndianHost) {
          MachO::section_64 Sec;
          memcpy((void *)&Sec, Curr, sizeof(MachO::section_64));
          MachO::swapStruct(Sec);
          LC.Sections.push_back(constructSection(Sec));
          TempSec = constructSection(Sec);
        } else {
          TempSec = constructSection(*Curr);
        }
        TempSec.reserved3 = Curr->reserved3;
        LC.Sections.push_back(TempSec);
      }
    }
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
