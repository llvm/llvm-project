//===- yaml2macho - Convert YAML to a Mach object file --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The Mach component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "yaml2obj.h"
#include "llvm/ObjectYAML/MachOYAML.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Format.h"

using namespace llvm;

namespace {

class MachOWriter {
public:
  MachOWriter(MachOYAML::Object &Obj) : Obj(Obj), is64Bit(true), fileStart(0) {
    is64Bit = Obj.Header.magic == MachO::MH_MAGIC_64 ||
              Obj.Header.magic == MachO::MH_CIGAM_64;
    memset(reinterpret_cast<void *>(&Header64), 0,
           sizeof(MachO::mach_header_64));
    assert((is64Bit || Obj.Header.reserved == 0xDEADBEEFu) &&
           "32-bit MachO has reserved in header");
    assert((!is64Bit || Obj.Header.reserved != 0xDEADBEEFu) &&
           "64-bit MachO has missing reserved in header");
  }

  Error writeMachO(raw_ostream &OS);

private:
  Error writeHeader(raw_ostream &OS);
  Error writeLoadCommands(raw_ostream &OS);
  Error writeSectionData(raw_ostream &OS);
  Error writeLinkEditData(raw_ostream &OS);

  void ZeroToOffset(raw_ostream &OS, size_t offset);

  MachOYAML::Object &Obj;
  bool is64Bit;
  uint64_t fileStart;

  union {
    MachO::mach_header_64 Header64;
    MachO::mach_header Header;
  };
};

Error MachOWriter::writeMachO(raw_ostream &OS) {
  fileStart = OS.tell();
  if (auto Err = writeHeader(OS))
    return Err;
  if (auto Err = writeLoadCommands(OS))
    return Err;
  if (auto Err = writeSectionData(OS))
    return Err;
  return Error::success();
}

Error MachOWriter::writeHeader(raw_ostream &OS) {
  Header.magic = Obj.Header.magic;
  Header.cputype = Obj.Header.cputype;
  Header.cpusubtype = Obj.Header.cpusubtype;
  Header.filetype = Obj.Header.filetype;
  Header.ncmds = Obj.Header.ncmds;
  Header.sizeofcmds = Obj.Header.sizeofcmds;
  Header.flags = Obj.Header.flags;
  Header64.reserved = Obj.Header.reserved;

  if (is64Bit)
    OS.write((const char *)&Header64, sizeof(MachO::mach_header_64));
  else
    OS.write((const char *)&Header, sizeof(MachO::mach_header));

  return Error::success();
}

template <typename SectionType>
SectionType constructSection(MachOYAML::Section Sec) {
  SectionType TempSec;
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
  return TempSec;
}

template <typename StructType>
size_t writeLoadCommandData(MachOYAML::LoadCommand &LC, raw_ostream &OS) {
  return 0;
}

template <>
size_t writeLoadCommandData<MachO::segment_command>(MachOYAML::LoadCommand &LC,
                                                    raw_ostream &OS) {
  size_t BytesWritten = 0;
  for (auto Sec : LC.Sections) {
    auto TempSec = constructSection<MachO::section>(Sec);
    OS.write(reinterpret_cast<const char *>(&(TempSec)),
             sizeof(MachO::section));
    BytesWritten += sizeof(MachO::section);
  }
  return BytesWritten;
}

template <>
size_t
writeLoadCommandData<MachO::segment_command_64>(MachOYAML::LoadCommand &LC,
                                                raw_ostream &OS) {
  size_t BytesWritten = 0;
  for (auto Sec : LC.Sections) {
    auto TempSec = constructSection<MachO::section_64>(Sec);
    TempSec.reserved3 = Sec.reserved3;
    OS.write(reinterpret_cast<const char *>(&(TempSec)),
             sizeof(MachO::section_64));
    BytesWritten += sizeof(MachO::section_64);
  }
  return BytesWritten;
}

size_t writePayloadString(MachOYAML::LoadCommand &LC, raw_ostream &OS) {
  size_t BytesWritten = 0;
  if (!LC.PayloadString.empty()) {
    OS.write(LC.PayloadString.c_str(), LC.PayloadString.length());
    BytesWritten = LC.PayloadString.length();
  }
  return BytesWritten;
}

template <>
size_t writeLoadCommandData<MachO::dylib_command>(MachOYAML::LoadCommand &LC,
                                                  raw_ostream &OS) {
  return writePayloadString(LC, OS);
}

template <>
size_t writeLoadCommandData<MachO::dylinker_command>(MachOYAML::LoadCommand &LC,
                                                     raw_ostream &OS) {
  return writePayloadString(LC, OS);
}

void ZeroFillBytes(raw_ostream &OS, size_t Size) {
  std::vector<uint8_t> FillData;
  FillData.insert(FillData.begin(), Size, 0);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

void Fill(raw_ostream &OS, size_t Size, uint32_t Data) {
  std::vector<uint32_t> FillData;
  FillData.insert(FillData.begin(), (Size / 4) + 1, Data);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

void MachOWriter::ZeroToOffset(raw_ostream &OS, size_t Offset) {
  auto currOffset = OS.tell() - fileStart;
  if (currOffset < Offset)
    ZeroFillBytes(OS, Offset - currOffset);
}

Error MachOWriter::writeLoadCommands(raw_ostream &OS) {
  for (auto &LC : Obj.LoadCommands) {
    size_t BytesWritten = 0;
#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  case MachO::LCName:                                                          \
    OS.write(reinterpret_cast<const char *>(&(LC.Data.LCStruct##_data)),       \
             sizeof(MachO::LCStruct));                                         \
    BytesWritten = sizeof(MachO::LCStruct);                                    \
    BytesWritten += writeLoadCommandData<MachO::LCStruct>(LC, OS);             \
    break;

    switch (LC.Data.load_command_data.cmd) {
    default:
      OS.write(reinterpret_cast<const char *>(&(LC.Data.load_command_data)),
               sizeof(MachO::load_command));
      BytesWritten = sizeof(MachO::load_command);
      BytesWritten += writeLoadCommandData<MachO::load_command>(LC, OS);
      break;
#include "llvm/Support/MachO.def"
    }

    if (LC.PayloadBytes.size() > 0) {
      OS.write(reinterpret_cast<const char *>(LC.PayloadBytes.data()),
               LC.PayloadBytes.size());
      BytesWritten += LC.PayloadBytes.size();
    }

    if (LC.ZeroPadBytes > 0) {
      ZeroFillBytes(OS, LC.ZeroPadBytes);
      BytesWritten += LC.ZeroPadBytes;
    }

    // Fill remaining bytes with 0. This will only get hit in partially
    // specified test cases.
    auto BytesRemaining = LC.Data.load_command_data.cmdsize - BytesWritten;
    if (BytesRemaining > 0) {
      ZeroFillBytes(OS, BytesRemaining);
    }
  }
  return Error::success();
}

Error MachOWriter::writeSectionData(raw_ostream &OS) {
  for (auto &LC : Obj.LoadCommands) {
    switch (LC.Data.load_command_data.cmd) {
    case MachO::LC_SEGMENT:
    case MachO::LC_SEGMENT_64:
      auto currOffset = OS.tell() - fileStart;
      auto segname = LC.Data.segment_command_data.segname;
      uint64_t segOff = is64Bit ? LC.Data.segment_command_64_data.fileoff
                                : LC.Data.segment_command_data.fileoff;

      if (0 == strncmp(&segname[0], "__LINKEDIT", 16)) {
        if (auto Err = writeLinkEditData(OS))
          return Err;
      } else {
        // Zero Fill any data between the end of the last thing we wrote and the
        // start of this section.
        if (currOffset < segOff) {
          ZeroFillBytes(OS, segOff - currOffset);
        }

        for (auto &Sec : LC.Sections) {
          // Zero Fill any data between the end of the last thing we wrote and
          // the
          // start of this section.
          assert(
              OS.tell() - fileStart <= Sec.offset &&
              "Wrote too much data somewhere, section offsets don't line up.");
          currOffset = OS.tell() - fileStart;
          if (currOffset < Sec.offset) {
            ZeroFillBytes(OS, Sec.offset - currOffset);
          }

          // Fills section data with 0xDEADBEEF
          Fill(OS, Sec.size, 0xDEADBEEFu);
        }
      }
      uint64_t segSize = is64Bit ? LC.Data.segment_command_64_data.filesize
                                 : LC.Data.segment_command_data.filesize;
      ZeroToOffset(OS, segOff + segSize);
      break;
    }
  }
  return Error::success();
}

Error MachOWriter::writeLinkEditData(raw_ostream &OS) {
  MachOYAML::LinkEditData &LinkEdit = Obj.LinkEdit;
  MachO::dyld_info_command *DyldInfoOnlyCmd = 0;
  MachO::symtab_command *SymtabCmd = 0;
  for (auto &LC : Obj.LoadCommands) {
    switch (LC.Data.load_command_data.cmd) {
    case MachO::LC_SYMTAB:
      SymtabCmd = &LC.Data.symtab_command_data;
      break;
    case MachO::LC_DYLD_INFO_ONLY:
      DyldInfoOnlyCmd = &LC.Data.dyld_info_command_data;
      break;
    }
  }

  ZeroToOffset(OS, DyldInfoOnlyCmd->rebase_off);

  for (auto Opcode : LinkEdit.RebaseOpcodes) {
    uint8_t OpByte = Opcode.Opcode | Opcode.Imm;
    OS.write(reinterpret_cast<char *>(&OpByte), 1);
    for (auto Data : Opcode.ExtraData) {
      encodeULEB128(Data, OS);
    }
  }

  // Fill to the end of the string table
  ZeroToOffset(OS, SymtabCmd->stroff + SymtabCmd->strsize);

  return Error::success();
}

} // end anonymous namespace

int yaml2macho(yaml::Input &YIn, raw_ostream &Out) {
  MachOYAML::Object Doc;
  YIn >> Doc;
  if (YIn.error()) {
    errs() << "yaml2obj: Failed to parse YAML file!\n";
    return 1;
  }

  MachOWriter Writer(Doc);
  if (auto Err = Writer.writeMachO(Out)) {
    errs() << toString(std::move(Err));
    return 1;
  }
  return 0;
}
