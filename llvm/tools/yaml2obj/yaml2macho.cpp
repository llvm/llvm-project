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
#include "llvm/Support/MachO.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

class MachOWriter {
public:
  MachOWriter(MachOYAML::Object &Obj) : Obj(Obj) {
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

  MachOYAML::Object &Obj;
  bool is64Bit;

  union {
    MachO::mach_header_64 Header64;
    MachO::mach_header Header;
  };
};

Error MachOWriter::writeMachO(raw_ostream &OS) {
  if (auto Err = writeHeader(OS))
    return Err;
  if (auto Err = writeLoadCommands(OS))
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

Error MachOWriter::writeLoadCommands(raw_ostream &OS) {
  for (auto &LC : Obj.LoadCommands) {
    size_t BytesWritten = 0;
#define HANDLE_LOAD_COMMAND(LCName, LCValue, LCStruct)                         \
  case MachO::LCName:                                                          \
    OS.write(reinterpret_cast<const char *>(&(LC.Data.LCStruct##_data)),       \
             sizeof(MachO::LCStruct));                                         \
    BytesWritten = sizeof(MachO::LCStruct);                                    \
    break;

    switch (LC.Data.load_command_data.cmd) {
    default:
      OS.write(reinterpret_cast<const char *>(&(LC.Data.load_command_data)),
               sizeof(MachO::load_command));
      BytesWritten = sizeof(MachO::load_command);
      break;
#include "llvm/Support/MachO.def"
    }

    auto BytesRemaining =
        LC.Data.load_command_data.cmdsize - BytesWritten;
    if (BytesRemaining > 0) {
      // TODO: Replace all this once the load command data is present in yaml.
      // For now I fill with 0xDEADBEEF because it is easy to spot on a hex
      // viewer.
      std::vector<uint32_t> FillData;
      FillData.insert(FillData.begin(), BytesRemaining / 4 + 1, 0xDEADBEEFu);
      OS.write(reinterpret_cast<char *>(FillData.data()), BytesRemaining);
    }
  }
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
