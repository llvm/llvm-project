//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "Error.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/Minidump.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace llvm;
using namespace llvm::object;

static std::error_code dumpObject(const ObjectFile &Obj) {
  if (Obj.isCOFF())
    return coff2yaml(outs(), cast<COFFObjectFile>(Obj));

  if (Obj.isXCOFF())
    return xcoff2yaml(outs(), cast<XCOFFObjectFile>(Obj));

  if (Obj.isELF())
    return elf2yaml(outs(), Obj);
  if (Obj.isWasm())
    return wasm2yaml(outs(), cast<WasmObjectFile>(Obj));

  return obj2yaml_error::unsupported_obj_file_format;
}

static Error dumpInput(StringRef File) {
  Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(File);
  if (!BinaryOrErr)
    return BinaryOrErr.takeError();

  Binary &Binary = *BinaryOrErr.get().getBinary();
  // Universal MachO is not a subclass of ObjectFile, so it needs to be handled
  // here with the other binary types.
  if (Binary.isMachO() || Binary.isMachOUniversalBinary())
    return errorCodeToError(macho2yaml(outs(), Binary));
  // TODO: If this is an archive, then burst it and dump each entry
  if (ObjectFile *Obj = dyn_cast<ObjectFile>(&Binary))
    return errorCodeToError(dumpObject(*Obj));
  if (MinidumpFile *Minidump = dyn_cast<MinidumpFile>(&Binary))
    return minidump2yaml(outs(), *Minidump);

  return Error::success();
}

static void reportError(StringRef Input, Error Err) {
  if (Input == "-")
    Input = "<stdin>";
  std::string ErrMsg;
  raw_string_ostream OS(ErrMsg);
  logAllUnhandledErrors(std::move(Err), OS);
  OS.flush();
  errs() << "Error reading file: " << Input << ": " << ErrMsg;
  errs().flush();
}

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"),
                                   cl::init("-"));

int main(int argc, char *argv[]) {
  InitLLVM X(argc, argv);
  cl::ParseCommandLineOptions(argc, argv);

  if (Error Err = dumpInput(InputFilename)) {
    reportError(InputFilename, std::move(Err));
    return 1;
  }

  return 0;
}
