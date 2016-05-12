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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Object/MachOUniversal.h"

using namespace llvm;

Error macho2yaml(raw_ostream &Out, const object::MachOObjectFile &Obj) {
  return make_error<Obj2YamlError>(obj2yaml_error::not_implemented);
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
