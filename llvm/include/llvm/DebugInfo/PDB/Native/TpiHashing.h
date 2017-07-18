//===- TpiHashing.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_TPIHASHING_H
#define LLVM_DEBUGINFO_PDB_TPIHASHING_H

#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/Support/Error.h"

namespace llvm {
namespace pdb {

Expected<uint32_t> hashTypeRecord(const llvm::codeview::CVType &Type);

} // end namespace pdb
} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_TPIHASHING_H
