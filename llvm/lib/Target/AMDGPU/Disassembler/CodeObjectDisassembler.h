//=-- CodeObjectDisassembler.hpp - Disassembler for HSA Code Object--- C++ --=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// This file contains declaration for HSA Code Object Dissassembler
//
//===----------------------------------------------------------------------===//

#ifndef AMDGPU_CODE_OBJECT_DISASSEMBLER_HPP
#define AMDGPU_CODE_OBJECT_DISASSEMBLER_HPP

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Object/ELFObjectFile.h"
#include <vector>

namespace llvm {

class AMDGPUTargetStreamer;
class MCContext;
class MCDisassembler;
class MCInstPrinter;
class MCTargetStreamer;
class raw_ostream;
class HSACodeObject;

class CodeObjectDisassembler {
private:
  MCContext *Ctx;
  StringRef TripleName;
  MCInstPrinter *InstPrinter;
  AMDGPUTargetStreamer *AsmStreamer;

  CodeObjectDisassembler(const CodeObjectDisassembler&) = delete;
  CodeObjectDisassembler& operator=(const CodeObjectDisassembler&) = delete;
  
  typedef std::vector<std::tuple<uint64_t, StringRef, uint8_t>> SymbolsTy;

  Expected<SymbolsTy> CollectSymbols(const HSACodeObject *CodeObject);

  std::error_code printNotes(const HSACodeObject *CodeObject);
  std::error_code printFunctions(const HSACodeObject *CodeObject,
                                 raw_ostream &ES);
  void printFunctionCode(const MCDisassembler &InstDisasm,
                         ArrayRef<uint8_t> Bytes, uint64_t Address,
                         const SymbolsTy &Symbols, raw_ostream &ES);

public:
  CodeObjectDisassembler(MCContext *C, StringRef TripleName, MCInstPrinter *IP,
                         MCTargetStreamer *TS);

  /// @brief Disassemble and print HSA Code Object
  std::error_code Disassemble(MemoryBufferRef Buffer, raw_ostream &ES);
};
} // namespace llvm

#endif
