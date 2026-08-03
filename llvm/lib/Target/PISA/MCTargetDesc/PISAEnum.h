//===-- PISAEnum.h --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAENUM_H
#define LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAENUM_H

#include "MCTargetDesc/PISAMCExpr.h"
#include "MCTargetDesc/PISAMCTargetDesc.h"
#include "MCTargetDesc/PISARegEncoder.h"
#include "MCTargetDesc/PISATargetStreamer.h"
#include "PISADefines.h"
#include "PISAMCInstLower.h"
#include "TargetInfo/PISATargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringTable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/AsmLexer.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm::PISA;

namespace llvm {
namespace PISA {

#define GET_BoolOptionID_DECL
#define GET_EnumOptionClass_DECL
#define GET_LoadCacheControl_DECL
#define GET_StoreCacheControl_DECL
#define GET_AtomicCacheControl_DECL
#include "PISAGenSearchableTables.inc"

struct BoolOptionTableEntry {
  BoolOptionID OptID;
  StringTable::Offset OptStr;
};

struct EnumOptionEntry {
  EnumOptionClass OptClass;
  unsigned Value;
  StringTable::Offset OptStr;
  unsigned CFlags;
};

#define GET_BoolOptionTable_DECL
#define GET_EnumOptionTable_DECL
#include "PISAGenSearchableTables.inc"

} // namespace PISA
} // namespace llvm
#endif // LLVM_LIB_TARGET_PISA_MCTARGETDESC_PISAENUM_H
