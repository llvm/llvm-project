//=- WebAssemblyInstPrinter.cpp - WebAssembly assembly instruction printing -=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Print MCInst instructions to wasm format.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "MCTargetDesc/WebAssemblyMCTypeUtilities.h"
#include "WebAssembly.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

#include "WebAssemblyGenAsmWriter.inc"

WebAssemblyInstPrinter::WebAssemblyInstPrinter(const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI)
    : MCInstPrinter(MAI, MII, MRI) {}

void WebAssemblyInstPrinter::printRegName(raw_ostream &OS,
                                          MCRegister Reg) const {
  assert(Reg.id() != WebAssembly::UnusedReg);
  // Note that there's an implicit local.get/local.set here!
  OS << "$" << Reg.id();
}

void WebAssemblyInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                       StringRef Annot,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &OS) {
  switch (MI->getOpcode()) {
  case WebAssembly::CALL_INDIRECT_S:
  case WebAssembly::RET_CALL_INDIRECT_S: {
    // A special case for call_indirect (and ret_call_indirect), if the table
    // operand is a symbol: the order of the type and table operands is inverted
    // in the text format relative to the binary format.  Otherwise if table the
    // operand isn't a symbol, then we have an MVP compilation unit, and the
    // table shouldn't appear in the output.
    OS << "\t";
    OS << getMnemonic(MI).first;
    OS << " ";

    assert(MI->getNumOperands() == 2);
    const unsigned TypeOperand = 0;
    const unsigned TableOperand = 1;
    if (MI->getOperand(TableOperand).isExpr()) {
      printOperand(MI, TableOperand, OS);
      OS << ", ";
    } else {
      assert(MI->getOperand(TableOperand).getImm() == 0);
    }
    printOperand(MI, TypeOperand, OS);
    break;
  }
  default:
    // Print the instruction (this uses the AsmStrings from the .td files).
    printInstruction(MI, Address, OS);
    break;
  }

  // Print any additional variadic operands.
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (Desc.isVariadic()) {
    if ((Desc.getNumOperands() == 0 && MI->getNumOperands() > 0) ||
        Desc.variadicOpsAreDefs())
      OS << "\t";
    unsigned Start = Desc.getNumOperands();
    unsigned NumVariadicDefs = 0;
    if (Desc.variadicOpsAreDefs()) {
      // The number of variadic defs is encoded in an immediate by MCInstLower
      NumVariadicDefs = MI->getOperand(0).getImm();
      Start = 1;
    }
    bool NeedsComma = Desc.getNumOperands() > 0 && !Desc.variadicOpsAreDefs();
    for (auto I = Start, E = MI->getNumOperands(); I < E; ++I) {
      if (MI->getOpcode() == WebAssembly::CALL_INDIRECT &&
          I - Start == NumVariadicDefs) {
        // Skip type and table arguments when printing for tests.
        ++I;
        continue;
      }
      if (NeedsComma)
        OS << ", ";
      printOperand(MI, I, OS, I - Start < NumVariadicDefs);
      NeedsComma = true;
    }
  }

  // Print any added annotation.
  printAnnotation(OS, Annot);

  auto PrintBranchAnnotation = [&](const MCOperand &Op,
                                   SmallSet<uint64_t, 8> &Printed) {
    uint64_t Depth = Op.getImm();
    if (!Printed.insert(Depth).second)
      return;
    if (Depth >= ControlFlowStack.size()) {
      printAnnotation(OS, "Invalid depth argument!");
    } else {
      const auto &Pair = ControlFlowStack.rbegin()[Depth];
      printAnnotation(OS, utostr(Depth) + ": " + (Pair.second ? "up" : "down") +
                              " to label" + utostr(Pair.first));
    }
  };

  if (CommentStream) {
    // Observe any effects on the control flow stack, for use in annotating
    // control flow label references.
    unsigned Opc = MI->getOpcode();
    switch (Opc) {
    default:
      break;

    case WebAssembly::LOOP:
    case WebAssembly::LOOP_S:
      printAnnotation(OS, "label" + utostr(ControlFlowCounter) + ':');
      ControlFlowStack.push_back(std::make_pair(ControlFlowCounter++, true));
      return;

    case WebAssembly::BLOCK:
    case WebAssembly::BLOCK_S:
      ControlFlowStack.push_back(std::make_pair(ControlFlowCounter++, false));
      return;

    case WebAssembly::TRY:
    case WebAssembly::TRY_S:
      ControlFlowStack.push_back(std::make_pair(ControlFlowCounter, false));
      TryStack.push_back(ControlFlowCounter++);
      EHInstStack.push_back(TRY);
      return;

    case WebAssembly::TRY_TABLE:
    case WebAssembly::TRY_TABLE_S: {
      SmallSet<uint64_t, 8> Printed;
      unsigned OpIdx = 1;
      const MCOperand &Op = MI->getOperand(OpIdx++);
      unsigned NumCatches = Op.getImm();
      for (unsigned I = 0; I < NumCatches; I++) {
        int64_t CatchOpcode = MI->getOperand(OpIdx++).getImm();
        if (CatchOpcode == wasm::WASM_OPCODE_CATCH ||
            CatchOpcode == wasm::WASM_OPCODE_CATCH_REF)
          OpIdx++; // Skip tag
        PrintBranchAnnotation(MI->getOperand(OpIdx++), Printed);
      }
      ControlFlowStack.push_back(std::make_pair(ControlFlowCounter++, false));
      return;
    }

    case WebAssembly::END_LOOP:
    case WebAssembly::END_LOOP_S:
      if (ControlFlowStack.empty()) {
        printAnnotation(OS, "End marker mismatch!");
      } else {
        ControlFlowStack.pop_back();
      }
      return;

    case WebAssembly::END_BLOCK:
    case WebAssembly::END_BLOCK_S:
    case WebAssembly::END_TRY_TABLE:
    case WebAssembly::END_TRY_TABLE_S:
      if (ControlFlowStack.empty()) {
        printAnnotation(OS, "End marker mismatch!");
      } else {
        printAnnotation(
            OS, "label" + utostr(ControlFlowStack.pop_back_val().first) + ':');
      }
      return;

    case WebAssembly::END_TRY:
    case WebAssembly::END_TRY_S:
      if (ControlFlowStack.empty() || EHInstStack.empty()) {
        printAnnotation(OS, "End marker mismatch!");
      } else {
        printAnnotation(
            OS, "label" + utostr(ControlFlowStack.pop_back_val().first) + ':');
        EHInstStack.pop_back();
      }
      return;

    case WebAssembly::CATCH_LEGACY:
    case WebAssembly::CATCH_LEGACY_S:
    case WebAssembly::CATCH_ALL_LEGACY:
    case WebAssembly::CATCH_ALL_LEGACY_S:
      // There can be multiple catch instructions for one try instruction, so
      // we print a label only for the first 'catch' label.
      if (EHInstStack.empty()) {
        printAnnotation(OS, "try-catch mismatch!");
      } else if (EHInstStack.back() == CATCH_ALL_LEGACY) {
        printAnnotation(OS, "catch/catch_all cannot occur after catch_all");
      } else if (EHInstStack.back() == TRY) {
        if (TryStack.empty()) {
          printAnnotation(OS, "try-catch mismatch!");
        } else {
          printAnnotation(OS, "catch" + utostr(TryStack.pop_back_val()) + ':');
        }
        EHInstStack.pop_back();
        if (Opc == WebAssembly::CATCH_LEGACY ||
            Opc == WebAssembly::CATCH_LEGACY_S) {
          EHInstStack.push_back(CATCH_LEGACY);
        } else {
          EHInstStack.push_back(CATCH_ALL_LEGACY);
        }
      }
      return;

    case WebAssembly::RETHROW:
    case WebAssembly::RETHROW_S:
      // 'rethrow' rethrows to the nearest enclosing catch scope, if any. If
      // there's no enclosing catch scope, it throws up to the caller.
      if (TryStack.empty()) {
        printAnnotation(OS, "to caller");
      } else {
        printAnnotation(OS, "down to catch" + utostr(TryStack.back()));
      }
      return;

    case WebAssembly::DELEGATE:
    case WebAssembly::DELEGATE_S:
      if (ControlFlowStack.empty() || TryStack.empty() || EHInstStack.empty()) {
        printAnnotation(OS, "try-delegate mismatch!");
      } else {
        // 'delegate' is
        // 1. A marker for the end of block label
        // 2. A destination for throwing instructions
        // 3. An instruction that itself rethrows to another 'catch'
        assert(ControlFlowStack.back().first == TryStack.back());
        std::string Label = "label/catch" +
                            utostr(ControlFlowStack.pop_back_val().first) +
                            ": ";
        TryStack.pop_back();
        EHInstStack.pop_back();
        uint64_t Depth = MI->getOperand(0).getImm();
        if (Depth >= ControlFlowStack.size()) {
          Label += "to caller";
        } else {
          const auto &Pair = ControlFlowStack.rbegin()[Depth];
          if (Pair.second)
            printAnnotation(OS, "delegate cannot target a loop");
          else
            Label += "down to catch" + utostr(Pair.first);
        }
        printAnnotation(OS, Label);
      }
      return;
    }

    // Annotate any control flow label references.

    unsigned NumFixedOperands = Desc.NumOperands;
    SmallSet<uint64_t, 8> Printed;
    for (unsigned I = 0, E = MI->getNumOperands(); I < E; ++I) {
      // See if this operand denotes a basic block target.
      if (I < NumFixedOperands) {
        // A non-variable_ops operand, check its type.
        if (Desc.operands()[I].OperandType != WebAssembly::OPERAND_BASIC_BLOCK)
          continue;
      } else {
        // A variable_ops operand, which currently can be immediates (used in
        // br_table) which are basic block targets, or for call instructions
        // when using -wasm-keep-registers (in which case they are registers,
        // and should not be processed).
        if (!MI->getOperand(I).isImm())
          continue;
      }
      PrintBranchAnnotation(MI->getOperand(I), Printed);
    }
  }
}

static std::string toString(const APFloat &FP) {
  // Print NaNs with custom payloads specially.
  if (FP.isNaN() && !FP.bitwiseIsEqual(APFloat::getQNaN(FP.getSemantics())) &&
      !FP.bitwiseIsEqual(
          APFloat::getQNaN(FP.getSemantics(), /*Negative=*/true))) {
    APInt AI = FP.bitcastToAPInt();
    return std::string(AI.isNegative() ? "-" : "") + "nan:0x" +
           utohexstr(AI.getZExtValue() &
                         (AI.getBitWidth() == 32 ? INT64_C(0x007fffff)
                                                 : INT64_C(0x000fffffffffffff)),
                     /*LowerCase=*/true);
  }

  // Use C99's hexadecimal floating-point representation.
  static const size_t BufBytes = 128;
  char Buf[BufBytes];
  auto Written = FP.convertToHexString(
      Buf, /*HexDigits=*/0, /*UpperCase=*/false, APFloat::rmNearestTiesToEven);
  (void)Written;
  assert(Written != 0);
  assert(Written < BufBytes);
  return Buf;
}

void WebAssemblyInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O, bool IsVariadicDef) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    unsigned WAReg = Op.getReg();
    if (int(WAReg) >= 0)
      printRegName(O, WAReg);
    else if (OpNo >= Desc.getNumDefs() && !IsVariadicDef)
      O << "$pop" << WebAssembly::getWARegStackId(WAReg);
    else if (WAReg != WebAssembly::UnusedReg)
      O << "$push" << WebAssembly::getWARegStackId(WAReg);
    else
      O << "$drop";
    // Add a '=' suffix if this is a def.
    if (OpNo < MII.get(MI->getOpcode()).getNumDefs() || IsVariadicDef)
      O << '=';
  } else if (Op.isImm()) {
    O << Op.getImm();
  } else if (Op.isSFPImm()) {
    O << ::toString(APFloat(APFloat::IEEEsingle(), APInt(32, Op.getSFPImm())));
  } else if (Op.isDFPImm()) {
    O << ::toString(APFloat(APFloat::IEEEdouble(), APInt(64, Op.getDFPImm())));
  } else {
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    // call_indirect instructions have a TYPEINDEX operand that we print
    // as a signature here, such that the assembler can recover this
    // information.
    auto SRE = static_cast<const MCSymbolRefExpr *>(Op.getExpr());
    if (SRE->getKind() == MCSymbolRefExpr::VK_WASM_TYPEINDEX) {
      auto &Sym = static_cast<const MCSymbolWasm &>(SRE->getSymbol());
      O << WebAssembly::signatureToString(Sym.getSignature());
    } else {
      Op.getExpr()->print(O, &MAI);
    }
  }
}

void WebAssemblyInstPrinter::printBrList(const MCInst *MI, unsigned OpNo,
                                         raw_ostream &O) {
  O << "{";
  for (unsigned I = OpNo, E = MI->getNumOperands(); I != E; ++I) {
    if (I != OpNo)
      O << ", ";
    O << MI->getOperand(I).getImm();
  }
  O << "}";
}

void WebAssemblyInstPrinter::printWebAssemblyP2AlignOperand(const MCInst *MI,
                                                            unsigned OpNo,
                                                            raw_ostream &O) {
  int64_t Imm = MI->getOperand(OpNo).getImm();
  if (Imm == WebAssembly::GetDefaultP2Align(MI->getOpcode()))
    return;
  O << ":p2align=" << Imm;
}

void WebAssemblyInstPrinter::printWebAssemblySignatureOperand(const MCInst *MI,
                                                              unsigned OpNo,
                                                              raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    auto Imm = static_cast<unsigned>(Op.getImm());
    if (Imm != wasm::WASM_TYPE_NORESULT)
      O << WebAssembly::anyTypeToString(Imm);
  } else {
    auto Expr = cast<MCSymbolRefExpr>(Op.getExpr());
    auto *Sym = cast<MCSymbolWasm>(&Expr->getSymbol());
    if (Sym->getSignature()) {
      O << WebAssembly::signatureToString(Sym->getSignature());
    } else {
      // Disassembler does not currently produce a signature
      O << "unknown_type";
    }
  }
}

void WebAssemblyInstPrinter::printCatchList(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  unsigned OpIdx = OpNo;
  const MCOperand &Op = MI->getOperand(OpIdx++);
  unsigned NumCatches = Op.getImm();

  auto PrintTagOp = [&](const MCOperand &Op) {
    const MCSymbolRefExpr *TagExpr = nullptr;
    const MCSymbolWasm *TagSym = nullptr;
    if (Op.isExpr()) {
      TagExpr = cast<MCSymbolRefExpr>(Op.getExpr());
      TagSym = cast<MCSymbolWasm>(&TagExpr->getSymbol());
      O << TagSym->getName() << " ";
    } else {
      // When instructions are parsed from the disassembler, we have an
      // immediate tag index and not a tag expr
      O << Op.getImm() << " ";
    }
  };

  for (unsigned I = 0; I < NumCatches; I++) {
    const MCOperand &Op = MI->getOperand(OpIdx++);
    O << "(";
    switch (Op.getImm()) {
    case wasm::WASM_OPCODE_CATCH:
      O << "catch ";
      PrintTagOp(MI->getOperand(OpIdx++));
      break;
    case wasm::WASM_OPCODE_CATCH_REF:
      O << "catch_ref ";
      PrintTagOp(MI->getOperand(OpIdx++));
      break;
    case wasm::WASM_OPCODE_CATCH_ALL:
      O << "catch_all ";
      break;
    case wasm::WASM_OPCODE_CATCH_ALL_REF:
      O << "catch_all_ref ";
      break;
    }
    O << MI->getOperand(OpIdx++).getImm(); // destination
    O << ")";
    if (I < NumCatches - 1)
      O << " ";
  }
}
