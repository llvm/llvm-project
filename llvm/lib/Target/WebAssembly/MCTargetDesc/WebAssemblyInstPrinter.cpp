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
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
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
                                          unsigned RegNo) const {
  assert(RegNo != WebAssemblyFunctionInfo::UnusedReg);
  // Note that there's an implicit local.get/local.set here!
  OS << "$" << RegNo;
}

void WebAssemblyInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                       StringRef Annot,
                                       const MCSubtargetInfo &STI,
                                       raw_ostream &OS) {
  // Print the instruction (this uses the AsmStrings from the .td files).
  printInstruction(MI, Address, OS);

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
        // Skip type and flags arguments when printing for tests
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
      break;

    case WebAssembly::BLOCK:
    case WebAssembly::BLOCK_S:
      ControlFlowStack.push_back(std::make_pair(ControlFlowCounter++, false));
      break;

    case WebAssembly::TRY:
    case WebAssembly::TRY_S:
      ControlFlowStack.push_back(std::make_pair(ControlFlowCounter++, false));
      EHPadStack.push_back(EHPadStackCounter++);
      LastSeenEHInst = TRY;
      break;

    case WebAssembly::END_LOOP:
    case WebAssembly::END_LOOP_S:
      if (ControlFlowStack.empty()) {
        printAnnotation(OS, "End marker mismatch!");
      } else {
        ControlFlowStack.pop_back();
      }
      break;

    case WebAssembly::END_BLOCK:
    case WebAssembly::END_BLOCK_S:
      if (ControlFlowStack.empty()) {
        printAnnotation(OS, "End marker mismatch!");
      } else {
        printAnnotation(
            OS, "label" + utostr(ControlFlowStack.pop_back_val().first) + ':');
      }
      break;

    case WebAssembly::END_TRY:
    case WebAssembly::END_TRY_S:
      if (ControlFlowStack.empty()) {
        printAnnotation(OS, "End marker mismatch!");
      } else {
        printAnnotation(
            OS, "label" + utostr(ControlFlowStack.pop_back_val().first) + ':');
        LastSeenEHInst = END_TRY;
      }
      break;

    case WebAssembly::CATCH:
    case WebAssembly::CATCH_S:
      if (EHPadStack.empty()) {
        printAnnotation(OS, "try-catch mismatch!");
      } else {
        printAnnotation(OS, "catch" + utostr(EHPadStack.pop_back_val()) + ':');
      }
      break;
    }

    // Annotate any control flow label references.

    // rethrow instruction does not take any depth argument and rethrows to the
    // nearest enclosing catch scope, if any. If there's no enclosing catch
    // scope, it throws up to the caller.
    if (Opc == WebAssembly::RETHROW || Opc == WebAssembly::RETHROW_S) {
      if (EHPadStack.empty()) {
        printAnnotation(OS, "to caller");
      } else {
        printAnnotation(OS, "down to catch" + utostr(EHPadStack.back()));
      }

    } else {
      unsigned NumFixedOperands = Desc.NumOperands;
      SmallSet<uint64_t, 8> Printed;
      for (unsigned I = 0, E = MI->getNumOperands(); I < E; ++I) {
        // See if this operand denotes a basic block target.
        if (I < NumFixedOperands) {
          // A non-variable_ops operand, check its type.
          if (Desc.OpInfo[I].OperandType != WebAssembly::OPERAND_BASIC_BLOCK)
            continue;
        } else {
          // A variable_ops operand, which currently can be immediates (used in
          // br_table) which are basic block targets, or for call instructions
          // when using -wasm-keep-registers (in which case they are registers,
          // and should not be processed).
          if (!MI->getOperand(I).isImm())
            continue;
        }
        uint64_t Depth = MI->getOperand(I).getImm();
        if (!Printed.insert(Depth).second)
          continue;
        if (Depth >= ControlFlowStack.size()) {
          printAnnotation(OS, "Invalid depth argument!");
        } else {
          const auto &Pair = ControlFlowStack.rbegin()[Depth];
          printAnnotation(OS, utostr(Depth) + ": " +
                                  (Pair.second ? "up" : "down") + " to label" +
                                  utostr(Pair.first));
        }
      }
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
      O << "$pop" << WebAssemblyFunctionInfo::getWARegStackId(WAReg);
    else if (WAReg != WebAssemblyFunctionInfo::UnusedReg)
      O << "$push" << WebAssemblyFunctionInfo::getWARegStackId(WAReg);
    else
      O << "$drop";
    // Add a '=' suffix if this is a def.
    if (OpNo < MII.get(MI->getOpcode()).getNumDefs() || IsVariadicDef)
      O << '=';
  } else if (Op.isImm()) {
    O << Op.getImm();
  } else if (Op.isFPImm()) {
    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    const MCOperandInfo &Info = Desc.OpInfo[OpNo];
    if (Info.OperandType == WebAssembly::OPERAND_F32IMM) {
      // TODO: MC converts all floating point immediate operands to double.
      // This is fine for numeric values, but may cause NaNs to change bits.
      O << ::toString(APFloat(float(Op.getFPImm())));
    } else {
      assert(Info.OperandType == WebAssembly::OPERAND_F64IMM);
      O << ::toString(APFloat(Op.getFPImm()));
    }
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

void WebAssemblyInstPrinter::printWebAssemblyHeapTypeOperand(const MCInst *MI,
                                                             unsigned OpNo,
                                                             raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isImm()) {
    switch (Op.getImm()) {
    case long(wasm::ValType::EXTERNREF):
      O << "extern";
      break;
    case long(wasm::ValType::FUNCREF):
      O << "func";
      break;
    default:
      O << "unsupported_heap_type_value";
      break;
    }
  } else {
    // Typed function references and other subtypes of funcref and externref
    // currently unimplemented.
    O << "unsupported_heap_type_operand";
  }
}

// We have various enums representing a subset of these types, use this
// function to convert any of them to text.
const char *WebAssembly::anyTypeToString(unsigned Ty) {
  switch (Ty) {
  case wasm::WASM_TYPE_I32:
    return "i32";
  case wasm::WASM_TYPE_I64:
    return "i64";
  case wasm::WASM_TYPE_F32:
    return "f32";
  case wasm::WASM_TYPE_F64:
    return "f64";
  case wasm::WASM_TYPE_V128:
    return "v128";
  case wasm::WASM_TYPE_FUNCREF:
    return "funcref";
  case wasm::WASM_TYPE_EXTERNREF:
    return "externref";
  case wasm::WASM_TYPE_FUNC:
    return "func";
  case wasm::WASM_TYPE_EXNREF:
    return "exnref";
  case wasm::WASM_TYPE_NORESULT:
    return "void";
  default:
    return "invalid_type";
  }
}

const char *WebAssembly::typeToString(wasm::ValType Ty) {
  return anyTypeToString(static_cast<unsigned>(Ty));
}

std::string WebAssembly::typeListToString(ArrayRef<wasm::ValType> List) {
  std::string S;
  for (auto &Ty : List) {
    if (&Ty != &List[0]) S += ", ";
    S += WebAssembly::typeToString(Ty);
  }
  return S;
}

std::string WebAssembly::signatureToString(const wasm::WasmSignature *Sig) {
  std::string S("(");
  S += typeListToString(Sig->Params);
  S += ") -> (";
  S += typeListToString(Sig->Returns);
  S += ")";
  return S;
}
