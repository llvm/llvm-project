//=- WebAssemblyInstPrinter.cpp - WebAssembly assembly instruction printing -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Print MCInst instructions to wasm format.
///
//===----------------------------------------------------------------------===//

#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
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
  // Note that there's an implicit get_local/set_local here!
  OS << "$" << RegNo;
}

void WebAssemblyInstPrinter::printInst(const MCInst *MI, raw_ostream &OS,
                                       StringRef Annot,
                                       const MCSubtargetInfo &STI) {
  // Print the instruction (this uses the AsmStrings from the .td files).
  printInstruction(MI, OS);

  // Print any additional variadic operands.
  const MCInstrDesc &Desc = MII.get(MI->getOpcode());
  if (Desc.isVariadic())
    for (auto i = Desc.getNumOperands(), e = MI->getNumOperands(); i < e; ++i) {
      // FIXME: For CALL_INDIRECT_VOID, don't print a leading comma, because
      // we have an extra flags operand which is not currently printed, for
      // compatiblity reasons.
      if (i != 0 && ((MI->getOpcode() != WebAssembly::CALL_INDIRECT_VOID &&
                      MI->getOpcode() != WebAssembly::CALL_INDIRECT_VOID_S) ||
                     i != Desc.getNumOperands()))
        OS << ", ";
      printOperand(MI, i, OS);
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
      assert(!ControlFlowStack.empty() && "End marker mismatch!");
      ControlFlowStack.pop_back();
      break;

    case WebAssembly::END_BLOCK:
    case WebAssembly::END_BLOCK_S:
      assert(!ControlFlowStack.empty() && "End marker mismatch!");
      printAnnotation(
          OS, "label" + utostr(ControlFlowStack.pop_back_val().first) + ':');
      break;

    case WebAssembly::END_TRY:
    case WebAssembly::END_TRY_S:
      assert(!ControlFlowStack.empty() && "End marker mismatch!");
      printAnnotation(
          OS, "label" + utostr(ControlFlowStack.pop_back_val().first) + ':');
      LastSeenEHInst = END_TRY;
      break;

    case WebAssembly::CATCH_I32:
    case WebAssembly::CATCH_I32_S:
    case WebAssembly::CATCH_I64:
    case WebAssembly::CATCH_I64_S:
    case WebAssembly::CATCH_ALL:
    case WebAssembly::CATCH_ALL_S:
      // There can be multiple catch instructions for one try instruction, so we
      // print a label only for the first 'catch' label.
      if (LastSeenEHInst != CATCH) {
        assert(!EHPadStack.empty() && "try-catch mismatch!");
        printAnnotation(OS, "catch" + utostr(EHPadStack.pop_back_val()) + ':');
      }
      LastSeenEHInst = CATCH;
      break;
    }

    // Annotate any control flow label references.
    unsigned NumFixedOperands = Desc.NumOperands;
    SmallSet<uint64_t, 8> Printed;
    for (unsigned i = 0, e = MI->getNumOperands(); i < e; ++i) {
      if (!(i < NumFixedOperands
                ? (Desc.OpInfo[i].OperandType ==
                   WebAssembly::OPERAND_BASIC_BLOCK)
                : (Desc.TSFlags & WebAssemblyII::VariableOpImmediateIsLabel)))
        continue;
      uint64_t Depth = MI->getOperand(i).getImm();
      if (!Printed.insert(Depth).second)
        continue;

      if (Opc == WebAssembly::RETHROW || Opc == WebAssembly::RETHROW_S) {
        assert(Depth <= EHPadStack.size() && "Invalid depth argument!");
        if (Depth == EHPadStack.size()) {
          // This can happen when rethrow instruction breaks out of all nests
          // and throws up to the current function's caller.
          printAnnotation(OS, utostr(Depth) + ": " + "to caller");
        } else {
          uint64_t CatchNo = EHPadStack.rbegin()[Depth];
          printAnnotation(OS, utostr(Depth) + ": " + "down to catch" +
                                  utostr(CatchNo));
        }

      } else {
        assert(Depth < ControlFlowStack.size() && "Invalid depth argument!");
        const auto &Pair = ControlFlowStack.rbegin()[Depth];
        printAnnotation(OS, utostr(Depth) + ": " +
                                (Pair.second ? "up" : "down") + " to label" +
                                utostr(Pair.first));
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
  char buf[BufBytes];
  auto Written = FP.convertToHexString(
      buf, /*hexDigits=*/0, /*upperCase=*/false, APFloat::rmNearestTiesToEven);
  (void)Written;
  assert(Written != 0);
  assert(Written < BufBytes);
  return buf;
}

void WebAssemblyInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                          raw_ostream &O) {
  const MCOperand &Op = MI->getOperand(OpNo);
  if (Op.isReg()) {
    assert((OpNo < MII.get(MI->getOpcode()).getNumOperands() ||
            MII.get(MI->getOpcode()).TSFlags == 0) &&
           "WebAssembly variable_ops register ops don't use TSFlags");
    unsigned WAReg = Op.getReg();
    if (int(WAReg) >= 0)
      printRegName(O, WAReg);
    else if (OpNo >= MII.get(MI->getOpcode()).getNumDefs())
      O << "$pop" << WebAssemblyFunctionInfo::getWARegStackId(WAReg);
    else if (WAReg != WebAssemblyFunctionInfo::UnusedReg)
      O << "$push" << WebAssemblyFunctionInfo::getWARegStackId(WAReg);
    else
      O << "$drop";
    // Add a '=' suffix if this is a def.
    if (OpNo < MII.get(MI->getOpcode()).getNumDefs())
      O << '=';
  } else if (Op.isImm()) {
    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    assert((OpNo < Desc.getNumOperands() ||
            (Desc.TSFlags & WebAssemblyII::VariableOpIsImmediate)) &&
           "WebAssemblyII::VariableOpIsImmediate should be set for "
           "variable_ops immediate ops");
    (void)Desc;
    // TODO: (MII.get(MI->getOpcode()).TSFlags &
    //        WebAssemblyII::VariableOpImmediateIsLabel)
    // can tell us whether this is an immediate referencing a label in the
    // control flow stack, and it may be nice to pretty-print.
    O << Op.getImm();
  } else if (Op.isFPImm()) {
    const MCInstrDesc &Desc = MII.get(MI->getOpcode());
    assert(OpNo < Desc.getNumOperands() &&
           "Unexpected floating-point immediate as a non-fixed operand");
    assert(Desc.TSFlags == 0 &&
           "WebAssembly variable_ops floating point ops don't use TSFlags");
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
    assert((OpNo < MII.get(MI->getOpcode()).getNumOperands() ||
            (MII.get(MI->getOpcode()).TSFlags &
             WebAssemblyII::VariableOpIsImmediate)) &&
           "WebAssemblyII::VariableOpIsImmediate should be set for "
           "variable_ops expr ops");
    assert(Op.isExpr() && "unknown operand kind in printOperand");
    Op.getExpr()->print(O, &MAI);
  }
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
  int64_t Imm = MI->getOperand(OpNo).getImm();
  switch (WebAssembly::ExprType(Imm)) {
  case WebAssembly::ExprType::Void:
    break;
  case WebAssembly::ExprType::I32:
    O << "i32";
    break;
  case WebAssembly::ExprType::I64:
    O << "i64";
    break;
  case WebAssembly::ExprType::F32:
    O << "f32";
    break;
  case WebAssembly::ExprType::F64:
    O << "f64";
    break;
  case WebAssembly::ExprType::V128:
    O << "v128";
    break;
  case WebAssembly::ExprType::ExceptRef:
    O << "except_ref";
    break;
  }
}

const char *llvm::WebAssembly::TypeToString(wasm::ValType Ty) {
  switch (Ty) {
  case wasm::ValType::I32:
    return "i32";
  case wasm::ValType::I64:
    return "i64";
  case wasm::ValType::F32:
    return "f32";
  case wasm::ValType::F64:
    return "f64";
  case wasm::ValType::V128:
    return "v128";
  case wasm::ValType::EXCEPT_REF:
    return "except_ref";
  }
  llvm_unreachable("Unknown wasm::ValType");
}
