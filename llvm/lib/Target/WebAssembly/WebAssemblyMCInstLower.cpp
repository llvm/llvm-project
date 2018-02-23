// WebAssemblyMCInstLower.cpp - Convert WebAssembly MachineInstr to an MCInst //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains code to lower WebAssembly MachineInstrs to their
/// corresponding MCInst records.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCInstLower.h"
#include "WebAssemblyAsmPrinter.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRuntimeLibcallSignatures.h"
#include "WebAssemblyUtilities.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

MCSymbol *
WebAssemblyMCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  const GlobalValue *Global = MO.getGlobal();
  MCSymbol *Sym = Printer.getSymbol(Global);
  if (isa<MCSymbolELF>(Sym))
    return Sym;

  MCSymbolWasm *WasmSym = cast<MCSymbolWasm>(Sym);

  if (const auto *FuncTy = dyn_cast<FunctionType>(Global->getValueType())) {
    const MachineFunction &MF = *MO.getParent()->getParent()->getParent();
    const TargetMachine &TM = MF.getTarget();
    const Function &CurrentFunc = MF.getFunction();

    SmallVector<wasm::ValType, 4> Returns;
    SmallVector<wasm::ValType, 4> Params;

    wasm::ValType iPTR =
        MF.getSubtarget<WebAssemblySubtarget>().hasAddr64() ?
        wasm::ValType::I64 :
        wasm::ValType::I32;

    SmallVector<MVT, 4> ResultMVTs;
    ComputeLegalValueVTs(CurrentFunc, TM, FuncTy->getReturnType(), ResultMVTs);
    // WebAssembly can't currently handle returning tuples.
    if (ResultMVTs.size() <= 1)
      for (MVT ResultMVT : ResultMVTs)
        Returns.push_back(WebAssembly::toValType(ResultMVT));
    else
      Params.push_back(iPTR);

    for (Type *Ty : FuncTy->params()) {
      SmallVector<MVT, 4> ParamMVTs;
      ComputeLegalValueVTs(CurrentFunc, TM, Ty, ParamMVTs);
      for (MVT ParamMVT : ParamMVTs)
        Params.push_back(WebAssembly::toValType(ParamMVT));
    }

    if (FuncTy->isVarArg())
      Params.push_back(iPTR);

    WasmSym->setReturns(std::move(Returns));
    WasmSym->setParams(std::move(Params));
    WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
  }

  return WasmSym;
}

MCSymbol *WebAssemblyMCInstLower::GetExternalSymbolSymbol(
    const MachineOperand &MO) const {
  const char *Name = MO.getSymbolName();
  MCSymbol *Sym = Printer.GetExternalSymbolSymbol(Name);
  if (isa<MCSymbolELF>(Sym))
    return Sym;

  MCSymbolWasm *WasmSym = cast<MCSymbolWasm>(Sym);
  const WebAssemblySubtarget &Subtarget = Printer.getSubtarget();

  // __stack_pointer is a global variable; all other external symbols used by
  // CodeGen are functions.  It's OK to hardcode knowledge of specific symbols
  // here; this method is precisely there for fetching the signatures of known
  // Clang-provided symbols.
  if (strcmp(Name, "__stack_pointer") == 0) {
    wasm::ValType iPTR =
        Subtarget.hasAddr64() ? wasm::ValType::I64 : wasm::ValType::I32;
    WasmSym->setType(wasm::WASM_SYMBOL_TYPE_GLOBAL);
    WasmSym->setGlobalType(wasm::WasmGlobalType{int32_t(iPTR), true});
    return WasmSym;
  }

  SmallVector<wasm::ValType, 4> Returns;
  SmallVector<wasm::ValType, 4> Params;
  GetSignature(Subtarget, Name, Returns, Params);

  WasmSym->setReturns(std::move(Returns));
  WasmSym->setParams(std::move(Params));
  WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);

  return WasmSym;
}

MCOperand WebAssemblyMCInstLower::LowerSymbolOperand(MCSymbol *Sym,
                                                     int64_t Offset,
                                                     bool IsFunc) const {
  MCSymbolRefExpr::VariantKind VK =
      IsFunc ? MCSymbolRefExpr::VK_WebAssembly_FUNCTION
             : MCSymbolRefExpr::VK_None;

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, VK, Ctx);

  if (Offset != 0) {
    if (IsFunc)
      report_fatal_error("Function addresses with offsets not supported");
    Expr =
        MCBinaryExpr::createAdd(Expr, MCConstantExpr::create(Offset, Ctx), Ctx);
  }

  return MCOperand::createExpr(Expr);
}

// Return the WebAssembly type associated with the given register class.
static wasm::ValType getType(const TargetRegisterClass *RC) {
  if (RC == &WebAssembly::I32RegClass)
    return wasm::ValType::I32;
  if (RC == &WebAssembly::I64RegClass)
    return wasm::ValType::I64;
  if (RC == &WebAssembly::F32RegClass)
    return wasm::ValType::F32;
  if (RC == &WebAssembly::F64RegClass)
    return wasm::ValType::F64;
  llvm_unreachable("Unexpected register class");
}

void WebAssemblyMCInstLower::Lower(const MachineInstr *MI,
                                   MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  const MCInstrDesc &Desc = MI->getDesc();
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);

    MCOperand MCOp;
    switch (MO.getType()) {
    default:
      MI->print(errs());
      llvm_unreachable("unknown operand type");
    case MachineOperand::MO_MachineBasicBlock:
      MI->print(errs());
      llvm_unreachable("MachineBasicBlock operand should have been rewritten");
    case MachineOperand::MO_Register: {
      // Ignore all implicit register operands.
      if (MO.isImplicit())
        continue;
      const WebAssemblyFunctionInfo &MFI =
          *MI->getParent()->getParent()->getInfo<WebAssemblyFunctionInfo>();
      unsigned WAReg = MFI.getWAReg(MO.getReg());
      MCOp = MCOperand::createReg(WAReg);
      break;
    }
    case MachineOperand::MO_Immediate:
      if (i < Desc.NumOperands) {
        const MCOperandInfo &Info = Desc.OpInfo[i];
        if (Info.OperandType == WebAssembly::OPERAND_TYPEINDEX) {
          MCSymbol *Sym = Printer.createTempSymbol("typeindex");
          if (!isa<MCSymbolELF>(Sym)) {
            SmallVector<wasm::ValType, 4> Returns;
            SmallVector<wasm::ValType, 4> Params;

            const MachineRegisterInfo &MRI =
                MI->getParent()->getParent()->getRegInfo();
            for (const MachineOperand &MO : MI->defs())
              Returns.push_back(getType(MRI.getRegClass(MO.getReg())));
            for (const MachineOperand &MO : MI->explicit_uses())
              if (MO.isReg())
                Params.push_back(getType(MRI.getRegClass(MO.getReg())));

            // call_indirect instructions have a callee operand at the end which
            // doesn't count as a param.
            if (WebAssembly::isCallIndirect(*MI))
              Params.pop_back();

            MCSymbolWasm *WasmSym = cast<MCSymbolWasm>(Sym);
            WasmSym->setReturns(std::move(Returns));
            WasmSym->setParams(std::move(Params));
            WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);

            const MCExpr *Expr =
                MCSymbolRefExpr::create(WasmSym,
                                        MCSymbolRefExpr::VK_WebAssembly_TYPEINDEX,
                                        Ctx);
            MCOp = MCOperand::createExpr(Expr);
            break;
          }
        }
      }
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    case MachineOperand::MO_FPImmediate: {
      // TODO: MC converts all floating point immediate operands to double.
      // This is fine for numeric values, but may cause NaNs to change bits.
      const ConstantFP *Imm = MO.getFPImm();
      if (Imm->getType()->isFloatTy())
        MCOp = MCOperand::createFPImm(Imm->getValueAPF().convertToFloat());
      else if (Imm->getType()->isDoubleTy())
        MCOp = MCOperand::createFPImm(Imm->getValueAPF().convertToDouble());
      else
        llvm_unreachable("unknown floating point immediate type");
      break;
    }
    case MachineOperand::MO_GlobalAddress:
      assert(MO.getTargetFlags() == 0 &&
             "WebAssembly does not use target flags on GlobalAddresses");
      MCOp = LowerSymbolOperand(GetGlobalAddressSymbol(MO), MO.getOffset(),
                                MO.getGlobal()->getValueType()->isFunctionTy());
      break;
    case MachineOperand::MO_ExternalSymbol:
      // The target flag indicates whether this is a symbol for a
      // variable or a function.
      assert((MO.getTargetFlags() & -2) == 0 &&
             "WebAssembly uses only one target flag bit on ExternalSymbols");
      MCOp = LowerSymbolOperand(GetExternalSymbolSymbol(MO), /*Offset=*/0,
                                MO.getTargetFlags() & 1);
      break;
    }

    OutMI.addOperand(MCOp);
  }
}
