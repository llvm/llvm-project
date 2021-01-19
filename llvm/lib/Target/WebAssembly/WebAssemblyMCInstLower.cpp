// WebAssemblyMCInstLower.cpp - Convert WebAssembly MachineInstr to an MCInst //
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains code to lower WebAssembly MachineInstrs to their
/// corresponding MCInst records.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyMCInstLower.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "TargetInfo/WebAssemblyTargetInfo.h"
#include "WebAssemblyAsmPrinter.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRuntimeLibcallSignatures.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/Constants.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// This disables the removal of registers when lowering into MC, as required
// by some current tests.
cl::opt<bool>
    WasmKeepRegisters("wasm-keep-registers", cl::Hidden,
                      cl::desc("WebAssembly: output stack registers in"
                               " instruction output for test purposes only."),
                      cl::init(false));

extern cl::opt<bool> EnableEmException;
extern cl::opt<bool> EnableEmSjLj;

static void removeRegisterOperands(const MachineInstr *MI, MCInst &OutMI);

MCSymbol *
WebAssemblyMCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  const GlobalValue *Global = MO.getGlobal();
  if (!isa<Function>(Global))
    return cast<MCSymbolWasm>(Printer.getSymbol(Global));

  const auto *FuncTy = cast<FunctionType>(Global->getValueType());
  const MachineFunction &MF = *MO.getParent()->getParent()->getParent();
  const TargetMachine &TM = MF.getTarget();
  const Function &CurrentFunc = MF.getFunction();

  SmallVector<MVT, 1> ResultMVTs;
  SmallVector<MVT, 4> ParamMVTs;
  const auto *const F = dyn_cast<Function>(Global);
  computeSignatureVTs(FuncTy, F, CurrentFunc, TM, ParamMVTs, ResultMVTs);
  auto Signature = signatureFromMVTs(ResultMVTs, ParamMVTs);

  bool InvokeDetected = false;
  auto *WasmSym = Printer.getMCSymbolForFunction(
      F, EnableEmException || EnableEmSjLj, Signature.get(), InvokeDetected);
  WasmSym->setSignature(Signature.get());
  Printer.addSignature(std::move(Signature));
  WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
  return WasmSym;
}

MCSymbol *WebAssemblyMCInstLower::GetExternalSymbolSymbol(
    const MachineOperand &MO) const {
  const char *Name = MO.getSymbolName();
  auto *WasmSym = cast<MCSymbolWasm>(Printer.GetExternalSymbolSymbol(Name));
  const WebAssemblySubtarget &Subtarget = Printer.getSubtarget();

  // Except for certain known symbols, all symbols used by CodeGen are
  // functions. It's OK to hardcode knowledge of specific symbols here; this
  // method is precisely there for fetching the signatures of known
  // Clang-provided symbols.
  if (strcmp(Name, "__stack_pointer") == 0 || strcmp(Name, "__tls_base") == 0 ||
      strcmp(Name, "__memory_base") == 0 || strcmp(Name, "__table_base") == 0 ||
      strcmp(Name, "__tls_size") == 0 || strcmp(Name, "__tls_align") == 0) {
    bool Mutable =
        strcmp(Name, "__stack_pointer") == 0 || strcmp(Name, "__tls_base") == 0;
    WasmSym->setType(wasm::WASM_SYMBOL_TYPE_GLOBAL);
    WasmSym->setGlobalType(wasm::WasmGlobalType{
        uint8_t(Subtarget.hasAddr64() && strcmp(Name, "__table_base") != 0
                    ? wasm::WASM_TYPE_I64
                    : wasm::WASM_TYPE_I32),
        Mutable});
    return WasmSym;
  }

  SmallVector<wasm::ValType, 4> Returns;
  SmallVector<wasm::ValType, 4> Params;
  if (strcmp(Name, "__cpp_exception") == 0) {
    WasmSym->setType(wasm::WASM_SYMBOL_TYPE_EVENT);
    // We can't confirm its signature index for now because there can be
    // imported exceptions. Set it to be 0 for now.
    WasmSym->setEventType(
        {wasm::WASM_EVENT_ATTRIBUTE_EXCEPTION, /* SigIndex */ 0});
    // We may have multiple C++ compilation units to be linked together, each of
    // which defines the exception symbol. To resolve them, we declare them as
    // weak.
    WasmSym->setWeak(true);
    WasmSym->setExternal(true);

    // All C++ exceptions are assumed to have a single i32 (for wasm32) or i64
    // (for wasm64) param type and void return type. The reaon is, all C++
    // exception values are pointers, and to share the type section with
    // functions, exceptions are assumed to have void return type.
    Params.push_back(Subtarget.hasAddr64() ? wasm::ValType::I64
                                           : wasm::ValType::I32);
  } else { // Function symbols
    WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
    getLibcallSignature(Subtarget, Name, Returns, Params);
  }
  auto Signature =
      std::make_unique<wasm::WasmSignature>(std::move(Returns), std::move(Params));
  WasmSym->setSignature(Signature.get());
  Printer.addSignature(std::move(Signature));

  return WasmSym;
}

MCOperand WebAssemblyMCInstLower::lowerSymbolOperand(const MachineOperand &MO,
                                                     MCSymbol *Sym) const {
  MCSymbolRefExpr::VariantKind Kind = MCSymbolRefExpr::VK_None;
  unsigned TargetFlags = MO.getTargetFlags();

  switch (TargetFlags) {
    case WebAssemblyII::MO_NO_FLAG:
      break;
    case WebAssemblyII::MO_GOT:
      Kind = MCSymbolRefExpr::VK_GOT;
      break;
    case WebAssemblyII::MO_MEMORY_BASE_REL:
      Kind = MCSymbolRefExpr::VK_WASM_MBREL;
      break;
    case WebAssemblyII::MO_TLS_BASE_REL:
      Kind = MCSymbolRefExpr::VK_WASM_TLSREL;
      break;
    case WebAssemblyII::MO_TABLE_BASE_REL:
      Kind = MCSymbolRefExpr::VK_WASM_TBREL;
      break;
    default:
      llvm_unreachable("Unknown target flag on GV operand");
  }

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, Kind, Ctx);

  if (MO.getOffset() != 0) {
    const auto *WasmSym = cast<MCSymbolWasm>(Sym);
    if (TargetFlags == WebAssemblyII::MO_GOT)
      report_fatal_error("GOT symbol references do not support offsets");
    if (WasmSym->isFunction())
      report_fatal_error("Function addresses with offsets not supported");
    if (WasmSym->isGlobal())
      report_fatal_error("Global indexes with offsets not supported");
    if (WasmSym->isEvent())
      report_fatal_error("Event indexes with offsets not supported");

    Expr = MCBinaryExpr::createAdd(
        Expr, MCConstantExpr::create(MO.getOffset(), Ctx), Ctx);
  }

  return MCOperand::createExpr(Expr);
}

MCOperand WebAssemblyMCInstLower::lowerTypeIndexOperand(
    SmallVector<wasm::ValType, 1> &&Returns,
    SmallVector<wasm::ValType, 4> &&Params) const {
  auto Signature = std::make_unique<wasm::WasmSignature>(std::move(Returns),
                                                         std::move(Params));
  MCSymbol *Sym = Printer.createTempSymbol("typeindex");
  auto *WasmSym = cast<MCSymbolWasm>(Sym);
  WasmSym->setSignature(Signature.get());
  Printer.addSignature(std::move(Signature));
  WasmSym->setType(wasm::WASM_SYMBOL_TYPE_FUNCTION);
  const MCExpr *Expr =
      MCSymbolRefExpr::create(WasmSym, MCSymbolRefExpr::VK_WASM_TYPEINDEX, Ctx);
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
  if (RC == &WebAssembly::V128RegClass)
    return wasm::ValType::V128;
  llvm_unreachable("Unexpected register class");
}

static void getFunctionReturns(const MachineInstr *MI,
                               SmallVectorImpl<wasm::ValType> &Returns) {
  const Function &F = MI->getMF()->getFunction();
  const TargetMachine &TM = MI->getMF()->getTarget();
  Type *RetTy = F.getReturnType();
  SmallVector<MVT, 4> CallerRetTys;
  computeLegalValueVTs(F, TM, RetTy, CallerRetTys);
  valTypesFromMVTs(CallerRetTys, Returns);
}

void WebAssemblyMCInstLower::lower(const MachineInstr *MI,
                                   MCInst &OutMI) const {
  OutMI.setOpcode(MI->getOpcode());

  const MCInstrDesc &Desc = MI->getDesc();
  unsigned NumVariadicDefs = MI->getNumExplicitDefs() - Desc.getNumDefs();
  for (unsigned I = 0, E = MI->getNumOperands(); I != E; ++I) {
    const MachineOperand &MO = MI->getOperand(I);

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
    case MachineOperand::MO_Immediate: {
      unsigned DescIndex = I - NumVariadicDefs;
      if (DescIndex < Desc.NumOperands) {
        const MCOperandInfo &Info = Desc.OpInfo[DescIndex];
        if (Info.OperandType == WebAssembly::OPERAND_TYPEINDEX) {
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
          if (WebAssembly::isCallIndirect(MI->getOpcode()))
            Params.pop_back();

          // return_call_indirect instructions have the return type of the
          // caller
          if (MI->getOpcode() == WebAssembly::RET_CALL_INDIRECT)
            getFunctionReturns(MI, Returns);

          MCOp = lowerTypeIndexOperand(std::move(Returns), std::move(Params));
          break;
        } else if (Info.OperandType == WebAssembly::OPERAND_SIGNATURE) {
          auto BT = static_cast<WebAssembly::BlockType>(MO.getImm());
          assert(BT != WebAssembly::BlockType::Invalid);
          if (BT == WebAssembly::BlockType::Multivalue) {
            SmallVector<wasm::ValType, 1> Returns;
            getFunctionReturns(MI, Returns);
            MCOp = lowerTypeIndexOperand(std::move(Returns),
                                         SmallVector<wasm::ValType, 4>());
            break;
          }
        } else if (Info.OperandType == WebAssembly::OPERAND_HEAPTYPE) {
          assert(static_cast<WebAssembly::HeapType>(MO.getImm()) !=
                 WebAssembly::HeapType::Invalid);
          // With typed function references, this will need a case for type
          // index operands.  Otherwise, fall through.
        }
      }
      MCOp = MCOperand::createImm(MO.getImm());
      break;
    }
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
      MCOp = lowerSymbolOperand(MO, GetGlobalAddressSymbol(MO));
      break;
    case MachineOperand::MO_ExternalSymbol:
      // The target flag indicates whether this is a symbol for a
      // variable or a function.
      assert(MO.getTargetFlags() == 0 &&
             "WebAssembly uses only symbol flags on ExternalSymbols");
      MCOp = lowerSymbolOperand(MO, GetExternalSymbolSymbol(MO));
      break;
    case MachineOperand::MO_MCSymbol:
      // This is currently used only for LSDA symbols (GCC_except_table),
      // because global addresses or other external symbols are handled above.
      assert(MO.getTargetFlags() == 0 &&
             "WebAssembly does not use target flags on MCSymbol");
      MCOp = lowerSymbolOperand(MO, MO.getMCSymbol());
      break;
    }

    OutMI.addOperand(MCOp);
  }

  if (!WasmKeepRegisters)
    removeRegisterOperands(MI, OutMI);
  else if (Desc.variadicOpsAreDefs())
    OutMI.insert(OutMI.begin(), MCOperand::createImm(MI->getNumExplicitDefs()));
}

static void removeRegisterOperands(const MachineInstr *MI, MCInst &OutMI) {
  // Remove all uses of stackified registers to bring the instruction format
  // into its final stack form used thruout MC, and transition opcodes to
  // their _S variant.
  // We do this separate from the above code that still may need these
  // registers for e.g. call_indirect signatures.
  // See comments in lib/Target/WebAssembly/WebAssemblyInstrFormats.td for
  // details.
  // TODO: the code above creates new registers which are then removed here.
  // That code could be slightly simplified by not doing that, though maybe
  // it is simpler conceptually to keep the code above in "register mode"
  // until this transition point.
  // FIXME: we are not processing inline assembly, which contains register
  // operands, because it is used by later target generic code.
  if (MI->isDebugInstr() || MI->isLabel() || MI->isInlineAsm())
    return;

  // Transform to _S instruction.
  auto RegOpcode = OutMI.getOpcode();
  auto StackOpcode = WebAssembly::getStackOpcode(RegOpcode);
  assert(StackOpcode != -1 && "Failed to stackify instruction");
  OutMI.setOpcode(StackOpcode);

  // Remove register operands.
  for (auto I = OutMI.getNumOperands(); I; --I) {
    auto &MO = OutMI.getOperand(I - 1);
    if (MO.isReg()) {
      OutMI.erase(&MO);
    }
  }
}
