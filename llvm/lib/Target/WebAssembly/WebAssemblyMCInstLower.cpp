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
/// This file contains code to lower WebAssembly MachineInstrs to their
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
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

// This disables the removal of registers when lowering into MC, as required
// by some current tests.
static cl::opt<bool> WasmKeepRegisters(
    "wasm-keep-registers", cl::Hidden,
    cl::desc("WebAssembly: output stack registers in"
             " instruction output for test purposes only."),
    cl::init(false));

static unsigned regInstructionToStackInstruction(unsigned OpCode);
static void removeRegisterOperands(const MachineInstr *MI, MCInst &OutMI);

MCSymbol *
WebAssemblyMCInstLower::GetGlobalAddressSymbol(const MachineOperand &MO) const {
  const GlobalValue *Global = MO.getGlobal();
  MCSymbolWasm *WasmSym = cast<MCSymbolWasm>(Printer.getSymbol(Global));

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
  MCSymbolWasm *WasmSym =
      cast<MCSymbolWasm>(Printer.GetExternalSymbolSymbol(Name));
  const WebAssemblySubtarget &Subtarget = Printer.getSubtarget();

  // __stack_pointer is a global variable; all other external symbols used by
  // CodeGen are functions.  It's OK to hardcode knowledge of specific symbols
  // here; this method is precisely there for fetching the signatures of known
  // Clang-provided symbols.
  if (strcmp(Name, "__stack_pointer") == 0) {
    WasmSym->setType(wasm::WASM_SYMBOL_TYPE_GLOBAL);
    WasmSym->setGlobalType(wasm::WasmGlobalType{
        uint8_t(Subtarget.hasAddr64() ? wasm::WASM_TYPE_I64
                                      : wasm::WASM_TYPE_I32),
        true});
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
                                                     bool IsFunc,
                                                     bool IsGlob) const {
  MCSymbolRefExpr::VariantKind VK =
      IsFunc ? MCSymbolRefExpr::VK_WebAssembly_FUNCTION :
      IsGlob ? MCSymbolRefExpr::VK_WebAssembly_GLOBAL
             : MCSymbolRefExpr::VK_None;

  const MCExpr *Expr = MCSymbolRefExpr::create(Sym, VK, Ctx);

  if (Offset != 0) {
    if (IsFunc)
      report_fatal_error("Function addresses with offsets not supported");
    if (IsGlob)
      report_fatal_error("Global indexes with offsets not supported");
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

          const MCExpr *Expr = MCSymbolRefExpr::create(
              WasmSym, MCSymbolRefExpr::VK_WebAssembly_TYPEINDEX, Ctx);
          MCOp = MCOperand::createExpr(Expr);
          break;
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
      assert(MO.getTargetFlags() == WebAssemblyII::MO_NO_FLAG &&
             "WebAssembly does not use target flags on GlobalAddresses");
      MCOp = LowerSymbolOperand(GetGlobalAddressSymbol(MO), MO.getOffset(),
                                MO.getGlobal()->getValueType()->isFunctionTy(),
                                false);
      break;
    case MachineOperand::MO_ExternalSymbol:
      // The target flag indicates whether this is a symbol for a
      // variable or a function.
      assert((MO.getTargetFlags() & ~WebAssemblyII::MO_SYMBOL_MASK) == 0 &&
             "WebAssembly uses only symbol flags on ExternalSymbols");
      MCOp = LowerSymbolOperand(GetExternalSymbolSymbol(MO), /*Offset=*/0,
          (MO.getTargetFlags() & WebAssemblyII::MO_SYMBOL_FUNCTION) != 0,
          (MO.getTargetFlags() & WebAssemblyII::MO_SYMBOL_GLOBAL) != 0);
      break;
    }

    OutMI.addOperand(MCOp);
  }

  if (!WasmKeepRegisters)
    removeRegisterOperands(MI, OutMI);
}

static void removeRegisterOperands(const MachineInstr *MI, MCInst &OutMI) {
  // Remove all uses of stackified registers to bring the instruction format
  // into its final stack form used thruout MC, and transition opcodes to
  // their _S variant.
  // We do this seperate from the above code that still may need these
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
  auto StackOpcode = regInstructionToStackInstruction(RegOpcode);
  OutMI.setOpcode(StackOpcode);

  // Remove register operands.
  for (auto I = OutMI.getNumOperands(); I; --I) {
    auto &MO = OutMI.getOperand(I - 1);
    if (MO.isReg()) {
      OutMI.erase(&MO);
    }
  }
}

static unsigned regInstructionToStackInstruction(unsigned OpCode) {
  switch (OpCode) {
  default:
    // You may hit this if you add new instructions, please add them below.
    // For most of these opcodes, this function could have been implemented
    // as "return OpCode + 1", but since table-gen alphabetically sorts them,
    // this cannot be guaranteed (see e.g. BR and BR_IF).
    // The approach below is the same as what the x87 backend does.
    // TODO(wvo): to make this code cleaner, create a custom tablegen
    // code generator that emits the table below automatically.
    llvm_unreachable(
          "unknown WebAssembly instruction in Explicit Locals pass");
  case WebAssembly::ABS_F32: return WebAssembly::ABS_F32_S;
  case WebAssembly::ABS_F64: return WebAssembly::ABS_F64_S;
  case WebAssembly::ADD_F32: return WebAssembly::ADD_F32_S;
  case WebAssembly::ADD_F64: return WebAssembly::ADD_F64_S;
  case WebAssembly::ADD_I32: return WebAssembly::ADD_I32_S;
  case WebAssembly::ADD_I64: return WebAssembly::ADD_I64_S;
  case WebAssembly::ADD_v16i8: return WebAssembly::ADD_v16i8_S;
  case WebAssembly::ADD_v2f64: return WebAssembly::ADD_v2f64_S;
  case WebAssembly::ADD_v2i64: return WebAssembly::ADD_v2i64_S;
  case WebAssembly::ADD_v4f32: return WebAssembly::ADD_v4f32_S;
  case WebAssembly::ADD_v4i32: return WebAssembly::ADD_v4i32_S;
  case WebAssembly::ADD_v8i16: return WebAssembly::ADD_v8i16_S;
  case WebAssembly::ADJCALLSTACKDOWN: return WebAssembly::ADJCALLSTACKDOWN_S;
  case WebAssembly::ADJCALLSTACKUP: return WebAssembly::ADJCALLSTACKUP_S;
  case WebAssembly::AND_I32: return WebAssembly::AND_I32_S;
  case WebAssembly::AND_I64: return WebAssembly::AND_I64_S;
  case WebAssembly::ARGUMENT_EXCEPT_REF: return WebAssembly::ARGUMENT_EXCEPT_REF_S;
  case WebAssembly::ARGUMENT_F32: return WebAssembly::ARGUMENT_F32_S;
  case WebAssembly::ARGUMENT_F64: return WebAssembly::ARGUMENT_F64_S;
  case WebAssembly::ARGUMENT_I32: return WebAssembly::ARGUMENT_I32_S;
  case WebAssembly::ARGUMENT_I64: return WebAssembly::ARGUMENT_I64_S;
  case WebAssembly::ARGUMENT_v16i8: return WebAssembly::ARGUMENT_v16i8_S;
  case WebAssembly::ARGUMENT_v4f32: return WebAssembly::ARGUMENT_v4f32_S;
  case WebAssembly::ARGUMENT_v4i32: return WebAssembly::ARGUMENT_v4i32_S;
  case WebAssembly::ARGUMENT_v8i16: return WebAssembly::ARGUMENT_v8i16_S;
  case WebAssembly::ARGUMENT_v2f64: return WebAssembly::ARGUMENT_v2f64_S;
  case WebAssembly::ARGUMENT_v2i64: return WebAssembly::ARGUMENT_v2i64_S;
  case WebAssembly::ATOMIC_LOAD16_U_I32: return WebAssembly::ATOMIC_LOAD16_U_I32_S;
  case WebAssembly::ATOMIC_LOAD16_U_I64: return WebAssembly::ATOMIC_LOAD16_U_I64_S;
  case WebAssembly::ATOMIC_LOAD32_U_I64: return WebAssembly::ATOMIC_LOAD32_U_I64_S;
  case WebAssembly::ATOMIC_LOAD8_U_I32: return WebAssembly::ATOMIC_LOAD8_U_I32_S;
  case WebAssembly::ATOMIC_LOAD8_U_I64: return WebAssembly::ATOMIC_LOAD8_U_I64_S;
  case WebAssembly::ATOMIC_LOAD_I32: return WebAssembly::ATOMIC_LOAD_I32_S;
  case WebAssembly::ATOMIC_LOAD_I64: return WebAssembly::ATOMIC_LOAD_I64_S;
  case WebAssembly::ATOMIC_STORE16_I32: return WebAssembly::ATOMIC_STORE16_I32_S;
  case WebAssembly::ATOMIC_STORE16_I64: return WebAssembly::ATOMIC_STORE16_I64_S;
  case WebAssembly::ATOMIC_STORE32_I64: return WebAssembly::ATOMIC_STORE32_I64_S;
  case WebAssembly::ATOMIC_STORE8_I32: return WebAssembly::ATOMIC_STORE8_I32_S;
  case WebAssembly::ATOMIC_STORE8_I64: return WebAssembly::ATOMIC_STORE8_I64_S;
  case WebAssembly::ATOMIC_STORE_I32: return WebAssembly::ATOMIC_STORE_I32_S;
  case WebAssembly::ATOMIC_STORE_I64: return WebAssembly::ATOMIC_STORE_I64_S;
  case WebAssembly::BLOCK: return WebAssembly::BLOCK_S;
  case WebAssembly::BR: return WebAssembly::BR_S;
  case WebAssembly::BR_IF: return WebAssembly::BR_IF_S;
  case WebAssembly::BR_TABLE_I32: return WebAssembly::BR_TABLE_I32_S;
  case WebAssembly::BR_TABLE_I64: return WebAssembly::BR_TABLE_I64_S;
  case WebAssembly::BR_UNLESS: return WebAssembly::BR_UNLESS_S;
  case WebAssembly::CALL_EXCEPT_REF: return WebAssembly::CALL_EXCEPT_REF_S;
  case WebAssembly::CALL_F32: return WebAssembly::CALL_F32_S;
  case WebAssembly::CALL_F64: return WebAssembly::CALL_F64_S;
  case WebAssembly::CALL_I32: return WebAssembly::CALL_I32_S;
  case WebAssembly::CALL_I64: return WebAssembly::CALL_I64_S;
  case WebAssembly::CALL_INDIRECT_EXCEPT_REF: return WebAssembly::CALL_INDIRECT_EXCEPT_REF_S;
  case WebAssembly::CALL_INDIRECT_F32: return WebAssembly::CALL_INDIRECT_F32_S;
  case WebAssembly::CALL_INDIRECT_F64: return WebAssembly::CALL_INDIRECT_F64_S;
  case WebAssembly::CALL_INDIRECT_I32: return WebAssembly::CALL_INDIRECT_I32_S;
  case WebAssembly::CALL_INDIRECT_I64: return WebAssembly::CALL_INDIRECT_I64_S;
  case WebAssembly::CALL_INDIRECT_VOID: return WebAssembly::CALL_INDIRECT_VOID_S;
  case WebAssembly::CALL_INDIRECT_v16i8: return WebAssembly::CALL_INDIRECT_v16i8_S;
  case WebAssembly::CALL_INDIRECT_v4f32: return WebAssembly::CALL_INDIRECT_v4f32_S;
  case WebAssembly::CALL_INDIRECT_v4i32: return WebAssembly::CALL_INDIRECT_v4i32_S;
  case WebAssembly::CALL_INDIRECT_v8i16: return WebAssembly::CALL_INDIRECT_v8i16_S;
  case WebAssembly::CALL_VOID: return WebAssembly::CALL_VOID_S;
  case WebAssembly::CALL_v16i8: return WebAssembly::CALL_v16i8_S;
  case WebAssembly::CALL_v4f32: return WebAssembly::CALL_v4f32_S;
  case WebAssembly::CALL_v4i32: return WebAssembly::CALL_v4i32_S;
  case WebAssembly::CALL_v8i16: return WebAssembly::CALL_v8i16_S;
  case WebAssembly::CATCHRET: return WebAssembly::CATCHRET_S;
  case WebAssembly::CATCH_ALL: return WebAssembly::CATCH_ALL_S;
  case WebAssembly::CATCH_I32: return WebAssembly::CATCH_I32_S;
  case WebAssembly::CATCH_I64: return WebAssembly::CATCH_I64_S;
  case WebAssembly::CEIL_F32: return WebAssembly::CEIL_F32_S;
  case WebAssembly::CEIL_F64: return WebAssembly::CEIL_F64_S;
  case WebAssembly::CLEANUPRET: return WebAssembly::CLEANUPRET_S;
  case WebAssembly::CLZ_I32: return WebAssembly::CLZ_I32_S;
  case WebAssembly::CLZ_I64: return WebAssembly::CLZ_I64_S;
  case WebAssembly::CONST_F32: return WebAssembly::CONST_F32_S;
  case WebAssembly::CONST_F64: return WebAssembly::CONST_F64_S;
  case WebAssembly::CONST_I32: return WebAssembly::CONST_I32_S;
  case WebAssembly::CONST_I64: return WebAssembly::CONST_I64_S;
  case WebAssembly::COPYSIGN_F32: return WebAssembly::COPYSIGN_F32_S;
  case WebAssembly::COPYSIGN_F64: return WebAssembly::COPYSIGN_F64_S;
  case WebAssembly::COPY_EXCEPT_REF: return WebAssembly::COPY_EXCEPT_REF_S;
  case WebAssembly::COPY_F32: return WebAssembly::COPY_F32_S;
  case WebAssembly::COPY_F64: return WebAssembly::COPY_F64_S;
  case WebAssembly::COPY_I32: return WebAssembly::COPY_I32_S;
  case WebAssembly::COPY_I64: return WebAssembly::COPY_I64_S;
  case WebAssembly::COPY_V128: return WebAssembly::COPY_V128_S;
  case WebAssembly::CTZ_I32: return WebAssembly::CTZ_I32_S;
  case WebAssembly::CTZ_I64: return WebAssembly::CTZ_I64_S;
  case WebAssembly::CURRENT_MEMORY_I32: return WebAssembly::CURRENT_MEMORY_I32_S;
  case WebAssembly::DIV_F32: return WebAssembly::DIV_F32_S;
  case WebAssembly::DIV_F64: return WebAssembly::DIV_F64_S;
  case WebAssembly::DIV_S_I32: return WebAssembly::DIV_S_I32_S;
  case WebAssembly::DIV_S_I64: return WebAssembly::DIV_S_I64_S;
  case WebAssembly::DIV_U_I32: return WebAssembly::DIV_U_I32_S;
  case WebAssembly::DIV_U_I64: return WebAssembly::DIV_U_I64_S;
  case WebAssembly::DROP_EXCEPT_REF: return WebAssembly::DROP_EXCEPT_REF_S;
  case WebAssembly::DROP_F32: return WebAssembly::DROP_F32_S;
  case WebAssembly::DROP_F64: return WebAssembly::DROP_F64_S;
  case WebAssembly::DROP_I32: return WebAssembly::DROP_I32_S;
  case WebAssembly::DROP_I64: return WebAssembly::DROP_I64_S;
  case WebAssembly::DROP_V128: return WebAssembly::DROP_V128_S;
  case WebAssembly::END_BLOCK: return WebAssembly::END_BLOCK_S;
  case WebAssembly::END_FUNCTION: return WebAssembly::END_FUNCTION_S;
  case WebAssembly::END_LOOP: return WebAssembly::END_LOOP_S;
  case WebAssembly::END_TRY: return WebAssembly::END_TRY_S;
  case WebAssembly::EQZ_I32: return WebAssembly::EQZ_I32_S;
  case WebAssembly::EQZ_I64: return WebAssembly::EQZ_I64_S;
  case WebAssembly::EQ_F32: return WebAssembly::EQ_F32_S;
  case WebAssembly::EQ_F64: return WebAssembly::EQ_F64_S;
  case WebAssembly::EQ_I32: return WebAssembly::EQ_I32_S;
  case WebAssembly::EQ_I64: return WebAssembly::EQ_I64_S;
  case WebAssembly::F32_CONVERT_S_I32: return WebAssembly::F32_CONVERT_S_I32_S;
  case WebAssembly::F32_CONVERT_S_I64: return WebAssembly::F32_CONVERT_S_I64_S;
  case WebAssembly::F32_CONVERT_U_I32: return WebAssembly::F32_CONVERT_U_I32_S;
  case WebAssembly::F32_CONVERT_U_I64: return WebAssembly::F32_CONVERT_U_I64_S;
  case WebAssembly::F32_DEMOTE_F64: return WebAssembly::F32_DEMOTE_F64_S;
  case WebAssembly::F32_REINTERPRET_I32: return WebAssembly::F32_REINTERPRET_I32_S;
  case WebAssembly::F64_CONVERT_S_I32: return WebAssembly::F64_CONVERT_S_I32_S;
  case WebAssembly::F64_CONVERT_S_I64: return WebAssembly::F64_CONVERT_S_I64_S;
  case WebAssembly::F64_CONVERT_U_I32: return WebAssembly::F64_CONVERT_U_I32_S;
  case WebAssembly::F64_CONVERT_U_I64: return WebAssembly::F64_CONVERT_U_I64_S;
  case WebAssembly::F64_PROMOTE_F32: return WebAssembly::F64_PROMOTE_F32_S;
  case WebAssembly::F64_REINTERPRET_I64: return WebAssembly::F64_REINTERPRET_I64_S;
  case WebAssembly::FALLTHROUGH_RETURN_EXCEPT_REF: return WebAssembly::FALLTHROUGH_RETURN_EXCEPT_REF_S;
  case WebAssembly::FALLTHROUGH_RETURN_F32: return WebAssembly::FALLTHROUGH_RETURN_F32_S;
  case WebAssembly::FALLTHROUGH_RETURN_F64: return WebAssembly::FALLTHROUGH_RETURN_F64_S;
  case WebAssembly::FALLTHROUGH_RETURN_I32: return WebAssembly::FALLTHROUGH_RETURN_I32_S;
  case WebAssembly::FALLTHROUGH_RETURN_I64: return WebAssembly::FALLTHROUGH_RETURN_I64_S;
  case WebAssembly::FALLTHROUGH_RETURN_VOID: return WebAssembly::FALLTHROUGH_RETURN_VOID_S;
  case WebAssembly::FALLTHROUGH_RETURN_v16i8: return WebAssembly::FALLTHROUGH_RETURN_v16i8_S;
  case WebAssembly::FALLTHROUGH_RETURN_v4f32: return WebAssembly::FALLTHROUGH_RETURN_v4f32_S;
  case WebAssembly::FALLTHROUGH_RETURN_v4i32: return WebAssembly::FALLTHROUGH_RETURN_v4i32_S;
  case WebAssembly::FALLTHROUGH_RETURN_v8i16: return WebAssembly::FALLTHROUGH_RETURN_v8i16_S;
  case WebAssembly::FALLTHROUGH_RETURN_v2f64: return WebAssembly::FALLTHROUGH_RETURN_v2f64_S;
  case WebAssembly::FALLTHROUGH_RETURN_v2i64: return WebAssembly::FALLTHROUGH_RETURN_v2i64_S;
  case WebAssembly::FLOOR_F32: return WebAssembly::FLOOR_F32_S;
  case WebAssembly::FLOOR_F64: return WebAssembly::FLOOR_F64_S;
  case WebAssembly::FP_TO_SINT_I32_F32: return WebAssembly::FP_TO_SINT_I32_F32_S;
  case WebAssembly::FP_TO_SINT_I32_F64: return WebAssembly::FP_TO_SINT_I32_F64_S;
  case WebAssembly::FP_TO_SINT_I64_F32: return WebAssembly::FP_TO_SINT_I64_F32_S;
  case WebAssembly::FP_TO_SINT_I64_F64: return WebAssembly::FP_TO_SINT_I64_F64_S;
  case WebAssembly::FP_TO_UINT_I32_F32: return WebAssembly::FP_TO_UINT_I32_F32_S;
  case WebAssembly::FP_TO_UINT_I32_F64: return WebAssembly::FP_TO_UINT_I32_F64_S;
  case WebAssembly::FP_TO_UINT_I64_F32: return WebAssembly::FP_TO_UINT_I64_F32_S;
  case WebAssembly::FP_TO_UINT_I64_F64: return WebAssembly::FP_TO_UINT_I64_F64_S;
  case WebAssembly::GET_GLOBAL_EXCEPT_REF: return WebAssembly::GET_GLOBAL_EXCEPT_REF_S;
  case WebAssembly::GET_GLOBAL_F32: return WebAssembly::GET_GLOBAL_F32_S;
  case WebAssembly::GET_GLOBAL_F64: return WebAssembly::GET_GLOBAL_F64_S;
  case WebAssembly::GET_GLOBAL_I32: return WebAssembly::GET_GLOBAL_I32_S;
  case WebAssembly::GET_GLOBAL_I64: return WebAssembly::GET_GLOBAL_I64_S;
  case WebAssembly::GET_GLOBAL_V128: return WebAssembly::GET_GLOBAL_V128_S;
  case WebAssembly::GET_LOCAL_EXCEPT_REF: return WebAssembly::GET_LOCAL_EXCEPT_REF_S;
  case WebAssembly::GET_LOCAL_F32: return WebAssembly::GET_LOCAL_F32_S;
  case WebAssembly::GET_LOCAL_F64: return WebAssembly::GET_LOCAL_F64_S;
  case WebAssembly::GET_LOCAL_I32: return WebAssembly::GET_LOCAL_I32_S;
  case WebAssembly::GET_LOCAL_I64: return WebAssembly::GET_LOCAL_I64_S;
  case WebAssembly::GET_LOCAL_V128: return WebAssembly::GET_LOCAL_V128_S;
  case WebAssembly::GE_F32: return WebAssembly::GE_F32_S;
  case WebAssembly::GE_F64: return WebAssembly::GE_F64_S;
  case WebAssembly::GE_S_I32: return WebAssembly::GE_S_I32_S;
  case WebAssembly::GE_S_I64: return WebAssembly::GE_S_I64_S;
  case WebAssembly::GE_U_I32: return WebAssembly::GE_U_I32_S;
  case WebAssembly::GE_U_I64: return WebAssembly::GE_U_I64_S;
  case WebAssembly::GROW_MEMORY_I32: return WebAssembly::GROW_MEMORY_I32_S;
  case WebAssembly::GT_F32: return WebAssembly::GT_F32_S;
  case WebAssembly::GT_F64: return WebAssembly::GT_F64_S;
  case WebAssembly::GT_S_I32: return WebAssembly::GT_S_I32_S;
  case WebAssembly::GT_S_I64: return WebAssembly::GT_S_I64_S;
  case WebAssembly::GT_U_I32: return WebAssembly::GT_U_I32_S;
  case WebAssembly::GT_U_I64: return WebAssembly::GT_U_I64_S;
  case WebAssembly::I32_EXTEND16_S_I32: return WebAssembly::I32_EXTEND16_S_I32_S;
  case WebAssembly::I32_EXTEND8_S_I32: return WebAssembly::I32_EXTEND8_S_I32_S;
  case WebAssembly::I32_REINTERPRET_F32: return WebAssembly::I32_REINTERPRET_F32_S;
  case WebAssembly::I32_TRUNC_S_F32: return WebAssembly::I32_TRUNC_S_F32_S;
  case WebAssembly::I32_TRUNC_S_F64: return WebAssembly::I32_TRUNC_S_F64_S;
  case WebAssembly::I32_TRUNC_S_SAT_F32: return WebAssembly::I32_TRUNC_S_SAT_F32_S;
  case WebAssembly::I32_TRUNC_S_SAT_F64: return WebAssembly::I32_TRUNC_S_SAT_F64_S;
  case WebAssembly::I32_TRUNC_U_F32: return WebAssembly::I32_TRUNC_U_F32_S;
  case WebAssembly::I32_TRUNC_U_F64: return WebAssembly::I32_TRUNC_U_F64_S;
  case WebAssembly::I32_TRUNC_U_SAT_F32: return WebAssembly::I32_TRUNC_U_SAT_F32_S;
  case WebAssembly::I32_TRUNC_U_SAT_F64: return WebAssembly::I32_TRUNC_U_SAT_F64_S;
  case WebAssembly::I32_WRAP_I64: return WebAssembly::I32_WRAP_I64_S;
  case WebAssembly::I64_EXTEND16_S_I64: return WebAssembly::I64_EXTEND16_S_I64_S;
  case WebAssembly::I64_EXTEND32_S_I64: return WebAssembly::I64_EXTEND32_S_I64_S;
  case WebAssembly::I64_EXTEND8_S_I64: return WebAssembly::I64_EXTEND8_S_I64_S;
  case WebAssembly::I64_EXTEND_S_I32: return WebAssembly::I64_EXTEND_S_I32_S;
  case WebAssembly::I64_EXTEND_U_I32: return WebAssembly::I64_EXTEND_U_I32_S;
  case WebAssembly::I64_REINTERPRET_F64: return WebAssembly::I64_REINTERPRET_F64_S;
  case WebAssembly::I64_TRUNC_S_F32: return WebAssembly::I64_TRUNC_S_F32_S;
  case WebAssembly::I64_TRUNC_S_F64: return WebAssembly::I64_TRUNC_S_F64_S;
  case WebAssembly::I64_TRUNC_S_SAT_F32: return WebAssembly::I64_TRUNC_S_SAT_F32_S;
  case WebAssembly::I64_TRUNC_S_SAT_F64: return WebAssembly::I64_TRUNC_S_SAT_F64_S;
  case WebAssembly::I64_TRUNC_U_F32: return WebAssembly::I64_TRUNC_U_F32_S;
  case WebAssembly::I64_TRUNC_U_F64: return WebAssembly::I64_TRUNC_U_F64_S;
  case WebAssembly::I64_TRUNC_U_SAT_F32: return WebAssembly::I64_TRUNC_U_SAT_F32_S;
  case WebAssembly::I64_TRUNC_U_SAT_F64: return WebAssembly::I64_TRUNC_U_SAT_F64_S;
  case WebAssembly::LE_F32: return WebAssembly::LE_F32_S;
  case WebAssembly::LE_F64: return WebAssembly::LE_F64_S;
  case WebAssembly::LE_S_I32: return WebAssembly::LE_S_I32_S;
  case WebAssembly::LE_S_I64: return WebAssembly::LE_S_I64_S;
  case WebAssembly::LE_U_I32: return WebAssembly::LE_U_I32_S;
  case WebAssembly::LE_U_I64: return WebAssembly::LE_U_I64_S;
  case WebAssembly::LOAD16_S_I32: return WebAssembly::LOAD16_S_I32_S;
  case WebAssembly::LOAD16_S_I64: return WebAssembly::LOAD16_S_I64_S;
  case WebAssembly::LOAD16_U_I32: return WebAssembly::LOAD16_U_I32_S;
  case WebAssembly::LOAD16_U_I64: return WebAssembly::LOAD16_U_I64_S;
  case WebAssembly::LOAD32_S_I64: return WebAssembly::LOAD32_S_I64_S;
  case WebAssembly::LOAD32_U_I64: return WebAssembly::LOAD32_U_I64_S;
  case WebAssembly::LOAD8_S_I32: return WebAssembly::LOAD8_S_I32_S;
  case WebAssembly::LOAD8_S_I64: return WebAssembly::LOAD8_S_I64_S;
  case WebAssembly::LOAD8_U_I32: return WebAssembly::LOAD8_U_I32_S;
  case WebAssembly::LOAD8_U_I64: return WebAssembly::LOAD8_U_I64_S;
  case WebAssembly::LOAD_F32: return WebAssembly::LOAD_F32_S;
  case WebAssembly::LOAD_F64: return WebAssembly::LOAD_F64_S;
  case WebAssembly::LOAD_I32: return WebAssembly::LOAD_I32_S;
  case WebAssembly::LOAD_I64: return WebAssembly::LOAD_I64_S;
  case WebAssembly::LOOP: return WebAssembly::LOOP_S;
  case WebAssembly::LT_F32: return WebAssembly::LT_F32_S;
  case WebAssembly::LT_F64: return WebAssembly::LT_F64_S;
  case WebAssembly::LT_S_I32: return WebAssembly::LT_S_I32_S;
  case WebAssembly::LT_S_I64: return WebAssembly::LT_S_I64_S;
  case WebAssembly::LT_U_I32: return WebAssembly::LT_U_I32_S;
  case WebAssembly::LT_U_I64: return WebAssembly::LT_U_I64_S;
  case WebAssembly::MAX_F32: return WebAssembly::MAX_F32_S;
  case WebAssembly::MAX_F64: return WebAssembly::MAX_F64_S;
  case WebAssembly::MEMORY_GROW_I32: return WebAssembly::MEMORY_GROW_I32_S;
  case WebAssembly::MEMORY_SIZE_I32: return WebAssembly::MEMORY_SIZE_I32_S;
  case WebAssembly::MEM_GROW_I32: return WebAssembly::MEM_GROW_I32_S;
  case WebAssembly::MEM_SIZE_I32: return WebAssembly::MEM_SIZE_I32_S;
  case WebAssembly::MIN_F32: return WebAssembly::MIN_F32_S;
  case WebAssembly::MIN_F64: return WebAssembly::MIN_F64_S;
  case WebAssembly::MUL_F32: return WebAssembly::MUL_F32_S;
  case WebAssembly::MUL_F64: return WebAssembly::MUL_F64_S;
  case WebAssembly::MUL_I32: return WebAssembly::MUL_I32_S;
  case WebAssembly::MUL_I64: return WebAssembly::MUL_I64_S;
  case WebAssembly::MUL_v16i8: return WebAssembly::MUL_v16i8_S;
  case WebAssembly::MUL_v2f64: return WebAssembly::MUL_v2f64_S;
  case WebAssembly::MUL_v4f32: return WebAssembly::MUL_v4f32_S;
  case WebAssembly::MUL_v4i32: return WebAssembly::MUL_v4i32_S;
  case WebAssembly::MUL_v8i16: return WebAssembly::MUL_v8i16_S;
  case WebAssembly::NEAREST_F32: return WebAssembly::NEAREST_F32_S;
  case WebAssembly::NEAREST_F64: return WebAssembly::NEAREST_F64_S;
  case WebAssembly::NEG_F32: return WebAssembly::NEG_F32_S;
  case WebAssembly::NEG_F64: return WebAssembly::NEG_F64_S;
  case WebAssembly::NE_F32: return WebAssembly::NE_F32_S;
  case WebAssembly::NE_F64: return WebAssembly::NE_F64_S;
  case WebAssembly::NE_I32: return WebAssembly::NE_I32_S;
  case WebAssembly::NE_I64: return WebAssembly::NE_I64_S;
  case WebAssembly::NOP: return WebAssembly::NOP_S;
  case WebAssembly::OR_I32: return WebAssembly::OR_I32_S;
  case WebAssembly::OR_I64: return WebAssembly::OR_I64_S;
  case WebAssembly::PCALL_INDIRECT_EXCEPT_REF: return WebAssembly::PCALL_INDIRECT_EXCEPT_REF_S;
  case WebAssembly::PCALL_INDIRECT_F32: return WebAssembly::PCALL_INDIRECT_F32_S;
  case WebAssembly::PCALL_INDIRECT_F64: return WebAssembly::PCALL_INDIRECT_F64_S;
  case WebAssembly::PCALL_INDIRECT_I32: return WebAssembly::PCALL_INDIRECT_I32_S;
  case WebAssembly::PCALL_INDIRECT_I64: return WebAssembly::PCALL_INDIRECT_I64_S;
  case WebAssembly::PCALL_INDIRECT_VOID: return WebAssembly::PCALL_INDIRECT_VOID_S;
  case WebAssembly::PCALL_INDIRECT_v16i8: return WebAssembly::PCALL_INDIRECT_v16i8_S;
  case WebAssembly::PCALL_INDIRECT_v4f32: return WebAssembly::PCALL_INDIRECT_v4f32_S;
  case WebAssembly::PCALL_INDIRECT_v4i32: return WebAssembly::PCALL_INDIRECT_v4i32_S;
  case WebAssembly::PCALL_INDIRECT_v8i16: return WebAssembly::PCALL_INDIRECT_v8i16_S;
  case WebAssembly::POPCNT_I32: return WebAssembly::POPCNT_I32_S;
  case WebAssembly::POPCNT_I64: return WebAssembly::POPCNT_I64_S;
  case WebAssembly::REM_S_I32: return WebAssembly::REM_S_I32_S;
  case WebAssembly::REM_S_I64: return WebAssembly::REM_S_I64_S;
  case WebAssembly::REM_U_I32: return WebAssembly::REM_U_I32_S;
  case WebAssembly::REM_U_I64: return WebAssembly::REM_U_I64_S;
  case WebAssembly::RETHROW: return WebAssembly::RETHROW_S;
  case WebAssembly::RETHROW_TO_CALLER: return WebAssembly::RETHROW_TO_CALLER_S;
  case WebAssembly::RETURN_EXCEPT_REF: return WebAssembly::RETURN_EXCEPT_REF_S;
  case WebAssembly::RETURN_F32: return WebAssembly::RETURN_F32_S;
  case WebAssembly::RETURN_F64: return WebAssembly::RETURN_F64_S;
  case WebAssembly::RETURN_I32: return WebAssembly::RETURN_I32_S;
  case WebAssembly::RETURN_I64: return WebAssembly::RETURN_I64_S;
  case WebAssembly::RETURN_VOID: return WebAssembly::RETURN_VOID_S;
  case WebAssembly::RETURN_v16i8: return WebAssembly::RETURN_v16i8_S;
  case WebAssembly::RETURN_v4f32: return WebAssembly::RETURN_v4f32_S;
  case WebAssembly::RETURN_v4i32: return WebAssembly::RETURN_v4i32_S;
  case WebAssembly::RETURN_v8i16: return WebAssembly::RETURN_v8i16_S;
  case WebAssembly::ROTL_I32: return WebAssembly::ROTL_I32_S;
  case WebAssembly::ROTL_I64: return WebAssembly::ROTL_I64_S;
  case WebAssembly::ROTR_I32: return WebAssembly::ROTR_I32_S;
  case WebAssembly::ROTR_I64: return WebAssembly::ROTR_I64_S;
  case WebAssembly::SELECT_EXCEPT_REF: return WebAssembly::SELECT_EXCEPT_REF_S;
  case WebAssembly::SELECT_F32: return WebAssembly::SELECT_F32_S;
  case WebAssembly::SELECT_F64: return WebAssembly::SELECT_F64_S;
  case WebAssembly::SELECT_I32: return WebAssembly::SELECT_I32_S;
  case WebAssembly::SELECT_I64: return WebAssembly::SELECT_I64_S;
  case WebAssembly::SET_GLOBAL_EXCEPT_REF: return WebAssembly::SET_GLOBAL_EXCEPT_REF_S;
  case WebAssembly::SET_GLOBAL_F32: return WebAssembly::SET_GLOBAL_F32_S;
  case WebAssembly::SET_GLOBAL_F64: return WebAssembly::SET_GLOBAL_F64_S;
  case WebAssembly::SET_GLOBAL_I32: return WebAssembly::SET_GLOBAL_I32_S;
  case WebAssembly::SET_GLOBAL_I64: return WebAssembly::SET_GLOBAL_I64_S;
  case WebAssembly::SET_GLOBAL_V128: return WebAssembly::SET_GLOBAL_V128_S;
  case WebAssembly::SET_LOCAL_EXCEPT_REF: return WebAssembly::SET_LOCAL_EXCEPT_REF_S;
  case WebAssembly::SET_LOCAL_F32: return WebAssembly::SET_LOCAL_F32_S;
  case WebAssembly::SET_LOCAL_F64: return WebAssembly::SET_LOCAL_F64_S;
  case WebAssembly::SET_LOCAL_I32: return WebAssembly::SET_LOCAL_I32_S;
  case WebAssembly::SET_LOCAL_I64: return WebAssembly::SET_LOCAL_I64_S;
  case WebAssembly::SET_LOCAL_V128: return WebAssembly::SET_LOCAL_V128_S;
  case WebAssembly::SHL_I32: return WebAssembly::SHL_I32_S;
  case WebAssembly::SHL_I64: return WebAssembly::SHL_I64_S;
  case WebAssembly::SHR_S_I32: return WebAssembly::SHR_S_I32_S;
  case WebAssembly::SHR_S_I64: return WebAssembly::SHR_S_I64_S;
  case WebAssembly::SHR_U_I32: return WebAssembly::SHR_U_I32_S;
  case WebAssembly::SHR_U_I64: return WebAssembly::SHR_U_I64_S;
  case WebAssembly::SQRT_F32: return WebAssembly::SQRT_F32_S;
  case WebAssembly::SQRT_F64: return WebAssembly::SQRT_F64_S;
  case WebAssembly::STORE16_I32: return WebAssembly::STORE16_I32_S;
  case WebAssembly::STORE16_I64: return WebAssembly::STORE16_I64_S;
  case WebAssembly::STORE32_I64: return WebAssembly::STORE32_I64_S;
  case WebAssembly::STORE8_I32: return WebAssembly::STORE8_I32_S;
  case WebAssembly::STORE8_I64: return WebAssembly::STORE8_I64_S;
  case WebAssembly::STORE_F32: return WebAssembly::STORE_F32_S;
  case WebAssembly::STORE_F64: return WebAssembly::STORE_F64_S;
  case WebAssembly::STORE_I32: return WebAssembly::STORE_I32_S;
  case WebAssembly::STORE_I64: return WebAssembly::STORE_I64_S;
  case WebAssembly::SUB_F32: return WebAssembly::SUB_F32_S;
  case WebAssembly::SUB_F64: return WebAssembly::SUB_F64_S;
  case WebAssembly::SUB_I32: return WebAssembly::SUB_I32_S;
  case WebAssembly::SUB_I64: return WebAssembly::SUB_I64_S;
  case WebAssembly::SUB_v16i8: return WebAssembly::SUB_v16i8_S;
  case WebAssembly::SUB_v2f64: return WebAssembly::SUB_v2f64_S;
  case WebAssembly::SUB_v2i64: return WebAssembly::SUB_v2i64_S;
  case WebAssembly::SUB_v4f32: return WebAssembly::SUB_v4f32_S;
  case WebAssembly::SUB_v4i32: return WebAssembly::SUB_v4i32_S;
  case WebAssembly::SUB_v8i16: return WebAssembly::SUB_v8i16_S;
  case WebAssembly::TEE_EXCEPT_REF: return WebAssembly::TEE_EXCEPT_REF_S;
  case WebAssembly::TEE_F32: return WebAssembly::TEE_F32_S;
  case WebAssembly::TEE_F64: return WebAssembly::TEE_F64_S;
  case WebAssembly::TEE_I32: return WebAssembly::TEE_I32_S;
  case WebAssembly::TEE_I64: return WebAssembly::TEE_I64_S;
  case WebAssembly::TEE_LOCAL_EXCEPT_REF: return WebAssembly::TEE_LOCAL_EXCEPT_REF_S;
  case WebAssembly::TEE_LOCAL_F32: return WebAssembly::TEE_LOCAL_F32_S;
  case WebAssembly::TEE_LOCAL_F64: return WebAssembly::TEE_LOCAL_F64_S;
  case WebAssembly::TEE_LOCAL_I32: return WebAssembly::TEE_LOCAL_I32_S;
  case WebAssembly::TEE_LOCAL_I64: return WebAssembly::TEE_LOCAL_I64_S;
  case WebAssembly::TEE_LOCAL_V128: return WebAssembly::TEE_LOCAL_V128_S;
  case WebAssembly::TEE_V128: return WebAssembly::TEE_V128_S;
  case WebAssembly::THROW_I32: return WebAssembly::THROW_I32_S;
  case WebAssembly::THROW_I64: return WebAssembly::THROW_I64_S;
  case WebAssembly::TRUNC_F32: return WebAssembly::TRUNC_F32_S;
  case WebAssembly::TRUNC_F64: return WebAssembly::TRUNC_F64_S;
  case WebAssembly::TRY: return WebAssembly::TRY_S;
  case WebAssembly::UNREACHABLE: return WebAssembly::UNREACHABLE_S;
  case WebAssembly::XOR_I32: return WebAssembly::XOR_I32_S;
  case WebAssembly::XOR_I64: return WebAssembly::XOR_I64_S;
  }
}
