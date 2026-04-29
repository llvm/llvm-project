//===-- WebAssemblyCallLowering.cpp - Call lowering for GlobalISel -*- C++ -*-//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the lowering of LLVM calls to machine code calls for
/// GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyCallLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "Utils/WasmAddressSpaces.h"
#include "WebAssemblyISelLowering.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyUtilities.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/CallLowering.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/LowLevelTypeUtils.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"

#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/MC/MCSymbolWasm.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "wasm-call-lowering"

using namespace llvm;

// Test whether the given calling convention is supported.
static bool callingConvSupported(CallingConv::ID CallConv) {
  // We currently support the language-independent target-independent
  // conventions. We don't yet have a way to annotate calls with properties like
  // "cold", and we don't have any call-clobbered registers, so these are mostly
  // all handled the same.
  return CallConv == CallingConv::C || CallConv == CallingConv::Fast ||
         CallConv == CallingConv::Cold ||
         CallConv == CallingConv::PreserveMost ||
         CallConv == CallingConv::PreserveAll ||
         CallConv == CallingConv::CXX_FAST_TLS ||
         CallConv == CallingConv::WASM_EmscriptenInvoke ||
         CallConv == CallingConv::Swift;
}

static unsigned extendOpFromFlags(ISD::ArgFlagsTy Flags) {
  if (Flags.isSExt())
    return TargetOpcode::G_SEXT;
  if (Flags.isZExt())
    return TargetOpcode::G_ZEXT;
  return TargetOpcode::G_ANYEXT;
}

static LLT getLLTForWasmMVT(MVT Ty, const DataLayout &DL) {
  if (Ty == MVT::externref) {
    return LLT::pointer(
        WebAssembly::WasmAddressSpace::WASM_ADDRESS_SPACE_EXTERNREF,
        DL.getPointerSizeInBits(
            WebAssembly::WasmAddressSpace::WASM_ADDRESS_SPACE_EXTERNREF));
  }

  if (Ty == MVT::funcref) {
    return LLT::pointer(
        WebAssembly::WasmAddressSpace::WASM_ADDRESS_SPACE_FUNCREF,
        DL.getPointerSizeInBits(
            WebAssembly::WasmAddressSpace::WASM_ADDRESS_SPACE_FUNCREF));
  }

  return llvm::getLLTForMVT(Ty);
}

WebAssemblyCallLowering::WebAssemblyCallLowering(
    const WebAssemblyTargetLowering &TLI)
    : CallLowering(&TLI) {}

bool WebAssemblyCallLowering::canLowerReturn(MachineFunction &MF,
                                             CallingConv::ID CallConv,
                                             SmallVectorImpl<BaseArgInfo> &Outs,
                                             bool IsVarArg) const {
  return WebAssembly::canLowerReturn(Outs.size(),
                                     &MF.getSubtarget<WebAssemblySubtarget>());
}

bool WebAssemblyCallLowering::lowerReturn(MachineIRBuilder &MIRBuilder,
                                          const Value *Val,
                                          ArrayRef<Register> VRegs,
                                          FunctionLoweringInfo &FLI,
                                          Register SwiftErrorVReg) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const Function &F = MF.getFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const WebAssemblyTargetLowering &TLI = *getTLI<WebAssemblyTargetLowering>();
  const DataLayout &DL = F.getDataLayout();

  MachineInstrBuilder MIB = MIRBuilder.buildInstrNoInsert(WebAssembly::RETURN);

  assert(((Val && !VRegs.empty()) || (!Val && VRegs.empty())) &&
         "Return value without a vreg or vice versa");

  if (Val) {
    LLVMContext &Ctx = Val->getType()->getContext();

    if (!FLI.CanLowerReturn) {
      insertSRetStores(MIRBuilder, Val->getType(), VRegs, FLI.DemoteRegister);
    } else {
      SmallVector<EVT, 4> SplitEVTs;
      ComputeValueVTs(TLI, DL, Val->getType(), SplitEVTs);
      assert(VRegs.size() == SplitEVTs.size() &&
             "For each split Type there should be exactly one VReg.");

      SmallVector<ArgInfo, 8> SplitRets;
      CallingConv::ID CallConv = F.getCallingConv();

      unsigned RetIdx = 0;
      for (EVT SplitEVT : SplitEVTs) {
        Register CurVReg = VRegs[RetIdx];
        ArgInfo CurRetInfo = ArgInfo{CurVReg, SplitEVT.getTypeForEVT(Ctx), 0};
        setArgFlags(CurRetInfo, AttributeList::ReturnIndex, DL, F);

        splitToValueTypes(CurRetInfo, SplitRets, DL, CallConv);
        ++RetIdx;
      }

      for (ArgInfo &Ret : SplitRets) {
        const EVT OrigVT = TLI.getValueType(DL, Ret.Ty);
        const MVT NewVT =
            TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
        const LLT OrigLLT =
            getLLTForType(*OrigVT.getTypeForEVT(F.getContext()), DL);
        const LLT NewLLT = getLLTForWasmMVT(NewVT, DL);

        const TargetRegisterClass &NewRegClass = *TLI.getRegClassFor(NewVT);

        // If we need to split the type over multiple regs, check it's a
        // scenario we currently support.
        const unsigned NumParts =
            TLI.getNumRegistersForCallingConv(Ctx, CallConv, OrigVT);

        const ISD::ArgFlagsTy OrigFlags = Ret.Flags[0];
        Ret.Flags.clear();

        for (unsigned Part = 0; Part < NumParts; ++Part) {
          ISD::ArgFlagsTy Flags = OrigFlags;

          if (Part == 0) {
            Flags.setSplit();
          } else {
            Flags.setOrigAlign(Align(1));
            if (Part == NumParts - 1)
              Flags.setSplitEnd();
          }

          Ret.Flags.push_back(Flags);
        }

        Ret.OrigRegs.assign(Ret.Regs.begin(), Ret.Regs.end());
        if (NumParts != 1 || OrigLLT != NewLLT) {
          // If we can't directly assign the register, we need one or more
          // intermediate values.
          Ret.Regs.resize(NumParts);

          // For each split register, create and assign a vreg that will store
          // the incoming component of the larger value. These will later be
          // merged to form the final vreg.
          for (unsigned Part = 0; Part < NumParts; ++Part) {
            Register NewReg = MRI.createVirtualRegister(&NewRegClass);
            MRI.setType(NewReg, NewLLT);
            MIB.addUse(NewReg);
            Ret.Regs[Part] = NewReg;
          }

          buildCopyToRegs(MIRBuilder, Ret.Regs, Ret.OrigRegs[0], OrigLLT,
                          NewLLT, extendOpFromFlags(Ret.Flags[0]));
        } else {
          MIB.addUse(Ret.Regs[0]);
        }
      }
    }
  }

  if (SwiftErrorVReg) {
    reportFatalInternalError(
        "Wasm does not `supportSwiftError`, yet SwiftErrorVReg is "
        "improperly valid.");
  }

  MIRBuilder.insertInstr(MIB);
  return true;
}

static unsigned getWasmArgumentOpcode(MVT ArgType) {
  switch (ArgType.SimpleTy) {
  case MVT::i32:
    return WebAssembly::ARGUMENT_i32;
  case MVT::i64:
    return WebAssembly::ARGUMENT_i64;
  case MVT::f32:
    return WebAssembly::ARGUMENT_f32;
  case MVT::f64:
    return WebAssembly::ARGUMENT_f64;

  case MVT::funcref:
    return WebAssembly::ARGUMENT_funcref;
  case MVT::externref:
    return WebAssembly::ARGUMENT_externref;
  case MVT::exnref:
    return WebAssembly::ARGUMENT_exnref;

  case MVT::v16i8:
    return WebAssembly::ARGUMENT_v16i8;
  case MVT::v8i16:
    return WebAssembly::ARGUMENT_v8i16;
  case MVT::v4i32:
    return WebAssembly::ARGUMENT_v4i32;
  case MVT::v2i64:
    return WebAssembly::ARGUMENT_v2i64;
  case MVT::v8f16:
    return WebAssembly::ARGUMENT_v8f16;
  case MVT::v4f32:
    return WebAssembly::ARGUMENT_v4f32;
  case MVT::v2f64:
    return WebAssembly::ARGUMENT_v2f64;
  default:
    break;
  }
  llvm_unreachable("Found unexpected type for Wasm argument");
}

static Register buildWasmArgument(unsigned Idx, MVT ArgVT, LLT ArgLLT,
                                  MachineIRBuilder &MIRBuilder,
                                  Register Def = Register()) {
  unsigned Op = getWasmArgumentOpcode(ArgVT);

  const TargetInstrInfo &TII = MIRBuilder.getTII();
  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterClass &RegClass = *TII.getRegClass(TII.get(Op), 0);

  Register NewReg;

  if (Def.isValid()) {
    assert(MRI.getRegClassOrRegBank(Def).isNull() &&
           "Def already has reg bank or reg class?");
    MRI.setRegClass(Def, &RegClass);

    NewReg = Def;
  } else {
    NewReg = MRI.createVirtualRegister(&RegClass);
    MRI.setType(NewReg, ArgLLT);
  }

  MIRBuilder.buildInstr(Op).addDef(NewReg).addImm(Idx);

  return NewReg;
}

bool WebAssemblyCallLowering::lowerFormalArguments(
    MachineIRBuilder &MIRBuilder, const Function &F,
    ArrayRef<ArrayRef<Register>> VRegs, FunctionLoweringInfo &FLI) const {
  MachineFunction &MF = MIRBuilder.getMF();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  WebAssemblyFunctionInfo *MFI = MF.getInfo<WebAssemblyFunctionInfo>();
  const DataLayout &DL = F.getDataLayout();
  const WebAssemblyTargetLowering &TLI = *getTLI<WebAssemblyTargetLowering>();

  LLVMContext &Ctx = MIRBuilder.getContext();
  const CallingConv::ID CallConv = F.getCallingConv();

  if (!callingConvSupported(CallConv))
    return false;

  MF.getRegInfo().addLiveIn(WebAssembly::ARGUMENTS);
  MF.front().addLiveIn(WebAssembly::ARGUMENTS);

  SmallVector<ArgInfo, 8> SplitArgs;

  if (!FLI.CanLowerReturn)
    insertSRetIncomingArgument(F, SplitArgs, FLI.DemoteRegister, MRI, DL);

  unsigned ArgIdx = 0;
  bool HasSwiftErrorArg = false;
  bool HasSwiftSelfArg = false;
  for (const Argument &Arg : F.args()) {
    ArgInfo OrigArg{VRegs[ArgIdx], Arg.getType(), ArgIdx};
    setArgFlags(OrigArg, ArgIdx + AttributeList::FirstArgIndex, DL, F);

    HasSwiftSelfArg |= Arg.hasSwiftSelfAttr();
    HasSwiftErrorArg |= Arg.hasSwiftErrorAttr();
    if (Arg.hasInAllocaAttr())
      return false;
    if (Arg.hasNestAttr())
      return false;

    splitToValueTypes(OrigArg, SplitArgs, DL, F.getCallingConv());
    ++ArgIdx;
  }

  unsigned FinalArgIdx = 0;
  for (ArgInfo &Arg : SplitArgs) {
    const EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
    const MVT NewVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
    const LLT OrigLLT =
        getLLTForType(*OrigVT.getTypeForEVT(F.getContext()), DL);
    const LLT NewLLT = getLLTForWasmMVT(NewVT, DL);

    // If we need to split the type over multiple regs, check it's a scenario
    // we currently support.
    const unsigned NumParts =
        TLI.getNumRegistersForCallingConv(Ctx, CallConv, OrigVT);

    const ISD::ArgFlagsTy OrigFlags = Arg.Flags[0];
    Arg.Flags.clear();

    for (unsigned Part = 0; Part < NumParts; ++Part) {
      ISD::ArgFlagsTy Flags = OrigFlags;
      if (Part == 0) {
        Flags.setSplit();
      } else {
        Flags.setOrigAlign(Align(1));
        if (Part == NumParts - 1)
          Flags.setSplitEnd();
      }

      Arg.Flags.push_back(Flags);
    }

    Arg.OrigRegs.assign(Arg.Regs.begin(), Arg.Regs.end());
    if (NumParts != 1 || OrigLLT != NewLLT) {
      // If we can't directly assign the register, we need one or more
      // intermediate values.
      Arg.Regs.resize(NumParts);

      // For each split register, create and assign a vreg that will store
      // the incoming component of the larger value. These will later be
      // merged to form the final vreg.
      for (unsigned Part = 0; Part < NumParts; ++Part) {
        Arg.Regs[Part] =
            buildWasmArgument(FinalArgIdx++, NewVT, NewLLT, MIRBuilder);
        MFI->addParam(NewVT);
      }

      buildCopyFromRegs(MIRBuilder, Arg.OrigRegs, Arg.Regs, OrigLLT, NewLLT,
                        Arg.Flags[0]);
    } else {
      buildWasmArgument(FinalArgIdx++, NewVT, NewLLT, MIRBuilder, Arg.Regs[0]);
      MFI->addParam(NewVT);
    }
  }

  // For swiftcc, emit additional swiftself and swifterror arguments
  // if there aren't. These additional arguments are also added for callee
  // signature They are necessary to match callee and caller signature for
  // indirect call.
  if (CallConv == CallingConv::Swift) {
    const MVT PtrVT = TLI.getPointerTy(DL);

    if (!HasSwiftSelfArg)
      MFI->addParam(PtrVT);
    if (!HasSwiftErrorArg)
      MFI->addParam(PtrVT);
  }

  // Varargs are copied into a buffer allocated by the caller, and a pointer to
  // the buffer is passed as an argument.
  if (F.isVarArg()) {
    const MVT PtrVT = TLI.getPointerTy(DL, 0);
    const LLT PtrLLT = LLT::pointer(0, DL.getPointerSizeInBits(0));

    Register VarargVreg =
        buildWasmArgument(FinalArgIdx++, PtrVT, PtrLLT, MIRBuilder);
    MFI->setVarargBufferVreg(VarargVreg);

    MFI->addParam(PtrVT);
    ++FinalArgIdx;
  }

  // Record the number and types of arguments and results.
  SmallVector<MVT, 4> Params;
  SmallVector<MVT, 4> Results;
  computeSignatureVTs(MF.getFunction().getFunctionType(), &MF.getFunction(),
                      MF.getFunction(), MF.getTarget(), Params, Results);
  for (MVT VT : Results)
    MFI->addResult(VT);

  // TODO: Use signatures in WebAssemblyMachineFunctionInfo too and unify
  // the param logic here with ComputeSignatureVTs
  assert(MFI->getParams().size() == Params.size() &&
         std::equal(MFI->getParams().begin(), MFI->getParams().end(),
                    Params.begin()));
  return true;
}

bool WebAssemblyCallLowering::lowerCall(MachineIRBuilder &MIRBuilder,
                                        CallLoweringInfo &Info) const {
  MachineFunction &MF = MIRBuilder.getMF();
  const DataLayout &DL = MIRBuilder.getDataLayout();
  LLVMContext &Ctx = MIRBuilder.getContext();
  const WebAssemblyTargetLowering &TLI = *getTLI<WebAssemblyTargetLowering>();
  MachineRegisterInfo &MRI = *MIRBuilder.getMRI();
  const WebAssemblySubtarget &Subtarget =
      MF.getSubtarget<WebAssemblySubtarget>();
  auto &TRI = *Subtarget.getRegisterInfo();
  auto &TII = *Subtarget.getInstrInfo();
  auto &RBI = *Subtarget.getRegBankInfo();
  const Function &F = MF.getFunction();

  CallingConv::ID CallConv = Info.CallConv;
  if (!callingConvSupported(CallConv))
    return false;

  // TODO: investigate "PatchPoint"
  /*
  if (Info.IsPatchPoint) {
    fail(MIRBuilder, "WebAssembly doesn't support patch point yet");
    return false;
    }
    */

  if (Info.IsTailCall) {
    Info.LoweredTailCall = true;
    auto NoTail = [&]() {
      if (Info.CB && Info.CB->isMustTailCall())
        return true;
      Info.LoweredTailCall = false;
      return false;
    };

    if (!Subtarget.hasTailCall())
      if (NoTail())
        return false;

    // Varargs calls cannot be tail calls because the buffer is on the stack
    if (Info.IsVarArg)
      if (NoTail())
        return false;

    // Do not tail call unless caller and callee return types match
    const TargetMachine &TM = TLI.getTargetMachine();
    Type *RetTy = F.getReturnType();
    SmallVector<MVT, 4> CallerRetTys;
    SmallVector<MVT, 4> CalleeRetTys;
    computeLegalValueVTs(F, TM, RetTy, CallerRetTys);
    computeLegalValueVTs(F, TM, Info.OrigRet.Ty, CalleeRetTys);
    bool TypesMatch = CallerRetTys.size() == CalleeRetTys.size() &&
                      std::equal(CallerRetTys.begin(), CallerRetTys.end(),
                                 CalleeRetTys.begin());
    if (!TypesMatch)
      if (NoTail())
        return false;

    // If pointers to local stack values are passed, we cannot tail call
    if (Info.CB) {
      for (auto &Arg : Info.CB->args()) {
        Value *Val = Arg.get();
        // Trace the value back through pointer operations
        while (true) {
          Value *Src = Val->stripPointerCastsAndAliases();
          if (auto *GEP = dyn_cast<GetElementPtrInst>(Src))
            Src = GEP->getPointerOperand();
          if (Val == Src)
            break;
          Val = Src;
        }
        if (isa<AllocaInst>(Val)) {
          if (NoTail())
            return false;
          break;
        }
      }
    }
  }

  if (Info.LoweredTailCall)
    MF.getFrameInfo().setHasTailCall();

  MachineInstrBuilder CallInst;

  bool IsIndirect = false;
  Register IndirectIdx;

  // Use indirect calls when callee is pointer in a register
  // or it's a weak (interposable) or non-function global.
  // Makes sure aliases behave.
  if (Info.Callee.isReg() ||
      (Info.Callee.isGlobal() &&
       (Info.Callee.getGlobal()->isInterposable() ||
        !Info.Callee.getGlobal()->getValueType()->isFunctionTy()))) {
    IsIndirect = true;
    CallInst = MIRBuilder.buildInstrNoInsert(Info.LoweredTailCall
                                         ? WebAssembly::RET_CALL_INDIRECT
                                         : WebAssembly::CALL_INDIRECT);
  } else {
    CallInst = MIRBuilder.buildInstrNoInsert(
        Info.LoweredTailCall ? WebAssembly::RET_CALL : WebAssembly::CALL);
  }

  if (Info.Callee.isReg()) {
    LLT CalleeType = MRI.getType(Info.Callee.getReg());
    assert(CalleeType.isPointer() &&
           "Trying to lower a call with a Callee other than a pointer???");

    // Placeholder for the type index.
    // This gets replaced with the correct value in WebAssemblyMCInstLower.cpp
    CallInst.addImm(0);

    MCSymbolWasm *Table;
    if (CalleeType.getAddressSpace() ==
        WebAssembly::WASM_ADDRESS_SPACE_DEFAULT) {
      Table = WebAssembly::getOrCreateFunctionTableSymbol(MF.getContext(),
                                                          &Subtarget);
      IndirectIdx = Info.Callee.getReg();

      auto PtrSize = CalleeType.getSizeInBits();
      auto PtrIntLLT = LLT::integer(PtrSize);

      IndirectIdx = MIRBuilder.buildPtrToInt(PtrIntLLT, IndirectIdx).getReg(0);
    } else if (CalleeType.getAddressSpace() ==
               WebAssembly::WASM_ADDRESS_SPACE_FUNCREF) {
      Table = WebAssembly::getOrCreateFuncrefCallTableSymbol(MF.getContext(),
                                                             &Subtarget);

      Type *PtrTy = PointerType::getUnqual(Ctx);
      LLT PtrLLT = getLLTForType(*PtrTy, DL);
      auto PtrIntLLT = LLT::integer(PtrLLT.getSizeInBits());

      IndirectIdx = MIRBuilder.buildConstant(PtrIntLLT, 0).getReg(0);

      auto TableSetInstr =
          MIRBuilder.buildInstr(WebAssembly::TABLE_SET_FUNCREF);
      TableSetInstr.addSym(Table);
      TableSetInstr.addUse(IndirectIdx);
      TableSetInstr.addUse(Info.Callee.getReg());

      constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *TableSetInstr,
                               TableSetInstr->getDesc(),
                               TableSetInstr->getOperand(1), 1);
      constrainOperandRegClass(MF, TRI, MRI, TII, RBI, *TableSetInstr,
                               TableSetInstr->getDesc(),
                               TableSetInstr->getOperand(2), 2);

    } else {
      return false;
    }

    if (Subtarget.hasCallIndirectOverlong()) {
      CallInst.addSym(Table);
    } else {
      // For the MVP there is at most one table whose number is 0, but we can't
      // write a table symbol or issue relocations.  Instead we just ensure the
      // table is live and write a zero.
      Table->setNoStrip();
      CallInst.addImm(0);
    }
  } else {
    if (Info.Callee.isGlobal()) {
      if (IsIndirect) {
        // Placeholder for the type index.
        // This gets replaced with the correct value in
        // WebAssemblyMCInstLower.cpp
        CallInst.addImm(0);

        Type *PtrTy = PointerType::getUnqual(Ctx);
        LLT PtrLLT = getLLTForType(*PtrTy, DL);
        auto PtrSize = PtrLLT.getSizeInBits();
        auto PtrIntLLT = LLT::integer(PtrSize);

        IndirectIdx =
            MIRBuilder.buildGlobalValue(PtrLLT, Info.Callee.getGlobal())
                .getReg(0);
        IndirectIdx =
            MIRBuilder.buildPtrToInt(PtrIntLLT, IndirectIdx).getReg(0);

        auto *Table = WebAssembly::getOrCreateFunctionTableSymbol(
            MF.getContext(), &Subtarget);
        if (Subtarget.hasCallIndirectOverlong()) {
          CallInst.addSym(Table);
        } else {
          // For the MVP there is at most one table whose number is 0, but we
          // can't write a table symbol or issue relocations.  Instead we just
          // ensure the table is live and write a zero.
          Table->setNoStrip();
          CallInst.addImm(0);
        }
      } else {
        CallInst.addGlobalAddress(Info.Callee.getGlobal());
      }
    } else if (Info.Callee.isSymbol()) {
      CallInst.addExternalSymbol(Info.Callee.getSymbolName());
    } else {
      llvm_unreachable("Trying to lower call with a callee other than reg, "
                       "global, or a symbol.");
    }
  }

  SmallVector<ArgInfo, 8> SplitArgs;

  bool HasSwiftErrorArg = false;
  bool HasSwiftSelfArg = false;

  for (const ArgInfo &Arg : Info.OrigArgs) {
    HasSwiftSelfArg |= Arg.Flags[0].isSwiftSelf();
    HasSwiftErrorArg |= Arg.Flags[0].isSwiftError();
    if (Arg.Flags[0].isNest())
      return false;
    if (Arg.Flags[0].isInAlloca())
      return false;
    if (Arg.Flags[0].isInConsecutiveRegs())
      return false;
    if (Arg.Flags[0].isInConsecutiveRegsLast())
      return false;

    if (Arg.Flags[0].isByVal() && Arg.Flags[0].getByValSize() != 0) {
      MachineFrameInfo &MFI = MF.getFrameInfo();

      unsigned MemSize = Arg.Flags[0].getByValSize();
      Align MemAlign = Arg.Flags[0].getNonZeroByValAlign();
      int FI = MFI.CreateStackObject(Arg.Flags[0].getByValSize(), MemAlign,
                                     /*isSS=*/false);

      auto StackAddrSpace = DL.getAllocaAddrSpace();
      auto PtrLLT =
          LLT::pointer(StackAddrSpace, DL.getPointerSizeInBits(StackAddrSpace));

      Register StackObjPtrVreg =
          MF.getRegInfo().createGenericVirtualRegister(PtrLLT);
      MRI.setRegClass(StackObjPtrVreg, TLI.getRepRegClassFor(TLI.getPointerTy(
                                           DL, StackAddrSpace)));

      MIRBuilder.buildFrameIndex(StackObjPtrVreg, FI);

      MachinePointerInfo DstPtrInfo = MachinePointerInfo::getFixedStack(MF, FI);

      MachinePointerInfo SrcPtrInfo(Arg.OrigValue);
      if (!Arg.OrigValue) {
        // We still need to accurately track the stack address space if we
        // don't know the underlying value.
        SrcPtrInfo = MachinePointerInfo::getUnknownStack(MF);
      }

      Align DstAlign =
          std::max(MemAlign, inferAlignFromPtrInfo(MF, DstPtrInfo));

      Align SrcAlign =
          std::max(MemAlign, inferAlignFromPtrInfo(MF, SrcPtrInfo));

      MachineMemOperand *SrcMMO = MF.getMachineMemOperand(
          SrcPtrInfo,
          MachineMemOperand::MOLoad | MachineMemOperand::MODereferenceable,
          MemSize, SrcAlign);

      MachineMemOperand *DstMMO = MF.getMachineMemOperand(
          DstPtrInfo,
          MachineMemOperand::MOStore | MachineMemOperand::MODereferenceable,
          MemSize, DstAlign);

      const LLT SizeTy = LLT::integer(PtrLLT.getSizeInBits());

      auto SizeConst = MIRBuilder.buildConstant(SizeTy, MemSize);
      MIRBuilder.buildMemCpy(StackObjPtrVreg, Arg.Regs[0], SizeConst, *DstMMO,
                             *SrcMMO);
    }

    splitToValueTypes(Arg, SplitArgs, DL, CallConv);
  }

  unsigned NumFixedArgs = 0;

  for (ArgInfo &Arg : SplitArgs) {
    const EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
    const MVT NewVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
    const LLT OrigLLT =
        getLLTForType(*OrigVT.getTypeForEVT(F.getContext()), DL);
    const LLT NewLLT = getLLTForWasmMVT(NewVT, DL);

    const TargetRegisterClass &NewRegClass = *TLI.getRegClassFor(NewVT);

    // If we need to split the type over multiple regs, check it's a scenario
    // we currently support.
    const unsigned NumParts =
        TLI.getNumRegistersForCallingConv(Ctx, CallConv, OrigVT);

    const ISD::ArgFlagsTy OrigFlags = Arg.Flags[0];
    Arg.Flags.clear();

    for (unsigned Part = 0; Part < NumParts; ++Part) {
      ISD::ArgFlagsTy Flags = OrigFlags;
      if (Part == 0) {
        Flags.setSplit();
      } else {
        Flags.setOrigAlign(Align(1));
        if (Part == NumParts - 1)
          Flags.setSplitEnd();
      }

      Arg.Flags.push_back(Flags);
    }

    Arg.OrigRegs.assign(Arg.Regs.begin(), Arg.Regs.end());
    if (NumParts != 1 || OrigLLT != NewLLT) {
      // If we can't directly assign the register, we need one or more
      // intermediate values.
      Arg.Regs.resize(NumParts);

      // For each split register, create and assign a vreg that will store
      // the incoming component of the larger value. These will later be
      // merged to form the final vreg.
      for (unsigned Part = 0; Part < NumParts; ++Part) {
        Register NewReg = MRI.createVirtualRegister(&NewRegClass);
        MRI.setType(NewReg, NewLLT);
        Arg.Regs[Part] = NewReg;

        if (!Arg.Flags[0].isVarArg())
          CallInst.addUse(NewReg);
      }

      buildCopyToRegs(MIRBuilder, Arg.Regs, Arg.OrigRegs[0], OrigLLT, NewLLT,
                      extendOpFromFlags(Arg.Flags[0]));
    } else {
      if (!Arg.Flags[0].isVarArg())
        CallInst.addUse(Arg.Regs[0]);
    }
  }

  if (CallConv == CallingConv::Swift) {
    Type *PtrTy = PointerType::getUnqual(Ctx);
    LLT PtrLLT = getLLTForType(*PtrTy, DL);
    auto &PtrRegClass = *TLI.getRegClassFor(TLI.getSimpleValueType(DL, PtrTy));

    if (!HasSwiftSelfArg) {
      auto NewUndefReg = MIRBuilder.buildUndef(PtrLLT).getReg(0);
      MRI.setRegClass(NewUndefReg, &PtrRegClass);
      CallInst.addUse(NewUndefReg);
    }
    if (!HasSwiftErrorArg) {
      auto NewUndefReg = MIRBuilder.buildUndef(PtrLLT).getReg(0);
      MRI.setRegClass(NewUndefReg, &PtrRegClass);
      CallInst.addUse(NewUndefReg);
    }
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, Info.IsVarArg, MF, ArgLocs, Ctx);

  if (Info.IsVarArg) {
    // Outgoing non-fixed arguments are placed in a buffer. First
    // compute their offsets and the total amount of buffer space needed.
    for (ArgInfo &Arg : drop_begin(SplitArgs, NumFixedArgs)) {
      EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
      MVT PartVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
      Type *Ty = EVT(PartVT).getTypeForEVT(Ctx);

      for (unsigned Part = 0; Part < Arg.Regs.size(); ++Part) {
        Align Alignment = std::max(Arg.Flags[Part].getNonZeroOrigAlign(),
                                   DL.getABITypeAlign(Ty));
        unsigned Offset =
            CCInfo.AllocateStack(DL.getTypeAllocSize(Ty), Alignment);
        CCInfo.addLoc(CCValAssign::getMem(ArgLocs.size(), PartVT, Offset,
                                          PartVT, CCValAssign::Full));
      }
    }
  }

  unsigned NumBytes = CCInfo.getAlignedCallFrameSize();

  auto StackAddrSpace = DL.getAllocaAddrSpace();
  auto PtrLLT = LLT::pointer(StackAddrSpace, DL.getPointerSizeInBits(0));
  auto SizeLLT = LLT::integer(DL.getPointerSizeInBits(StackAddrSpace));
  auto *PtrRegClass = TLI.getRegClassFor(TLI.getPointerTy(DL, StackAddrSpace));

  if (Info.IsVarArg && NumBytes) {
    Register VarArgStackPtr =
        MF.getRegInfo().createGenericVirtualRegister(PtrLLT);
    MRI.setRegClass(VarArgStackPtr, PtrRegClass);

    MaybeAlign StackAlign = DL.getStackAlignment();
    assert(StackAlign && "data layout string is missing stack alignment");
    int FI = MF.getFrameInfo().CreateStackObject(NumBytes, *StackAlign,
                                                 /*isSS=*/false);

    MIRBuilder.buildFrameIndex(VarArgStackPtr, FI);

    unsigned ValNo = 0;
    for (ArgInfo &Arg : drop_begin(SplitArgs, NumFixedArgs)) {
      EVT OrigVT = TLI.getValueType(DL, Arg.Ty);
      MVT PartVT = TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
      Type *Ty = EVT(PartVT).getTypeForEVT(Ctx);

      for (unsigned Part = 0; Part < Arg.Regs.size(); ++Part) {
        Align Alignment = std::max(Arg.Flags[Part].getNonZeroOrigAlign(),
                                   DL.getABITypeAlign(Ty));

        unsigned Offset = ArgLocs[ValNo++].getLocMemOffset();

        Register DstPtr =
            MIRBuilder
                .buildPtrAdd(
                    PtrLLT, VarArgStackPtr,
                    MIRBuilder.buildConstant(SizeLLT, Offset).getReg(0),
                    MachineInstr::MIFlag::NoUWrap)
                .getReg(0);

        MachineMemOperand *DstMMO = MF.getMachineMemOperand(
            MachinePointerInfo::getFixedStack(MF, FI, Offset),
            MachineMemOperand::MOStore | MachineMemOperand::MODereferenceable,
            PartVT.getStoreSize(), Alignment);

        MIRBuilder.buildStore(Arg.Regs[Part], DstPtr, *DstMMO);
      }
    }

    CallInst.addUse(VarArgStackPtr);
  } else if (Info.IsVarArg) {
    auto NewArgReg = MIRBuilder.buildConstant(PtrLLT, 0).getReg(0);
    MRI.setRegClass(NewArgReg, PtrRegClass);
    CallInst.addUse(NewArgReg);
  }

  if (IsIndirect) {
    auto NewArgReg =
        constrainRegToClass(MRI, TII, RBI, IndirectIdx, *PtrRegClass);
    if (IndirectIdx != NewArgReg)
      MIRBuilder.buildCopy(NewArgReg, IndirectIdx);
    CallInst.addUse(IndirectIdx);
  }

  MIRBuilder.insertInstr(CallInst);

  if (!Info.LoweredTailCall) {
    if (Info.CanLowerReturn && !Info.OrigRet.Ty->isVoidTy()) {
      SmallVector<EVT, 4> SplitEVTs;
      ComputeValueVTs(TLI, DL, Info.OrigRet.Ty, SplitEVTs);
      assert(Info.OrigRet.Regs.size() == SplitEVTs.size() &&
            "For each split Type there should be exactly one VReg.");

      SmallVector<ArgInfo, 8> SplitRets;

      unsigned RetIdx = 0;
      for (EVT SplitEVT : SplitEVTs) {
        Register CurVReg = Info.OrigRet.Regs[RetIdx];
        ArgInfo CurArgInfo = ArgInfo{CurVReg, SplitEVT.getTypeForEVT(Ctx), 0};

        if (Info.CB) {
          setArgFlags(CurArgInfo, AttributeList::ReturnIndex, DL, *Info.CB);
        } else {
          // We don't have a call base, so chances are we're looking at a
          // libcall (external symbol).

          // TODO: figure out how to get ALL the correct attributes
          ISD::ArgFlagsTy &Flags = CurArgInfo.Flags[0];
          PointerType *PtrTy =
              dyn_cast<PointerType>(CurArgInfo.Ty->getScalarType());
          if (PtrTy) {
            Flags.setPointer();
            Flags.setPointerAddrSpace(PtrTy->getPointerAddressSpace());
          }
          Align MemAlign = DL.getABITypeAlign(CurArgInfo.Ty);
          Flags.setMemAlign(MemAlign);
          Flags.setOrigAlign(MemAlign);
        }

        splitToValueTypes(CurArgInfo, SplitRets, DL, CallConv);
        ++RetIdx;
      }

      for (ArgInfo &Ret : SplitRets) {
        const EVT OrigVT = TLI.getValueType(DL, Ret.Ty);
        const MVT NewVT =
            TLI.getRegisterTypeForCallingConv(Ctx, CallConv, OrigVT);
        const LLT OrigLLT =
            getLLTForType(*OrigVT.getTypeForEVT(F.getContext()), DL);
        const LLT NewLLT = getLLTForWasmMVT(NewVT, DL);

        const TargetRegisterClass &NewRegClass = *TLI.getRegClassFor(NewVT);

        // If we need to split the type over multiple regs, check it's a scenario
        // we currently support.
        const unsigned NumParts =
            TLI.getNumRegistersForCallingConv(Ctx, CallConv, OrigVT);

        const ISD::ArgFlagsTy OrigFlags = Ret.Flags[0];
        Ret.Flags.clear();

        for (unsigned Part = 0; Part < NumParts; ++Part) {
          ISD::ArgFlagsTy Flags = OrigFlags;
          if (Part == 0) {
            Flags.setSplit();
          } else {
            Flags.setOrigAlign(Align(1));
            if (Part == NumParts - 1)
              Flags.setSplitEnd();
          }

          Ret.Flags.push_back(Flags);
        }

        Ret.OrigRegs.assign(Ret.Regs.begin(), Ret.Regs.end());

        if (NumParts != 1 || OrigLLT != NewLLT) {
          // If we can't directly assign the register, we need one or more
          // intermediate values.
          Ret.Regs.resize(NumParts);

          // For each split register, create and assign a vreg that will store
          // the incoming component of the larger value. These will later be
          // merged to form the final vreg.
          for (unsigned Part = 0; Part < NumParts; ++Part) {
            Register NewReg = MRI.createVirtualRegister(&NewRegClass);
            MRI.setType(NewReg, NewLLT);
            CallInst.addDef(NewReg);
            Ret.Regs[Part] = NewReg;
          }

          buildCopyFromRegs(MIRBuilder, Ret.OrigRegs, Ret.Regs, OrigLLT, NewLLT,
                            Ret.Flags[0]);
        } else {
          CallInst.addDef(Ret.Regs[0]);
        }
      }
    }
  }

  return true;
}
