//=- WebAssemblyISelLowering.cpp - WebAssembly DAG Lowering Implementation -==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the WebAssemblyTargetLowering class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyISelLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/WasmEHFuncInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsWebAssembly.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-lower"

WebAssemblyTargetLowering::WebAssemblyTargetLowering(
    const TargetMachine &TM, const WebAssemblySubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  auto MVTPtr = Subtarget->hasAddr64() ? MVT::i64 : MVT::i32;

  // Booleans always contain 0 or 1.
  setBooleanContents(ZeroOrOneBooleanContent);
  // Except in SIMD vectors
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);
  // We don't know the microarchitecture here, so just reduce register pressure.
  setSchedulingPreference(Sched::RegPressure);
  // Tell ISel that we have a stack pointer.
  setStackPointerRegisterToSaveRestore(
      Subtarget->hasAddr64() ? WebAssembly::SP64 : WebAssembly::SP32);
  // Set up the register classes.
  addRegisterClass(MVT::i32, &WebAssembly::I32RegClass);
  addRegisterClass(MVT::i64, &WebAssembly::I64RegClass);
  addRegisterClass(MVT::f32, &WebAssembly::F32RegClass);
  addRegisterClass(MVT::f64, &WebAssembly::F64RegClass);
  if (Subtarget->hasSIMD128()) {
    addRegisterClass(MVT::v16i8, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v8i16, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v4i32, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v4f32, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v2i64, &WebAssembly::V128RegClass);
    addRegisterClass(MVT::v2f64, &WebAssembly::V128RegClass);
  }
  // Compute derived properties from the register classes.
  computeRegisterProperties(Subtarget->getRegisterInfo());

  setOperationAction(ISD::GlobalAddress, MVTPtr, Custom);
  setOperationAction(ISD::ExternalSymbol, MVTPtr, Custom);
  setOperationAction(ISD::JumpTable, MVTPtr, Custom);
  setOperationAction(ISD::BlockAddress, MVTPtr, Custom);
  setOperationAction(ISD::BRIND, MVT::Other, Custom);

  // Take the default expansion for va_arg, va_copy, and va_end. There is no
  // default action for va_start, so we do that custom.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  for (auto T : {MVT::f32, MVT::f64, MVT::v4f32, MVT::v2f64}) {
    // Don't expand the floating-point types to constant pools.
    setOperationAction(ISD::ConstantFP, T, Legal);
    // Expand floating-point comparisons.
    for (auto CC : {ISD::SETO, ISD::SETUO, ISD::SETUEQ, ISD::SETONE,
                    ISD::SETULT, ISD::SETULE, ISD::SETUGT, ISD::SETUGE})
      setCondCodeAction(CC, T, Expand);
    // Expand floating-point library function operators.
    for (auto Op :
         {ISD::FSIN, ISD::FCOS, ISD::FSINCOS, ISD::FPOW, ISD::FREM, ISD::FMA})
      setOperationAction(Op, T, Expand);
    // Note supported floating-point library function operators that otherwise
    // default to expand.
    for (auto Op :
         {ISD::FCEIL, ISD::FFLOOR, ISD::FTRUNC, ISD::FNEARBYINT, ISD::FRINT})
      setOperationAction(Op, T, Legal);
    // Support minimum and maximum, which otherwise default to expand.
    setOperationAction(ISD::FMINIMUM, T, Legal);
    setOperationAction(ISD::FMAXIMUM, T, Legal);
    // WebAssembly currently has no builtin f16 support.
    setOperationAction(ISD::FP16_TO_FP, T, Expand);
    setOperationAction(ISD::FP_TO_FP16, T, Expand);
    setLoadExtAction(ISD::EXTLOAD, T, MVT::f16, Expand);
    setTruncStoreAction(T, MVT::f16, Expand);
  }

  // Expand unavailable integer operations.
  for (auto Op :
       {ISD::BSWAP, ISD::SMUL_LOHI, ISD::UMUL_LOHI, ISD::MULHS, ISD::MULHU,
        ISD::SDIVREM, ISD::UDIVREM, ISD::SHL_PARTS, ISD::SRA_PARTS,
        ISD::SRL_PARTS, ISD::ADDC, ISD::ADDE, ISD::SUBC, ISD::SUBE}) {
    for (auto T : {MVT::i32, MVT::i64})
      setOperationAction(Op, T, Expand);
    if (Subtarget->hasSIMD128())
      for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v2i64})
        setOperationAction(Op, T, Expand);
  }

  // SIMD-specific configuration
  if (Subtarget->hasSIMD128()) {
    // Hoist bitcasts out of shuffles
    setTargetDAGCombine(ISD::VECTOR_SHUFFLE);

    // Combine extends of extract_subvectors into widening ops
    setTargetDAGCombine(ISD::SIGN_EXTEND);
    setTargetDAGCombine(ISD::ZERO_EXTEND);

    // Support saturating add for i8x16 and i16x8
    for (auto Op : {ISD::SADDSAT, ISD::UADDSAT})
      for (auto T : {MVT::v16i8, MVT::v8i16})
        setOperationAction(Op, T, Legal);

    // Support integer abs
    for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32})
      setOperationAction(ISD::ABS, T, Legal);

    // Custom lower BUILD_VECTORs to minimize number of replace_lanes
    for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v4f32, MVT::v2i64,
                   MVT::v2f64})
      setOperationAction(ISD::BUILD_VECTOR, T, Custom);

    // We have custom shuffle lowering to expose the shuffle mask
    for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v4f32, MVT::v2i64,
                   MVT::v2f64})
      setOperationAction(ISD::VECTOR_SHUFFLE, T, Custom);

    // Custom lowering since wasm shifts must have a scalar shift amount
    for (auto Op : {ISD::SHL, ISD::SRA, ISD::SRL})
      for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v2i64})
        setOperationAction(Op, T, Custom);

    // Custom lower lane accesses to expand out variable indices
    for (auto Op : {ISD::EXTRACT_VECTOR_ELT, ISD::INSERT_VECTOR_ELT})
      for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v4f32, MVT::v2i64,
                     MVT::v2f64})
        setOperationAction(Op, T, Custom);

    // There is no i8x16.mul instruction
    setOperationAction(ISD::MUL, MVT::v16i8, Expand);

    // There is no vector conditional select instruction
    for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v4f32, MVT::v2i64,
                   MVT::v2f64})
      setOperationAction(ISD::SELECT_CC, T, Expand);

    // Expand integer operations supported for scalars but not SIMD
    for (auto Op : {ISD::CTLZ, ISD::CTTZ, ISD::CTPOP, ISD::SDIV, ISD::UDIV,
                    ISD::SREM, ISD::UREM, ISD::ROTL, ISD::ROTR})
      for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v2i64})
        setOperationAction(Op, T, Expand);

    // But we do have integer min and max operations
    for (auto Op : {ISD::SMIN, ISD::SMAX, ISD::UMIN, ISD::UMAX})
      for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32})
        setOperationAction(Op, T, Legal);

    // Expand float operations supported for scalars but not SIMD
    for (auto Op : {ISD::FCEIL, ISD::FFLOOR, ISD::FTRUNC, ISD::FNEARBYINT,
                    ISD::FCOPYSIGN, ISD::FLOG, ISD::FLOG2, ISD::FLOG10,
                    ISD::FEXP, ISD::FEXP2, ISD::FRINT})
      for (auto T : {MVT::v4f32, MVT::v2f64})
        setOperationAction(Op, T, Expand);

    // Expand operations not supported for i64x2 vectors
    for (unsigned CC = 0; CC < ISD::SETCC_INVALID; ++CC)
      setCondCodeAction(static_cast<ISD::CondCode>(CC), MVT::v2i64, Custom);

    // 64x2 conversions are not in the spec
    for (auto Op :
         {ISD::SINT_TO_FP, ISD::UINT_TO_FP, ISD::FP_TO_SINT, ISD::FP_TO_UINT})
      for (auto T : {MVT::v2i64, MVT::v2f64})
        setOperationAction(Op, T, Expand);
  }

  // As a special case, these operators use the type to mean the type to
  // sign-extend from.
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  if (!Subtarget->hasSignExt()) {
    // Sign extends are legal only when extending a vector extract
    auto Action = Subtarget->hasSIMD128() ? Custom : Expand;
    for (auto T : {MVT::i8, MVT::i16, MVT::i32})
      setOperationAction(ISD::SIGN_EXTEND_INREG, T, Action);
  }
  for (auto T : MVT::integer_fixedlen_vector_valuetypes())
    setOperationAction(ISD::SIGN_EXTEND_INREG, T, Expand);

  // Dynamic stack allocation: use the default expansion.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVTPtr, Expand);

  setOperationAction(ISD::FrameIndex, MVT::i32, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i64, Custom);
  setOperationAction(ISD::CopyToReg, MVT::Other, Custom);

  // Expand these forms; we pattern-match the forms that we can handle in isel.
  for (auto T : {MVT::i32, MVT::i64, MVT::f32, MVT::f64})
    for (auto Op : {ISD::BR_CC, ISD::SELECT_CC})
      setOperationAction(Op, T, Expand);

  // We have custom switch handling.
  setOperationAction(ISD::BR_JT, MVT::Other, Custom);

  // WebAssembly doesn't have:
  //  - Floating-point extending loads.
  //  - Floating-point truncating stores.
  //  - i1 extending loads.
  //  - truncating SIMD stores and most extending loads
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  for (auto T : MVT::integer_valuetypes())
    for (auto Ext : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD})
      setLoadExtAction(Ext, T, MVT::i1, Promote);
  if (Subtarget->hasSIMD128()) {
    for (auto T : {MVT::v16i8, MVT::v8i16, MVT::v4i32, MVT::v2i64, MVT::v4f32,
                   MVT::v2f64}) {
      for (auto MemT : MVT::fixedlen_vector_valuetypes()) {
        if (MVT(T) != MemT) {
          setTruncStoreAction(T, MemT, Expand);
          for (auto Ext : {ISD::EXTLOAD, ISD::ZEXTLOAD, ISD::SEXTLOAD})
            setLoadExtAction(Ext, T, MemT, Expand);
        }
      }
    }
    // But some vector extending loads are legal
    for (auto Ext : {ISD::EXTLOAD, ISD::SEXTLOAD, ISD::ZEXTLOAD}) {
      setLoadExtAction(Ext, MVT::v8i16, MVT::v8i8, Legal);
      setLoadExtAction(Ext, MVT::v4i32, MVT::v4i16, Legal);
      setLoadExtAction(Ext, MVT::v2i64, MVT::v2i32, Legal);
    }
    // And some truncating stores are legal as well
    setTruncStoreAction(MVT::v8i16, MVT::v8i8, Legal);
    setTruncStoreAction(MVT::v4i32, MVT::v4i16, Legal);
  }

  // Don't do anything clever with build_pairs
  setOperationAction(ISD::BUILD_PAIR, MVT::i64, Expand);

  // Trap lowers to wasm unreachable
  setOperationAction(ISD::TRAP, MVT::Other, Legal);
  setOperationAction(ISD::DEBUGTRAP, MVT::Other, Legal);

  // Exception handling intrinsics
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);
  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  setMaxAtomicSizeInBitsSupported(64);

  // Override the __gnu_f2h_ieee/__gnu_h2f_ieee names so that the f32 name is
  // consistent with the f64 and f128 names.
  setLibcallName(RTLIB::FPEXT_F16_F32, "__extendhfsf2");
  setLibcallName(RTLIB::FPROUND_F32_F16, "__truncsfhf2");

  // Define the emscripten name for return address helper.
  // TODO: when implementing other Wasm backends, make this generic or only do
  // this on emscripten depending on what they end up doing.
  setLibcallName(RTLIB::RETURN_ADDRESS, "emscripten_return_address");

  // Always convert switches to br_tables unless there is only one case, which
  // is equivalent to a simple branch. This reduces code size for wasm, and we
  // defer possible jump table optimizations to the VM.
  setMinimumJumpTableEntries(2);
}

TargetLowering::AtomicExpansionKind
WebAssemblyTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *AI) const {
  // We have wasm instructions for these
  switch (AI->getOperation()) {
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::Xchg:
    return AtomicExpansionKind::None;
  default:
    break;
  }
  return AtomicExpansionKind::CmpXChg;
}

FastISel *WebAssemblyTargetLowering::createFastISel(
    FunctionLoweringInfo &FuncInfo, const TargetLibraryInfo *LibInfo) const {
  return WebAssembly::createFastISel(FuncInfo, LibInfo);
}

MVT WebAssemblyTargetLowering::getScalarShiftAmountTy(const DataLayout & /*DL*/,
                                                      EVT VT) const {
  unsigned BitWidth = NextPowerOf2(VT.getSizeInBits() - 1);
  if (BitWidth > 1 && BitWidth < 8)
    BitWidth = 8;

  if (BitWidth > 64) {
    // The shift will be lowered to a libcall, and compiler-rt libcalls expect
    // the count to be an i32.
    BitWidth = 32;
    assert(BitWidth >= Log2_32_Ceil(VT.getSizeInBits()) &&
           "32-bit shift counts ought to be enough for anyone");
  }

  MVT Result = MVT::getIntegerVT(BitWidth);
  assert(Result != MVT::INVALID_SIMPLE_VALUE_TYPE &&
         "Unable to represent scalar shift amount type");
  return Result;
}

// Lower an fp-to-int conversion operator from the LLVM opcode, which has an
// undefined result on invalid/overflow, to the WebAssembly opcode, which
// traps on invalid/overflow.
static MachineBasicBlock *LowerFPToInt(MachineInstr &MI, DebugLoc DL,
                                       MachineBasicBlock *BB,
                                       const TargetInstrInfo &TII,
                                       bool IsUnsigned, bool Int64,
                                       bool Float64, unsigned LoweredOpcode) {
  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  Register OutReg = MI.getOperand(0).getReg();
  Register InReg = MI.getOperand(1).getReg();

  unsigned Abs = Float64 ? WebAssembly::ABS_F64 : WebAssembly::ABS_F32;
  unsigned FConst = Float64 ? WebAssembly::CONST_F64 : WebAssembly::CONST_F32;
  unsigned LT = Float64 ? WebAssembly::LT_F64 : WebAssembly::LT_F32;
  unsigned GE = Float64 ? WebAssembly::GE_F64 : WebAssembly::GE_F32;
  unsigned IConst = Int64 ? WebAssembly::CONST_I64 : WebAssembly::CONST_I32;
  unsigned Eqz = WebAssembly::EQZ_I32;
  unsigned And = WebAssembly::AND_I32;
  int64_t Limit = Int64 ? INT64_MIN : INT32_MIN;
  int64_t Substitute = IsUnsigned ? 0 : Limit;
  double CmpVal = IsUnsigned ? -(double)Limit * 2.0 : -(double)Limit;
  auto &Context = BB->getParent()->getFunction().getContext();
  Type *Ty = Float64 ? Type::getDoubleTy(Context) : Type::getFloatTy(Context);

  const BasicBlock *LLVMBB = BB->getBasicBlock();
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *TrueMBB = F->CreateMachineBasicBlock(LLVMBB);
  MachineBasicBlock *FalseMBB = F->CreateMachineBasicBlock(LLVMBB);
  MachineBasicBlock *DoneMBB = F->CreateMachineBasicBlock(LLVMBB);

  MachineFunction::iterator It = ++BB->getIterator();
  F->insert(It, FalseMBB);
  F->insert(It, TrueMBB);
  F->insert(It, DoneMBB);

  // Transfer the remainder of BB and its successor edges to DoneMBB.
  DoneMBB->splice(DoneMBB->begin(), BB, std::next(MI.getIterator()), BB->end());
  DoneMBB->transferSuccessorsAndUpdatePHIs(BB);

  BB->addSuccessor(TrueMBB);
  BB->addSuccessor(FalseMBB);
  TrueMBB->addSuccessor(DoneMBB);
  FalseMBB->addSuccessor(DoneMBB);

  unsigned Tmp0, Tmp1, CmpReg, EqzReg, FalseReg, TrueReg;
  Tmp0 = MRI.createVirtualRegister(MRI.getRegClass(InReg));
  Tmp1 = MRI.createVirtualRegister(MRI.getRegClass(InReg));
  CmpReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  EqzReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
  FalseReg = MRI.createVirtualRegister(MRI.getRegClass(OutReg));
  TrueReg = MRI.createVirtualRegister(MRI.getRegClass(OutReg));

  MI.eraseFromParent();
  // For signed numbers, we can do a single comparison to determine whether
  // fabs(x) is within range.
  if (IsUnsigned) {
    Tmp0 = InReg;
  } else {
    BuildMI(BB, DL, TII.get(Abs), Tmp0).addReg(InReg);
  }
  BuildMI(BB, DL, TII.get(FConst), Tmp1)
      .addFPImm(cast<ConstantFP>(ConstantFP::get(Ty, CmpVal)));
  BuildMI(BB, DL, TII.get(LT), CmpReg).addReg(Tmp0).addReg(Tmp1);

  // For unsigned numbers, we have to do a separate comparison with zero.
  if (IsUnsigned) {
    Tmp1 = MRI.createVirtualRegister(MRI.getRegClass(InReg));
    Register SecondCmpReg =
        MRI.createVirtualRegister(&WebAssembly::I32RegClass);
    Register AndReg = MRI.createVirtualRegister(&WebAssembly::I32RegClass);
    BuildMI(BB, DL, TII.get(FConst), Tmp1)
        .addFPImm(cast<ConstantFP>(ConstantFP::get(Ty, 0.0)));
    BuildMI(BB, DL, TII.get(GE), SecondCmpReg).addReg(Tmp0).addReg(Tmp1);
    BuildMI(BB, DL, TII.get(And), AndReg).addReg(CmpReg).addReg(SecondCmpReg);
    CmpReg = AndReg;
  }

  BuildMI(BB, DL, TII.get(Eqz), EqzReg).addReg(CmpReg);

  // Create the CFG diamond to select between doing the conversion or using
  // the substitute value.
  BuildMI(BB, DL, TII.get(WebAssembly::BR_IF)).addMBB(TrueMBB).addReg(EqzReg);
  BuildMI(FalseMBB, DL, TII.get(LoweredOpcode), FalseReg).addReg(InReg);
  BuildMI(FalseMBB, DL, TII.get(WebAssembly::BR)).addMBB(DoneMBB);
  BuildMI(TrueMBB, DL, TII.get(IConst), TrueReg).addImm(Substitute);
  BuildMI(*DoneMBB, DoneMBB->begin(), DL, TII.get(TargetOpcode::PHI), OutReg)
      .addReg(FalseReg)
      .addMBB(FalseMBB)
      .addReg(TrueReg)
      .addMBB(TrueMBB);

  return DoneMBB;
}

static MachineBasicBlock *LowerCallResults(MachineInstr &CallResults,
                                           DebugLoc DL, MachineBasicBlock *BB,
                                           const TargetInstrInfo &TII) {
  MachineInstr &CallParams = *CallResults.getPrevNode();
  assert(CallParams.getOpcode() == WebAssembly::CALL_PARAMS);
  assert(CallResults.getOpcode() == WebAssembly::CALL_RESULTS ||
         CallResults.getOpcode() == WebAssembly::RET_CALL_RESULTS);

  bool IsIndirect = CallParams.getOperand(0).isReg();
  bool IsRetCall = CallResults.getOpcode() == WebAssembly::RET_CALL_RESULTS;

  unsigned CallOp;
  if (IsIndirect && IsRetCall) {
    CallOp = WebAssembly::RET_CALL_INDIRECT;
  } else if (IsIndirect) {
    CallOp = WebAssembly::CALL_INDIRECT;
  } else if (IsRetCall) {
    CallOp = WebAssembly::RET_CALL;
  } else {
    CallOp = WebAssembly::CALL;
  }

  MachineFunction &MF = *BB->getParent();
  const MCInstrDesc &MCID = TII.get(CallOp);
  MachineInstrBuilder MIB(MF, MF.CreateMachineInstr(MCID, DL));

  // See if we must truncate the function pointer.
  // CALL_INDIRECT takes an i32, but in wasm64 we represent function pointers
  // as 64-bit for uniformity with other pointer types.
  if (IsIndirect && MF.getSubtarget<WebAssemblySubtarget>().hasAddr64()) {
    Register Reg32 =
        MF.getRegInfo().createVirtualRegister(&WebAssembly::I32RegClass);
    auto &FnPtr = CallParams.getOperand(0);
    BuildMI(*BB, CallResults.getIterator(), DL,
            TII.get(WebAssembly::I32_WRAP_I64), Reg32)
        .addReg(FnPtr.getReg());
    FnPtr.setReg(Reg32);
  }

  // Move the function pointer to the end of the arguments for indirect calls
  if (IsIndirect) {
    auto FnPtr = CallParams.getOperand(0);
    CallParams.RemoveOperand(0);
    CallParams.addOperand(FnPtr);
  }

  for (auto Def : CallResults.defs())
    MIB.add(Def);

  // Add placeholders for the type index and immediate flags
  if (IsIndirect) {
    MIB.addImm(0);
    MIB.addImm(0);
  }

  for (auto Use : CallParams.uses())
    MIB.add(Use);

  BB->insert(CallResults.getIterator(), MIB);
  CallParams.eraseFromParent();
  CallResults.eraseFromParent();

  return BB;
}

MachineBasicBlock *WebAssemblyTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *BB) const {
  const TargetInstrInfo &TII = *Subtarget->getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case WebAssembly::FP_TO_SINT_I32_F32:
    return LowerFPToInt(MI, DL, BB, TII, false, false, false,
                        WebAssembly::I32_TRUNC_S_F32);
  case WebAssembly::FP_TO_UINT_I32_F32:
    return LowerFPToInt(MI, DL, BB, TII, true, false, false,
                        WebAssembly::I32_TRUNC_U_F32);
  case WebAssembly::FP_TO_SINT_I64_F32:
    return LowerFPToInt(MI, DL, BB, TII, false, true, false,
                        WebAssembly::I64_TRUNC_S_F32);
  case WebAssembly::FP_TO_UINT_I64_F32:
    return LowerFPToInt(MI, DL, BB, TII, true, true, false,
                        WebAssembly::I64_TRUNC_U_F32);
  case WebAssembly::FP_TO_SINT_I32_F64:
    return LowerFPToInt(MI, DL, BB, TII, false, false, true,
                        WebAssembly::I32_TRUNC_S_F64);
  case WebAssembly::FP_TO_UINT_I32_F64:
    return LowerFPToInt(MI, DL, BB, TII, true, false, true,
                        WebAssembly::I32_TRUNC_U_F64);
  case WebAssembly::FP_TO_SINT_I64_F64:
    return LowerFPToInt(MI, DL, BB, TII, false, true, true,
                        WebAssembly::I64_TRUNC_S_F64);
  case WebAssembly::FP_TO_UINT_I64_F64:
    return LowerFPToInt(MI, DL, BB, TII, true, true, true,
                        WebAssembly::I64_TRUNC_U_F64);
  case WebAssembly::CALL_RESULTS:
  case WebAssembly::RET_CALL_RESULTS:
    return LowerCallResults(MI, DL, BB, TII);
  }
}

const char *
WebAssemblyTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (static_cast<WebAssemblyISD::NodeType>(Opcode)) {
  case WebAssemblyISD::FIRST_NUMBER:
  case WebAssemblyISD::FIRST_MEM_OPCODE:
    break;
#define HANDLE_NODETYPE(NODE)                                                  \
  case WebAssemblyISD::NODE:                                                   \
    return "WebAssemblyISD::" #NODE;
#define HANDLE_MEM_NODETYPE(NODE) HANDLE_NODETYPE(NODE)
#include "WebAssemblyISD.def"
#undef HANDLE_MEM_NODETYPE
#undef HANDLE_NODETYPE
  }
  return nullptr;
}

std::pair<unsigned, const TargetRegisterClass *>
WebAssemblyTargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  // First, see if this is a constraint that directly corresponds to a
  // WebAssembly register class.
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      assert(VT != MVT::iPTR && "Pointer MVT not expected here");
      if (Subtarget->hasSIMD128() && VT.isVector()) {
        if (VT.getSizeInBits() == 128)
          return std::make_pair(0U, &WebAssembly::V128RegClass);
      }
      if (VT.isInteger() && !VT.isVector()) {
        if (VT.getSizeInBits() <= 32)
          return std::make_pair(0U, &WebAssembly::I32RegClass);
        if (VT.getSizeInBits() <= 64)
          return std::make_pair(0U, &WebAssembly::I64RegClass);
      }
      break;
    default:
      break;
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

bool WebAssemblyTargetLowering::isCheapToSpeculateCttz() const {
  // Assume ctz is a relatively cheap operation.
  return true;
}

bool WebAssemblyTargetLowering::isCheapToSpeculateCtlz() const {
  // Assume clz is a relatively cheap operation.
  return true;
}

bool WebAssemblyTargetLowering::isLegalAddressingMode(const DataLayout &DL,
                                                      const AddrMode &AM,
                                                      Type *Ty, unsigned AS,
                                                      Instruction *I) const {
  // WebAssembly offsets are added as unsigned without wrapping. The
  // isLegalAddressingMode gives us no way to determine if wrapping could be
  // happening, so we approximate this by accepting only non-negative offsets.
  if (AM.BaseOffs < 0)
    return false;

  // WebAssembly has no scale register operands.
  if (AM.Scale != 0)
    return false;

  // Everything else is legal.
  return true;
}

bool WebAssemblyTargetLowering::allowsMisalignedMemoryAccesses(
    EVT /*VT*/, unsigned /*AddrSpace*/, unsigned /*Align*/,
    MachineMemOperand::Flags /*Flags*/, bool *Fast) const {
  // WebAssembly supports unaligned accesses, though it should be declared
  // with the p2align attribute on loads and stores which do so, and there
  // may be a performance impact. We tell LLVM they're "fast" because
  // for the kinds of things that LLVM uses this for (merging adjacent stores
  // of constants, etc.), WebAssembly implementations will either want the
  // unaligned access or they'll split anyway.
  if (Fast)
    *Fast = true;
  return true;
}

bool WebAssemblyTargetLowering::isIntDivCheap(EVT VT,
                                              AttributeList Attr) const {
  // The current thinking is that wasm engines will perform this optimization,
  // so we can save on code size.
  return true;
}

bool WebAssemblyTargetLowering::isVectorLoadExtDesirable(SDValue ExtVal) const {
  EVT ExtT = ExtVal.getValueType();
  EVT MemT = cast<LoadSDNode>(ExtVal->getOperand(0))->getValueType(0);
  return (ExtT == MVT::v8i16 && MemT == MVT::v8i8) ||
         (ExtT == MVT::v4i32 && MemT == MVT::v4i16) ||
         (ExtT == MVT::v2i64 && MemT == MVT::v2i32);
}

EVT WebAssemblyTargetLowering::getSetCCResultType(const DataLayout &DL,
                                                  LLVMContext &C,
                                                  EVT VT) const {
  if (VT.isVector())
    return VT.changeVectorElementTypeToInteger();

  // So far, all branch instructions in Wasm take an I32 condition.
  // The default TargetLowering::getSetCCResultType returns the pointer size,
  // which would be useful to reduce instruction counts when testing
  // against 64-bit pointers/values if at some point Wasm supports that.
  return EVT::getIntegerVT(C, 32);
}

bool WebAssemblyTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                                   const CallInst &I,
                                                   MachineFunction &MF,
                                                   unsigned Intrinsic) const {
  switch (Intrinsic) {
  case Intrinsic::wasm_atomic_notify:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    // atomic.notify instruction does not really load the memory specified with
    // this argument, but MachineMemOperand should either be load or store, so
    // we set this to a load.
    // FIXME Volatile isn't really correct, but currently all LLVM atomic
    // instructions are treated as volatiles in the backend, so we should be
    // consistent. The same applies for wasm_atomic_wait intrinsics too.
    Info.flags = MachineMemOperand::MOVolatile | MachineMemOperand::MOLoad;
    return true;
  case Intrinsic::wasm_atomic_wait_i32:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i32;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(4);
    Info.flags = MachineMemOperand::MOVolatile | MachineMemOperand::MOLoad;
    return true;
  case Intrinsic::wasm_atomic_wait_i64:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::i64;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Align(8);
    Info.flags = MachineMemOperand::MOVolatile | MachineMemOperand::MOLoad;
    return true;
  case Intrinsic::wasm_load32_zero:
  case Intrinsic::wasm_load64_zero:
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = Intrinsic == Intrinsic::wasm_load32_zero ? MVT::i32 : MVT::i64;
    Info.ptrVal = I.getArgOperand(0);
    Info.offset = 0;
    Info.align = Info.memVT == MVT::i32 ? Align(4) : Align(8);
    Info.flags = MachineMemOperand::MOLoad;
    return true;
  default:
    return false;
  }
}

//===----------------------------------------------------------------------===//
// WebAssembly Lowering private implementation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//

static void fail(const SDLoc &DL, SelectionDAG &DAG, const char *Msg) {
  MachineFunction &MF = DAG.getMachineFunction();
  DAG.getContext()->diagnose(
      DiagnosticInfoUnsupported(MF.getFunction(), Msg, DL.getDebugLoc()));
}

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

SDValue
WebAssemblyTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                     SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc DL = CLI.DL;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  MachineFunction &MF = DAG.getMachineFunction();
  auto Layout = MF.getDataLayout();

  CallingConv::ID CallConv = CLI.CallConv;
  if (!callingConvSupported(CallConv))
    fail(DL, DAG,
         "WebAssembly doesn't support language-specific or target-specific "
         "calling conventions yet");
  if (CLI.IsPatchPoint)
    fail(DL, DAG, "WebAssembly doesn't support patch point yet");

  if (CLI.IsTailCall) {
    auto NoTail = [&](const char *Msg) {
      if (CLI.CB && CLI.CB->isMustTailCall())
        fail(DL, DAG, Msg);
      CLI.IsTailCall = false;
    };

    if (!Subtarget->hasTailCall())
      NoTail("WebAssembly 'tail-call' feature not enabled");

    // Varargs calls cannot be tail calls because the buffer is on the stack
    if (CLI.IsVarArg)
      NoTail("WebAssembly does not support varargs tail calls");

    // Do not tail call unless caller and callee return types match
    const Function &F = MF.getFunction();
    const TargetMachine &TM = getTargetMachine();
    Type *RetTy = F.getReturnType();
    SmallVector<MVT, 4> CallerRetTys;
    SmallVector<MVT, 4> CalleeRetTys;
    computeLegalValueVTs(F, TM, RetTy, CallerRetTys);
    computeLegalValueVTs(F, TM, CLI.RetTy, CalleeRetTys);
    bool TypesMatch = CallerRetTys.size() == CalleeRetTys.size() &&
                      std::equal(CallerRetTys.begin(), CallerRetTys.end(),
                                 CalleeRetTys.begin());
    if (!TypesMatch)
      NoTail("WebAssembly tail call requires caller and callee return types to "
             "match");

    // If pointers to local stack values are passed, we cannot tail call
    if (CLI.CB) {
      for (auto &Arg : CLI.CB->args()) {
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
          NoTail(
              "WebAssembly does not support tail calling with stack arguments");
          break;
        }
      }
    }
  }

  SmallVectorImpl<ISD::InputArg> &Ins = CLI.Ins;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals = CLI.OutVals;

  // The generic code may have added an sret argument. If we're lowering an
  // invoke function, the ABI requires that the function pointer be the first
  // argument, so we may have to swap the arguments.
  if (CallConv == CallingConv::WASM_EmscriptenInvoke && Outs.size() >= 2 &&
      Outs[0].Flags.isSRet()) {
    std::swap(Outs[0], Outs[1]);
    std::swap(OutVals[0], OutVals[1]);
  }

  bool HasSwiftSelfArg = false;
  bool HasSwiftErrorArg = false;
  unsigned NumFixedArgs = 0;
  for (unsigned I = 0; I < Outs.size(); ++I) {
    const ISD::OutputArg &Out = Outs[I];
    SDValue &OutVal = OutVals[I];
    HasSwiftSelfArg |= Out.Flags.isSwiftSelf();
    HasSwiftErrorArg |= Out.Flags.isSwiftError();
    if (Out.Flags.isNest())
      fail(DL, DAG, "WebAssembly hasn't implemented nest arguments");
    if (Out.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca arguments");
    if (Out.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs arguments");
    if (Out.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last arguments");
    if (Out.Flags.isByVal() && Out.Flags.getByValSize() != 0) {
      auto &MFI = MF.getFrameInfo();
      int FI = MFI.CreateStackObject(Out.Flags.getByValSize(),
                                     Out.Flags.getNonZeroByValAlign(),
                                     /*isSS=*/false);
      SDValue SizeNode =
          DAG.getConstant(Out.Flags.getByValSize(), DL, MVT::i32);
      SDValue FINode = DAG.getFrameIndex(FI, getPointerTy(Layout));
      Chain = DAG.getMemcpy(
          Chain, DL, FINode, OutVal, SizeNode, Out.Flags.getNonZeroByValAlign(),
          /*isVolatile*/ false, /*AlwaysInline=*/false,
          /*isTailCall*/ false, MachinePointerInfo(), MachinePointerInfo());
      OutVal = FINode;
    }
    // Count the number of fixed args *after* legalization.
    NumFixedArgs += Out.IsFixed;
  }

  bool IsVarArg = CLI.IsVarArg;
  auto PtrVT = getPointerTy(Layout);

  // For swiftcc, emit additional swiftself and swifterror arguments
  // if there aren't. These additional arguments are also added for callee
  // signature They are necessary to match callee and caller signature for
  // indirect call.
  if (CallConv == CallingConv::Swift) {
    if (!HasSwiftSelfArg) {
      NumFixedArgs++;
      ISD::OutputArg Arg;
      Arg.Flags.setSwiftSelf();
      CLI.Outs.push_back(Arg);
      SDValue ArgVal = DAG.getUNDEF(PtrVT);
      CLI.OutVals.push_back(ArgVal);
    }
    if (!HasSwiftErrorArg) {
      NumFixedArgs++;
      ISD::OutputArg Arg;
      Arg.Flags.setSwiftError();
      CLI.Outs.push_back(Arg);
      SDValue ArgVal = DAG.getUNDEF(PtrVT);
      CLI.OutVals.push_back(ArgVal);
    }
  }

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  if (IsVarArg) {
    // Outgoing non-fixed arguments are placed in a buffer. First
    // compute their offsets and the total amount of buffer space needed.
    for (unsigned I = NumFixedArgs; I < Outs.size(); ++I) {
      const ISD::OutputArg &Out = Outs[I];
      SDValue &Arg = OutVals[I];
      EVT VT = Arg.getValueType();
      assert(VT != MVT::iPTR && "Legalized args should be concrete");
      Type *Ty = VT.getTypeForEVT(*DAG.getContext());
      Align Alignment =
          std::max(Out.Flags.getNonZeroOrigAlign(), Layout.getABITypeAlign(Ty));
      unsigned Offset =
          CCInfo.AllocateStack(Layout.getTypeAllocSize(Ty), Alignment);
      CCInfo.addLoc(CCValAssign::getMem(ArgLocs.size(), VT.getSimpleVT(),
                                        Offset, VT.getSimpleVT(),
                                        CCValAssign::Full));
    }
  }

  unsigned NumBytes = CCInfo.getAlignedCallFrameSize();

  SDValue FINode;
  if (IsVarArg && NumBytes) {
    // For non-fixed arguments, next emit stores to store the argument values
    // to the stack buffer at the offsets computed above.
    int FI = MF.getFrameInfo().CreateStackObject(NumBytes,
                                                 Layout.getStackAlignment(),
                                                 /*isSS=*/false);
    unsigned ValNo = 0;
    SmallVector<SDValue, 8> Chains;
    for (SDValue Arg :
         make_range(OutVals.begin() + NumFixedArgs, OutVals.end())) {
      assert(ArgLocs[ValNo].getValNo() == ValNo &&
             "ArgLocs should remain in order and only hold varargs args");
      unsigned Offset = ArgLocs[ValNo++].getLocMemOffset();
      FINode = DAG.getFrameIndex(FI, getPointerTy(Layout));
      SDValue Add = DAG.getNode(ISD::ADD, DL, PtrVT, FINode,
                                DAG.getConstant(Offset, DL, PtrVT));
      Chains.push_back(
          DAG.getStore(Chain, DL, Arg, Add,
                       MachinePointerInfo::getFixedStack(MF, FI, Offset)));
    }
    if (!Chains.empty())
      Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
  } else if (IsVarArg) {
    FINode = DAG.getIntPtrConstant(0, DL);
  }

  if (Callee->getOpcode() == ISD::GlobalAddress) {
    // If the callee is a GlobalAddress node (quite common, every direct call
    // is) turn it into a TargetGlobalAddress node so that LowerGlobalAddress
    // doesn't at MO_GOT which is not needed for direct calls.
    GlobalAddressSDNode* GA = cast<GlobalAddressSDNode>(Callee);
    Callee = DAG.getTargetGlobalAddress(GA->getGlobal(), DL,
                                        getPointerTy(DAG.getDataLayout()),
                                        GA->getOffset());
    Callee = DAG.getNode(WebAssemblyISD::Wrapper, DL,
                         getPointerTy(DAG.getDataLayout()), Callee);
  }

  // Compute the operands for the CALLn node.
  SmallVector<SDValue, 16> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add all fixed arguments. Note that for non-varargs calls, NumFixedArgs
  // isn't reliable.
  Ops.append(OutVals.begin(),
             IsVarArg ? OutVals.begin() + NumFixedArgs : OutVals.end());
  // Add a pointer to the vararg buffer.
  if (IsVarArg)
    Ops.push_back(FINode);

  SmallVector<EVT, 8> InTys;
  for (const auto &In : Ins) {
    assert(!In.Flags.isByVal() && "byval is not valid for return values");
    assert(!In.Flags.isNest() && "nest is not valid for return values");
    if (In.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca return values");
    if (In.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs return values");
    if (In.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG,
           "WebAssembly hasn't implemented cons regs last return values");
    // Ignore In.getNonZeroOrigAlign() because all our arguments are passed in
    // registers.
    InTys.push_back(In.VT);
  }

  if (CLI.IsTailCall) {
    // ret_calls do not return values to the current frame
    SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
    return DAG.getNode(WebAssemblyISD::RET_CALL, DL, NodeTys, Ops);
  }

  InTys.push_back(MVT::Other);
  SDVTList InTyList = DAG.getVTList(InTys);
  SDValue Res = DAG.getNode(WebAssemblyISD::CALL, DL, InTyList, Ops);

  for (size_t I = 0; I < Ins.size(); ++I)
    InVals.push_back(Res.getValue(I));

  // Return the chain
  return Res.getValue(Ins.size());
}

bool WebAssemblyTargetLowering::CanLowerReturn(
    CallingConv::ID /*CallConv*/, MachineFunction & /*MF*/, bool /*IsVarArg*/,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    LLVMContext & /*Context*/) const {
  // WebAssembly can only handle returning tuples with multivalue enabled
  return Subtarget->hasMultivalue() || Outs.size() <= 1;
}

SDValue WebAssemblyTargetLowering::LowerReturn(
    SDValue Chain, CallingConv::ID CallConv, bool /*IsVarArg*/,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
    SelectionDAG &DAG) const {
  assert((Subtarget->hasMultivalue() || Outs.size() <= 1) &&
         "MVP WebAssembly can only return up to one value");
  if (!callingConvSupported(CallConv))
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");

  SmallVector<SDValue, 4> RetOps(1, Chain);
  RetOps.append(OutVals.begin(), OutVals.end());
  Chain = DAG.getNode(WebAssemblyISD::RETURN, DL, MVT::Other, RetOps);

  // Record the number and types of the return values.
  for (const ISD::OutputArg &Out : Outs) {
    assert(!Out.Flags.isByVal() && "byval is not valid for return values");
    assert(!Out.Flags.isNest() && "nest is not valid for return values");
    assert(Out.IsFixed && "non-fixed return value is not valid");
    if (Out.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca results");
    if (Out.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs results");
    if (Out.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last results");
  }

  return Chain;
}

SDValue WebAssemblyTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  if (!callingConvSupported(CallConv))
    fail(DL, DAG, "WebAssembly doesn't support non-C calling conventions");

  MachineFunction &MF = DAG.getMachineFunction();
  auto *MFI = MF.getInfo<WebAssemblyFunctionInfo>();

  // Set up the incoming ARGUMENTS value, which serves to represent the liveness
  // of the incoming values before they're represented by virtual registers.
  MF.getRegInfo().addLiveIn(WebAssembly::ARGUMENTS);

  bool HasSwiftErrorArg = false;
  bool HasSwiftSelfArg = false;
  for (const ISD::InputArg &In : Ins) {
    HasSwiftSelfArg |= In.Flags.isSwiftSelf();
    HasSwiftErrorArg |= In.Flags.isSwiftError();
    if (In.Flags.isInAlloca())
      fail(DL, DAG, "WebAssembly hasn't implemented inalloca arguments");
    if (In.Flags.isNest())
      fail(DL, DAG, "WebAssembly hasn't implemented nest arguments");
    if (In.Flags.isInConsecutiveRegs())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs arguments");
    if (In.Flags.isInConsecutiveRegsLast())
      fail(DL, DAG, "WebAssembly hasn't implemented cons regs last arguments");
    // Ignore In.getNonZeroOrigAlign() because all our arguments are passed in
    // registers.
    InVals.push_back(In.Used ? DAG.getNode(WebAssemblyISD::ARGUMENT, DL, In.VT,
                                           DAG.getTargetConstant(InVals.size(),
                                                                 DL, MVT::i32))
                             : DAG.getUNDEF(In.VT));

    // Record the number and types of arguments.
    MFI->addParam(In.VT);
  }

  // For swiftcc, emit additional swiftself and swifterror arguments
  // if there aren't. These additional arguments are also added for callee
  // signature They are necessary to match callee and caller signature for
  // indirect call.
  auto PtrVT = getPointerTy(MF.getDataLayout());
  if (CallConv == CallingConv::Swift) {
    if (!HasSwiftSelfArg) {
      MFI->addParam(PtrVT);
    }
    if (!HasSwiftErrorArg) {
      MFI->addParam(PtrVT);
    }
  }
  // Varargs are copied into a buffer allocated by the caller, and a pointer to
  // the buffer is passed as an argument.
  if (IsVarArg) {
    MVT PtrVT = getPointerTy(MF.getDataLayout());
    Register VarargVreg =
        MF.getRegInfo().createVirtualRegister(getRegClassFor(PtrVT));
    MFI->setVarargBufferVreg(VarargVreg);
    Chain = DAG.getCopyToReg(
        Chain, DL, VarargVreg,
        DAG.getNode(WebAssemblyISD::ARGUMENT, DL, PtrVT,
                    DAG.getTargetConstant(Ins.size(), DL, MVT::i32)));
    MFI->addParam(PtrVT);
  }

  // Record the number and types of arguments and results.
  SmallVector<MVT, 4> Params;
  SmallVector<MVT, 4> Results;
  computeSignatureVTs(MF.getFunction().getFunctionType(), &MF.getFunction(),
                      MF.getFunction(), DAG.getTarget(), Params, Results);
  for (MVT VT : Results)
    MFI->addResult(VT);
  // TODO: Use signatures in WebAssemblyMachineFunctionInfo too and unify
  // the param logic here with ComputeSignatureVTs
  assert(MFI->getParams().size() == Params.size() &&
         std::equal(MFI->getParams().begin(), MFI->getParams().end(),
                    Params.begin()));

  return Chain;
}

void WebAssemblyTargetLowering::ReplaceNodeResults(
    SDNode *N, SmallVectorImpl<SDValue> &Results, SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  case ISD::SIGN_EXTEND_INREG:
    // Do not add any results, signifying that N should not be custom lowered
    // after all. This happens because simd128 turns on custom lowering for
    // SIGN_EXTEND_INREG, but for non-vector sign extends the result might be an
    // illegal type.
    break;
  default:
    llvm_unreachable(
        "ReplaceNodeResults not implemented for this op for WebAssembly!");
  }
}

//===----------------------------------------------------------------------===//
//  Custom lowering hooks.
//===----------------------------------------------------------------------===//

SDValue WebAssemblyTargetLowering::LowerOperation(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  switch (Op.getOpcode()) {
  default:
    llvm_unreachable("unimplemented operation lowering");
    return SDValue();
  case ISD::FrameIndex:
    return LowerFrameIndex(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::ExternalSymbol:
    return LowerExternalSymbol(Op, DAG);
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);
  case ISD::BR_JT:
    return LowerBR_JT(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::BlockAddress:
  case ISD::BRIND:
    fail(DL, DAG, "WebAssembly hasn't implemented computed gotos");
    return SDValue();
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  case ISD::FRAMEADDR:
    return LowerFRAMEADDR(Op, DAG);
  case ISD::CopyToReg:
    return LowerCopyToReg(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT:
  case ISD::INSERT_VECTOR_ELT:
    return LowerAccessVectorElement(Op, DAG);
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_W_CHAIN:
    return LowerIntrinsic(Op, DAG);
  case ISD::SIGN_EXTEND_INREG:
    return LowerSIGN_EXTEND_INREG(Op, DAG);
  case ISD::BUILD_VECTOR:
    return LowerBUILD_VECTOR(Op, DAG);
  case ISD::VECTOR_SHUFFLE:
    return LowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::SETCC:
    return LowerSETCC(Op, DAG);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:
    return LowerShift(Op, DAG);
  }
}

SDValue WebAssemblyTargetLowering::LowerCopyToReg(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDValue Src = Op.getOperand(2);
  if (isa<FrameIndexSDNode>(Src.getNode())) {
    // CopyToReg nodes don't support FrameIndex operands. Other targets select
    // the FI to some LEA-like instruction, but since we don't have that, we
    // need to insert some kind of instruction that can take an FI operand and
    // produces a value usable by CopyToReg (i.e. in a vreg). So insert a dummy
    // local.copy between Op and its FI operand.
    SDValue Chain = Op.getOperand(0);
    SDLoc DL(Op);
    unsigned Reg = cast<RegisterSDNode>(Op.getOperand(1))->getReg();
    EVT VT = Src.getValueType();
    SDValue Copy(DAG.getMachineNode(VT == MVT::i32 ? WebAssembly::COPY_I32
                                                   : WebAssembly::COPY_I64,
                                    DL, VT, Src),
                 0);
    return Op.getNode()->getNumValues() == 1
               ? DAG.getCopyToReg(Chain, DL, Reg, Copy)
               : DAG.getCopyToReg(Chain, DL, Reg, Copy,
                                  Op.getNumOperands() == 4 ? Op.getOperand(3)
                                                           : SDValue());
  }
  return SDValue();
}

SDValue WebAssemblyTargetLowering::LowerFrameIndex(SDValue Op,
                                                   SelectionDAG &DAG) const {
  int FI = cast<FrameIndexSDNode>(Op)->getIndex();
  return DAG.getTargetFrameIndex(FI, Op.getValueType());
}

SDValue WebAssemblyTargetLowering::LowerRETURNADDR(SDValue Op,
                                                   SelectionDAG &DAG) const {
  SDLoc DL(Op);

  if (!Subtarget->getTargetTriple().isOSEmscripten()) {
    fail(DL, DAG,
         "Non-Emscripten WebAssembly hasn't implemented "
         "__builtin_return_address");
    return SDValue();
  }

  if (verifyReturnAddressArgumentIsConstant(Op, DAG))
    return SDValue();

  unsigned Depth = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
  MakeLibCallOptions CallOptions;
  return makeLibCall(DAG, RTLIB::RETURN_ADDRESS, Op.getValueType(),
                     {DAG.getConstant(Depth, DL, MVT::i32)}, CallOptions, DL)
      .first;
}

SDValue WebAssemblyTargetLowering::LowerFRAMEADDR(SDValue Op,
                                                  SelectionDAG &DAG) const {
  // Non-zero depths are not supported by WebAssembly currently. Use the
  // legalizer's default expansion, which is to return 0 (what this function is
  // documented to do).
  if (Op.getConstantOperandVal(0) > 0)
    return SDValue();

  DAG.getMachineFunction().getFrameInfo().setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  Register FP =
      Subtarget->getRegisterInfo()->getFrameRegister(DAG.getMachineFunction());
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), FP, VT);
}

SDValue WebAssemblyTargetLowering::LowerGlobalAddress(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const auto *GA = cast<GlobalAddressSDNode>(Op);
  EVT VT = Op.getValueType();
  assert(GA->getTargetFlags() == 0 &&
         "Unexpected target flags on generic GlobalAddressSDNode");
  if (GA->getAddressSpace() != 0)
    fail(DL, DAG, "WebAssembly only expects the 0 address space");

  unsigned OperandFlags = 0;
  if (isPositionIndependent()) {
    const GlobalValue *GV = GA->getGlobal();
    if (getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV)) {
      MachineFunction &MF = DAG.getMachineFunction();
      MVT PtrVT = getPointerTy(MF.getDataLayout());
      const char *BaseName;
      if (GV->getValueType()->isFunctionTy()) {
        BaseName = MF.createExternalSymbolName("__table_base");
        OperandFlags = WebAssemblyII::MO_TABLE_BASE_REL;
      }
      else {
        BaseName = MF.createExternalSymbolName("__memory_base");
        OperandFlags = WebAssemblyII::MO_MEMORY_BASE_REL;
      }
      SDValue BaseAddr =
          DAG.getNode(WebAssemblyISD::Wrapper, DL, PtrVT,
                      DAG.getTargetExternalSymbol(BaseName, PtrVT));

      SDValue SymAddr = DAG.getNode(
          WebAssemblyISD::WrapperPIC, DL, VT,
          DAG.getTargetGlobalAddress(GA->getGlobal(), DL, VT, GA->getOffset(),
                                     OperandFlags));

      return DAG.getNode(ISD::ADD, DL, VT, BaseAddr, SymAddr);
    } else {
      OperandFlags = WebAssemblyII::MO_GOT;
    }
  }

  return DAG.getNode(WebAssemblyISD::Wrapper, DL, VT,
                     DAG.getTargetGlobalAddress(GA->getGlobal(), DL, VT,
                                                GA->getOffset(), OperandFlags));
}

SDValue
WebAssemblyTargetLowering::LowerExternalSymbol(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const auto *ES = cast<ExternalSymbolSDNode>(Op);
  EVT VT = Op.getValueType();
  assert(ES->getTargetFlags() == 0 &&
         "Unexpected target flags on generic ExternalSymbolSDNode");
  return DAG.getNode(WebAssemblyISD::Wrapper, DL, VT,
                     DAG.getTargetExternalSymbol(ES->getSymbol(), VT));
}

SDValue WebAssemblyTargetLowering::LowerJumpTable(SDValue Op,
                                                  SelectionDAG &DAG) const {
  // There's no need for a Wrapper node because we always incorporate a jump
  // table operand into a BR_TABLE instruction, rather than ever
  // materializing it in a register.
  const JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  return DAG.getTargetJumpTable(JT->getIndex(), Op.getValueType(),
                                JT->getTargetFlags());
}

SDValue WebAssemblyTargetLowering::LowerBR_JT(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  const auto *JT = cast<JumpTableSDNode>(Op.getOperand(1));
  SDValue Index = Op.getOperand(2);
  assert(JT->getTargetFlags() == 0 && "WebAssembly doesn't set target flags");

  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Index);

  MachineJumpTableInfo *MJTI = DAG.getMachineFunction().getJumpTableInfo();
  const auto &MBBs = MJTI->getJumpTables()[JT->getIndex()].MBBs;

  // Add an operand for each case.
  for (auto MBB : MBBs)
    Ops.push_back(DAG.getBasicBlock(MBB));

  // Add the first MBB as a dummy default target for now. This will be replaced
  // with the proper default target (and the preceding range check eliminated)
  // if possible by WebAssemblyFixBrTableDefaults.
  Ops.push_back(DAG.getBasicBlock(*MBBs.begin()));
  return DAG.getNode(WebAssemblyISD::BR_TABLE, DL, MVT::Other, Ops);
}

SDValue WebAssemblyTargetLowering::LowerVASTART(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT PtrVT = getPointerTy(DAG.getMachineFunction().getDataLayout());

  auto *MFI = DAG.getMachineFunction().getInfo<WebAssemblyFunctionInfo>();
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  SDValue ArgN = DAG.getCopyFromReg(DAG.getEntryNode(), DL,
                                    MFI->getVarargBufferVreg(), PtrVT);
  return DAG.getStore(Op.getOperand(0), DL, ArgN, Op.getOperand(1),
                      MachinePointerInfo(SV));
}

SDValue WebAssemblyTargetLowering::LowerIntrinsic(SDValue Op,
                                                  SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  unsigned IntNo;
  switch (Op.getOpcode()) {
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN:
    IntNo = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
    break;
  case ISD::INTRINSIC_WO_CHAIN:
    IntNo = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    break;
  default:
    llvm_unreachable("Invalid intrinsic");
  }
  SDLoc DL(Op);

  switch (IntNo) {
  default:
    return SDValue(); // Don't custom lower most intrinsics.

  case Intrinsic::wasm_lsda: {
    EVT VT = Op.getValueType();
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    MVT PtrVT = TLI.getPointerTy(DAG.getDataLayout());
    auto &Context = MF.getMMI().getContext();
    MCSymbol *S = Context.getOrCreateSymbol(Twine("GCC_except_table") +
                                            Twine(MF.getFunctionNumber()));
    return DAG.getNode(WebAssemblyISD::Wrapper, DL, VT,
                       DAG.getMCSymbol(S, PtrVT));
  }

  case Intrinsic::wasm_throw: {
    // We only support C++ exceptions for now
    int Tag = cast<ConstantSDNode>(Op.getOperand(2).getNode())->getZExtValue();
    if (Tag != CPP_EXCEPTION)
      llvm_unreachable("Invalid tag!");
    const TargetLowering &TLI = DAG.getTargetLoweringInfo();
    MVT PtrVT = TLI.getPointerTy(DAG.getDataLayout());
    const char *SymName = MF.createExternalSymbolName("__cpp_exception");
    SDValue SymNode = DAG.getNode(WebAssemblyISD::Wrapper, DL, PtrVT,
                                  DAG.getTargetExternalSymbol(SymName, PtrVT));
    return DAG.getNode(WebAssemblyISD::THROW, DL,
                       MVT::Other, // outchain type
                       {
                           Op.getOperand(0), // inchain
                           SymNode,          // exception symbol
                           Op.getOperand(3)  // thrown value
                       });
  }

  case Intrinsic::wasm_shuffle: {
    // Drop in-chain and replace undefs, but otherwise pass through unchanged
    SDValue Ops[18];
    size_t OpIdx = 0;
    Ops[OpIdx++] = Op.getOperand(1);
    Ops[OpIdx++] = Op.getOperand(2);
    while (OpIdx < 18) {
      const SDValue &MaskIdx = Op.getOperand(OpIdx + 1);
      if (MaskIdx.isUndef() ||
          cast<ConstantSDNode>(MaskIdx.getNode())->getZExtValue() >= 32) {
        Ops[OpIdx++] = DAG.getConstant(0, DL, MVT::i32);
      } else {
        Ops[OpIdx++] = MaskIdx;
      }
    }
    return DAG.getNode(WebAssemblyISD::SHUFFLE, DL, Op.getValueType(), Ops);
  }
  }
}

SDValue
WebAssemblyTargetLowering::LowerSIGN_EXTEND_INREG(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  // If sign extension operations are disabled, allow sext_inreg only if operand
  // is a vector extract of an i8 or i16 lane. SIMD does not depend on sign
  // extension operations, but allowing sext_inreg in this context lets us have
  // simple patterns to select extract_lane_s instructions. Expanding sext_inreg
  // everywhere would be simpler in this file, but would necessitate large and
  // brittle patterns to undo the expansion and select extract_lane_s
  // instructions.
  assert(!Subtarget->hasSignExt() && Subtarget->hasSIMD128());
  if (Op.getOperand(0).getOpcode() != ISD::EXTRACT_VECTOR_ELT)
    return SDValue();

  const SDValue &Extract = Op.getOperand(0);
  MVT VecT = Extract.getOperand(0).getSimpleValueType();
  if (VecT.getVectorElementType().getSizeInBits() > 32)
    return SDValue();
  MVT ExtractedLaneT =
      cast<VTSDNode>(Op.getOperand(1).getNode())->getVT().getSimpleVT();
  MVT ExtractedVecT =
      MVT::getVectorVT(ExtractedLaneT, 128 / ExtractedLaneT.getSizeInBits());
  if (ExtractedVecT == VecT)
    return Op;

  // Bitcast vector to appropriate type to ensure ISel pattern coverage
  const SDNode *Index = Extract.getOperand(1).getNode();
  if (!isa<ConstantSDNode>(Index))
    return SDValue();
  unsigned IndexVal = cast<ConstantSDNode>(Index)->getZExtValue();
  unsigned Scale =
      ExtractedVecT.getVectorNumElements() / VecT.getVectorNumElements();
  assert(Scale > 1);
  SDValue NewIndex =
      DAG.getConstant(IndexVal * Scale, DL, Index->getValueType(0));
  SDValue NewExtract = DAG.getNode(
      ISD::EXTRACT_VECTOR_ELT, DL, Extract.getValueType(),
      DAG.getBitcast(ExtractedVecT, Extract.getOperand(0)), NewIndex);
  return DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, Op.getValueType(), NewExtract,
                     Op.getOperand(1));
}

SDValue WebAssemblyTargetLowering::LowerBUILD_VECTOR(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc DL(Op);
  const EVT VecT = Op.getValueType();
  const EVT LaneT = Op.getOperand(0).getValueType();
  const size_t Lanes = Op.getNumOperands();
  bool CanSwizzle = VecT == MVT::v16i8;

  // BUILD_VECTORs are lowered to the instruction that initializes the highest
  // possible number of lanes at once followed by a sequence of replace_lane
  // instructions to individually initialize any remaining lanes.

  // TODO: Tune this. For example, lanewise swizzling is very expensive, so
  // swizzled lanes should be given greater weight.

  // TODO: Investigate building vectors by shuffling together vectors built by
  // separately specialized means.

  auto IsConstant = [](const SDValue &V) {
    return V.getOpcode() == ISD::Constant || V.getOpcode() == ISD::ConstantFP;
  };

  // Returns the source vector and index vector pair if they exist. Checks for:
  //   (extract_vector_elt
  //     $src,
  //     (sign_extend_inreg (extract_vector_elt $indices, $i))
  //   )
  auto GetSwizzleSrcs = [](size_t I, const SDValue &Lane) {
    auto Bail = std::make_pair(SDValue(), SDValue());
    if (Lane->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return Bail;
    const SDValue &SwizzleSrc = Lane->getOperand(0);
    const SDValue &IndexExt = Lane->getOperand(1);
    if (IndexExt->getOpcode() != ISD::SIGN_EXTEND_INREG)
      return Bail;
    const SDValue &Index = IndexExt->getOperand(0);
    if (Index->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return Bail;
    const SDValue &SwizzleIndices = Index->getOperand(0);
    if (SwizzleSrc.getValueType() != MVT::v16i8 ||
        SwizzleIndices.getValueType() != MVT::v16i8 ||
        Index->getOperand(1)->getOpcode() != ISD::Constant ||
        Index->getConstantOperandVal(1) != I)
      return Bail;
    return std::make_pair(SwizzleSrc, SwizzleIndices);
  };

  using ValueEntry = std::pair<SDValue, size_t>;
  SmallVector<ValueEntry, 16> SplatValueCounts;

  using SwizzleEntry = std::pair<std::pair<SDValue, SDValue>, size_t>;
  SmallVector<SwizzleEntry, 16> SwizzleCounts;

  auto AddCount = [](auto &Counts, const auto &Val) {
    auto CountIt = std::find_if(Counts.begin(), Counts.end(),
                                [&Val](auto E) { return E.first == Val; });
    if (CountIt == Counts.end()) {
      Counts.emplace_back(Val, 1);
    } else {
      CountIt->second++;
    }
  };

  auto GetMostCommon = [](auto &Counts) {
    auto CommonIt =
        std::max_element(Counts.begin(), Counts.end(),
                         [](auto A, auto B) { return A.second < B.second; });
    assert(CommonIt != Counts.end() && "Unexpected all-undef build_vector");
    return *CommonIt;
  };

  size_t NumConstantLanes = 0;

  // Count eligible lanes for each type of vector creation op
  for (size_t I = 0; I < Lanes; ++I) {
    const SDValue &Lane = Op->getOperand(I);
    if (Lane.isUndef())
      continue;

    AddCount(SplatValueCounts, Lane);

    if (IsConstant(Lane)) {
      NumConstantLanes++;
    } else if (CanSwizzle) {
      auto SwizzleSrcs = GetSwizzleSrcs(I, Lane);
      if (SwizzleSrcs.first)
        AddCount(SwizzleCounts, SwizzleSrcs);
    }
  }

  SDValue SplatValue;
  size_t NumSplatLanes;
  std::tie(SplatValue, NumSplatLanes) = GetMostCommon(SplatValueCounts);

  SDValue SwizzleSrc;
  SDValue SwizzleIndices;
  size_t NumSwizzleLanes = 0;
  if (SwizzleCounts.size())
    std::forward_as_tuple(std::tie(SwizzleSrc, SwizzleIndices),
                          NumSwizzleLanes) = GetMostCommon(SwizzleCounts);

  // Predicate returning true if the lane is properly initialized by the
  // original instruction
  std::function<bool(size_t, const SDValue &)> IsLaneConstructed;
  SDValue Result;
  // Prefer swizzles over vector consts over splats
  if (NumSwizzleLanes >= NumSplatLanes &&
      (!Subtarget->hasUnimplementedSIMD128() ||
       NumSwizzleLanes >= NumConstantLanes)) {
    Result = DAG.getNode(WebAssemblyISD::SWIZZLE, DL, VecT, SwizzleSrc,
                         SwizzleIndices);
    auto Swizzled = std::make_pair(SwizzleSrc, SwizzleIndices);
    IsLaneConstructed = [&, Swizzled](size_t I, const SDValue &Lane) {
      return Swizzled == GetSwizzleSrcs(I, Lane);
    };
  } else if (NumConstantLanes >= NumSplatLanes &&
             Subtarget->hasUnimplementedSIMD128()) {
    SmallVector<SDValue, 16> ConstLanes;
    for (const SDValue &Lane : Op->op_values()) {
      if (IsConstant(Lane)) {
        ConstLanes.push_back(Lane);
      } else if (LaneT.isFloatingPoint()) {
        ConstLanes.push_back(DAG.getConstantFP(0, DL, LaneT));
      } else {
        ConstLanes.push_back(DAG.getConstant(0, DL, LaneT));
      }
    }
    Result = DAG.getBuildVector(VecT, DL, ConstLanes);
    IsLaneConstructed = [&](size_t _, const SDValue &Lane) {
      return IsConstant(Lane);
    };
  }
  if (!Result) {
    // Use a splat, but possibly a load_splat
    LoadSDNode *SplattedLoad;
    if ((SplattedLoad = dyn_cast<LoadSDNode>(SplatValue)) &&
        SplattedLoad->getMemoryVT() == VecT.getVectorElementType()) {
      Result = DAG.getMemIntrinsicNode(
          WebAssemblyISD::LOAD_SPLAT, DL, DAG.getVTList(VecT),
          {SplattedLoad->getChain(), SplattedLoad->getBasePtr(),
           SplattedLoad->getOffset()},
          SplattedLoad->getMemoryVT(), SplattedLoad->getMemOperand());
    } else {
      Result = DAG.getSplatBuildVector(VecT, DL, SplatValue);
    }
    IsLaneConstructed = [&](size_t _, const SDValue &Lane) {
      return Lane == SplatValue;
    };
  }

  // Add replace_lane instructions for any unhandled values
  for (size_t I = 0; I < Lanes; ++I) {
    const SDValue &Lane = Op->getOperand(I);
    if (!Lane.isUndef() && !IsLaneConstructed(I, Lane))
      Result = DAG.getNode(ISD::INSERT_VECTOR_ELT, DL, VecT, Result, Lane,
                           DAG.getConstant(I, DL, MVT::i32));
  }

  return Result;
}

SDValue
WebAssemblyTargetLowering::LowerVECTOR_SHUFFLE(SDValue Op,
                                               SelectionDAG &DAG) const {
  SDLoc DL(Op);
  ArrayRef<int> Mask = cast<ShuffleVectorSDNode>(Op.getNode())->getMask();
  MVT VecType = Op.getOperand(0).getSimpleValueType();
  assert(VecType.is128BitVector() && "Unexpected shuffle vector type");
  size_t LaneBytes = VecType.getVectorElementType().getSizeInBits() / 8;

  // Space for two vector args and sixteen mask indices
  SDValue Ops[18];
  size_t OpIdx = 0;
  Ops[OpIdx++] = Op.getOperand(0);
  Ops[OpIdx++] = Op.getOperand(1);

  // Expand mask indices to byte indices and materialize them as operands
  for (int M : Mask) {
    for (size_t J = 0; J < LaneBytes; ++J) {
      // Lower undefs (represented by -1 in mask) to zero
      uint64_t ByteIndex = M == -1 ? 0 : (uint64_t)M * LaneBytes + J;
      Ops[OpIdx++] = DAG.getConstant(ByteIndex, DL, MVT::i32);
    }
  }

  return DAG.getNode(WebAssemblyISD::SHUFFLE, DL, Op.getValueType(), Ops);
}

SDValue WebAssemblyTargetLowering::LowerSETCC(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  // The legalizer does not know how to expand the comparison modes of i64x2
  // vectors because no comparison modes are supported. We could solve this by
  // expanding all i64x2 SETCC nodes, but that seems to expand f64x2 SETCC nodes
  // (which return i64x2 results) as well. So instead we manually unroll i64x2
  // comparisons here.
  assert(Op->getOperand(0)->getSimpleValueType(0) == MVT::v2i64);
  SmallVector<SDValue, 2> LHS, RHS;
  DAG.ExtractVectorElements(Op->getOperand(0), LHS);
  DAG.ExtractVectorElements(Op->getOperand(1), RHS);
  const SDValue &CC = Op->getOperand(2);
  auto MakeLane = [&](unsigned I) {
    return DAG.getNode(ISD::SELECT_CC, DL, MVT::i64, LHS[I], RHS[I],
                       DAG.getConstant(uint64_t(-1), DL, MVT::i64),
                       DAG.getConstant(uint64_t(0), DL, MVT::i64), CC);
  };
  return DAG.getBuildVector(Op->getValueType(0), DL,
                            {MakeLane(0), MakeLane(1)});
}

SDValue
WebAssemblyTargetLowering::LowerAccessVectorElement(SDValue Op,
                                                    SelectionDAG &DAG) const {
  // Allow constant lane indices, expand variable lane indices
  SDNode *IdxNode = Op.getOperand(Op.getNumOperands() - 1).getNode();
  if (isa<ConstantSDNode>(IdxNode) || IdxNode->isUndef())
    return Op;
  else
    // Perform default expansion
    return SDValue();
}

static SDValue unrollVectorShift(SDValue Op, SelectionDAG &DAG) {
  EVT LaneT = Op.getSimpleValueType().getVectorElementType();
  // 32-bit and 64-bit unrolled shifts will have proper semantics
  if (LaneT.bitsGE(MVT::i32))
    return DAG.UnrollVectorOp(Op.getNode());
  // Otherwise mask the shift value to get proper semantics from 32-bit shift
  SDLoc DL(Op);
  size_t NumLanes = Op.getSimpleValueType().getVectorNumElements();
  SDValue Mask = DAG.getConstant(LaneT.getSizeInBits() - 1, DL, MVT::i32);
  unsigned ShiftOpcode = Op.getOpcode();
  SmallVector<SDValue, 16> ShiftedElements;
  DAG.ExtractVectorElements(Op.getOperand(0), ShiftedElements, 0, 0, MVT::i32);
  SmallVector<SDValue, 16> ShiftElements;
  DAG.ExtractVectorElements(Op.getOperand(1), ShiftElements, 0, 0, MVT::i32);
  SmallVector<SDValue, 16> UnrolledOps;
  for (size_t i = 0; i < NumLanes; ++i) {
    SDValue MaskedShiftValue =
        DAG.getNode(ISD::AND, DL, MVT::i32, ShiftElements[i], Mask);
    SDValue ShiftedValue = ShiftedElements[i];
    if (ShiftOpcode == ISD::SRA)
      ShiftedValue = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, MVT::i32,
                                 ShiftedValue, DAG.getValueType(LaneT));
    UnrolledOps.push_back(
        DAG.getNode(ShiftOpcode, DL, MVT::i32, ShiftedValue, MaskedShiftValue));
  }
  return DAG.getBuildVector(Op.getValueType(), DL, UnrolledOps);
}

SDValue WebAssemblyTargetLowering::LowerShift(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);

  // Only manually lower vector shifts
  assert(Op.getSimpleValueType().isVector());

  auto ShiftVal = DAG.getSplatValue(Op.getOperand(1));
  if (!ShiftVal)
    return unrollVectorShift(Op, DAG);

  // Use anyext because none of the high bits can affect the shift
  ShiftVal = DAG.getAnyExtOrTrunc(ShiftVal, DL, MVT::i32);

  unsigned Opcode;
  switch (Op.getOpcode()) {
  case ISD::SHL:
    Opcode = WebAssemblyISD::VEC_SHL;
    break;
  case ISD::SRA:
    Opcode = WebAssemblyISD::VEC_SHR_S;
    break;
  case ISD::SRL:
    Opcode = WebAssemblyISD::VEC_SHR_U;
    break;
  default:
    llvm_unreachable("unexpected opcode");
  }

  return DAG.getNode(Opcode, DL, Op.getValueType(), Op.getOperand(0), ShiftVal);
}

//===----------------------------------------------------------------------===//
//   Custom DAG combine hooks
//===----------------------------------------------------------------------===//
static SDValue
performVECTOR_SHUFFLECombine(SDNode *N, TargetLowering::DAGCombinerInfo &DCI) {
  auto &DAG = DCI.DAG;
  auto Shuffle = cast<ShuffleVectorSDNode>(N);

  // Hoist vector bitcasts that don't change the number of lanes out of unary
  // shuffles, where they are less likely to get in the way of other combines.
  // (shuffle (vNxT1 (bitcast (vNxT0 x))), undef, mask) ->
  //  (vNxT1 (bitcast (vNxT0 (shuffle x, undef, mask))))
  SDValue Bitcast = N->getOperand(0);
  if (Bitcast.getOpcode() != ISD::BITCAST)
    return SDValue();
  if (!N->getOperand(1).isUndef())
    return SDValue();
  SDValue CastOp = Bitcast.getOperand(0);
  MVT SrcType = CastOp.getSimpleValueType();
  MVT DstType = Bitcast.getSimpleValueType();
  if (!SrcType.is128BitVector() ||
      SrcType.getVectorNumElements() != DstType.getVectorNumElements())
    return SDValue();
  SDValue NewShuffle = DAG.getVectorShuffle(
      SrcType, SDLoc(N), CastOp, DAG.getUNDEF(SrcType), Shuffle->getMask());
  return DAG.getBitcast(DstType, NewShuffle);
}

static SDValue performVectorWidenCombine(SDNode *N,
                                         TargetLowering::DAGCombinerInfo &DCI) {
  auto &DAG = DCI.DAG;
  assert(N->getOpcode() == ISD::SIGN_EXTEND ||
         N->getOpcode() == ISD::ZERO_EXTEND);

  // Combine ({s,z}ext (extract_subvector src, i)) into a widening operation if
  // possible before the extract_subvector can be expanded.
  auto Extract = N->getOperand(0);
  if (Extract.getOpcode() != ISD::EXTRACT_SUBVECTOR)
    return SDValue();
  auto Source = Extract.getOperand(0);
  auto *IndexNode = dyn_cast<ConstantSDNode>(Extract.getOperand(1));
  if (IndexNode == nullptr)
    return SDValue();
  auto Index = IndexNode->getZExtValue();

  // Only v8i8 and v4i16 extracts can be widened, and only if the extracted
  // subvector is the low or high half of its source.
  EVT ResVT = N->getValueType(0);
  if (ResVT == MVT::v8i16) {
    if (Extract.getValueType() != MVT::v8i8 ||
        Source.getValueType() != MVT::v16i8 || (Index != 0 && Index != 8))
      return SDValue();
  } else if (ResVT == MVT::v4i32) {
    if (Extract.getValueType() != MVT::v4i16 ||
        Source.getValueType() != MVT::v8i16 || (Index != 0 && Index != 4))
      return SDValue();
  } else {
    return SDValue();
  }

  bool IsSext = N->getOpcode() == ISD::SIGN_EXTEND;
  bool IsLow = Index == 0;

  unsigned Op = IsSext ? (IsLow ? WebAssemblyISD::WIDEN_LOW_S
                                : WebAssemblyISD::WIDEN_HIGH_S)
                       : (IsLow ? WebAssemblyISD::WIDEN_LOW_U
                                : WebAssemblyISD::WIDEN_HIGH_U);

  return DAG.getNode(Op, SDLoc(N), ResVT, Source);
}

SDValue
WebAssemblyTargetLowering::PerformDAGCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default:
    return SDValue();
  case ISD::VECTOR_SHUFFLE:
    return performVECTOR_SHUFFLECombine(N, DCI);
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
    return performVectorWidenCombine(N, DCI);
  }
}
