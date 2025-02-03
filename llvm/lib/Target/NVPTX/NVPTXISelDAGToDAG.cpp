//===-- NVPTXISelDAGToDAG.cpp - A dag to dag inst selector for NVPTX ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the NVPTX target.
//
//===----------------------------------------------------------------------===//

#include "NVPTXISelDAGToDAG.h"
#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Target/TargetIntrinsicInfo.h"

using namespace llvm;

#define DEBUG_TYPE "nvptx-isel"
#define PASS_NAME "NVPTX DAG->DAG Pattern Instruction Selection"

static cl::opt<bool>
    EnableRsqrtOpt("nvptx-rsqrt-approx-opt", cl::init(true), cl::Hidden,
                   cl::desc("Enable reciprocal sqrt optimization"));

/// createNVPTXISelDag - This pass converts a legalized DAG into a
/// NVPTX-specific DAG, ready for instruction scheduling.
FunctionPass *llvm::createNVPTXISelDag(NVPTXTargetMachine &TM,
                                       llvm::CodeGenOptLevel OptLevel) {
  return new NVPTXDAGToDAGISelLegacy(TM, OptLevel);
}

NVPTXDAGToDAGISelLegacy::NVPTXDAGToDAGISelLegacy(NVPTXTargetMachine &tm,
                                                 CodeGenOptLevel OptLevel)
    : SelectionDAGISelLegacy(
          ID, std::make_unique<NVPTXDAGToDAGISel>(tm, OptLevel)) {}

char NVPTXDAGToDAGISelLegacy::ID = 0;

INITIALIZE_PASS(NVPTXDAGToDAGISelLegacy, DEBUG_TYPE, PASS_NAME, false, false)

NVPTXDAGToDAGISel::NVPTXDAGToDAGISel(NVPTXTargetMachine &tm,
                                     CodeGenOptLevel OptLevel)
    : SelectionDAGISel(tm, OptLevel), TM(tm) {
  doMulWide = (OptLevel > CodeGenOptLevel::None);
}

bool NVPTXDAGToDAGISel::runOnMachineFunction(MachineFunction &MF) {
  Subtarget = &MF.getSubtarget<NVPTXSubtarget>();
  Scopes = NVPTXScopes(MF.getFunction().getContext());
  return SelectionDAGISel::runOnMachineFunction(MF);
}

int NVPTXDAGToDAGISel::getDivF32Level() const {
  return Subtarget->getTargetLowering()->getDivF32Level();
}

bool NVPTXDAGToDAGISel::usePrecSqrtF32() const {
  return Subtarget->getTargetLowering()->usePrecSqrtF32();
}

bool NVPTXDAGToDAGISel::useF32FTZ() const {
  return Subtarget->getTargetLowering()->useF32FTZ(*MF);
}

bool NVPTXDAGToDAGISel::allowFMA() const {
  const NVPTXTargetLowering *TL = Subtarget->getTargetLowering();
  return TL->allowFMA(*MF, OptLevel);
}

bool NVPTXDAGToDAGISel::allowUnsafeFPMath() const {
  const NVPTXTargetLowering *TL = Subtarget->getTargetLowering();
  return TL->allowUnsafeFPMath(*MF);
}

bool NVPTXDAGToDAGISel::doRsqrtOpt() const { return EnableRsqrtOpt; }

/// Select - Select instructions not customized! Used for
/// expanded, promoted and normal instructions.
void NVPTXDAGToDAGISel::Select(SDNode *N) {

  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return; // Already selected.
  }

  switch (N->getOpcode()) {
  case ISD::LOAD:
  case ISD::ATOMIC_LOAD:
    if (tryLoad(N))
      return;
    break;
  case ISD::STORE:
  case ISD::ATOMIC_STORE:
    if (tryStore(N))
      return;
    break;
  case ISD::ATOMIC_FENCE:
    if (tryFence(N))
      return;
    break;
  case ISD::EXTRACT_VECTOR_ELT:
    if (tryEXTRACT_VECTOR_ELEMENT(N))
      return;
    break;
  case NVPTXISD::SETP_F16X2:
    SelectSETP_F16X2(N);
    return;
  case NVPTXISD::SETP_BF16X2:
    SelectSETP_BF16X2(N);
    return;
  case NVPTXISD::FADD_F32X2:
  case NVPTXISD::FSUB_F32X2:
  case NVPTXISD::FMUL_F32X2:
  case NVPTXISD::FMA_F32X2:
    SelectF32X2Op(N);
    return;
  case NVPTXISD::LoadV2:
  case NVPTXISD::LoadV4:
    if (tryLoadVector(N))
      return;
    break;
  case NVPTXISD::LDUV2:
  case NVPTXISD::LDUV4:
    if (tryLDGLDU(N))
      return;
    break;
  case NVPTXISD::StoreV2:
  case NVPTXISD::StoreV4:
    if (tryStoreVector(N))
      return;
    break;
  case NVPTXISD::LoadParam:
  case NVPTXISD::LoadParamV2:
  case NVPTXISD::LoadParamV4:
    if (tryLoadParam(N))
      return;
    break;
  case NVPTXISD::StoreRetval:
  case NVPTXISD::StoreRetvalV2:
  case NVPTXISD::StoreRetvalV4:
    if (tryStoreRetval(N))
      return;
    break;
  case NVPTXISD::StoreParam:
  case NVPTXISD::StoreParamV2:
  case NVPTXISD::StoreParamV4:
  case NVPTXISD::StoreParamS32:
  case NVPTXISD::StoreParamU32:
    if (tryStoreParam(N))
      return;
    break;
  case ISD::INTRINSIC_WO_CHAIN:
    if (tryIntrinsicNoChain(N))
      return;
    break;
  case ISD::INTRINSIC_W_CHAIN:
    if (tryIntrinsicChain(N))
      return;
    break;
  case ISD::INTRINSIC_VOID:
    if (tryIntrinsicVoid(N))
      return;
    break;
  case ISD::AND:
  case ISD::SRA:
  case ISD::SRL:
    // Try to select BFE
    if (tryBFE(N))
      return;
    break;
  case ISD::ADDRSPACECAST:
    SelectAddrSpaceCast(N);
    return;
  case ISD::CopyToReg: {
    if (N->getOperand(1).getValueType() == MVT::i128) {
      SelectV2I64toI128(N);
      return;
    }
    break;
  }
  case ISD::CopyFromReg: {
    if (N->getOperand(1).getValueType() == MVT::i128) {
      SelectI128toV2I64(N);
      return;
    }
    if (N->getOperand(1).getValueType() == MVT::i64 && N->getNumValues() == 3) {
      SelectI64ToV2I32(N);
      return;
    }
    break;
  }
  case ISD::FADD:
  case ISD::FMUL:
  case ISD::FSUB:
    if (tryBF16ArithToFMA(N))
      return;
    break;
  default:
    break;
  }
  SelectCode(N);
}

bool NVPTXDAGToDAGISel::tryIntrinsicChain(SDNode *N) {
  unsigned IID = N->getConstantOperandVal(1);
  switch (IID) {
  default:
    return false;
  case Intrinsic::nvvm_ldu_global_f:
  case Intrinsic::nvvm_ldu_global_i:
  case Intrinsic::nvvm_ldu_global_p:
    return tryLDGLDU(N);
  }
}

// Map ISD:CONDCODE value to appropriate CmpMode expected by
// NVPTXInstPrinter::printCmpMode()
static unsigned getPTXCmpMode(const CondCodeSDNode &CondCode, bool FTZ) {
  using NVPTX::PTXCmpMode::CmpMode;
  unsigned PTXCmpMode = [](ISD::CondCode CC) {
    switch (CC) {
    default:
      llvm_unreachable("Unexpected condition code.");
    case ISD::SETOEQ:
      return CmpMode::EQ;
    case ISD::SETOGT:
      return CmpMode::GT;
    case ISD::SETOGE:
      return CmpMode::GE;
    case ISD::SETOLT:
      return CmpMode::LT;
    case ISD::SETOLE:
      return CmpMode::LE;
    case ISD::SETONE:
      return CmpMode::NE;
    case ISD::SETO:
      return CmpMode::NUM;
    case ISD::SETUO:
      return CmpMode::NotANumber;
    case ISD::SETUEQ:
      return CmpMode::EQU;
    case ISD::SETUGT:
      return CmpMode::GTU;
    case ISD::SETUGE:
      return CmpMode::GEU;
    case ISD::SETULT:
      return CmpMode::LTU;
    case ISD::SETULE:
      return CmpMode::LEU;
    case ISD::SETUNE:
      return CmpMode::NEU;
    case ISD::SETEQ:
      return CmpMode::EQ;
    case ISD::SETGT:
      return CmpMode::GT;
    case ISD::SETGE:
      return CmpMode::GE;
    case ISD::SETLT:
      return CmpMode::LT;
    case ISD::SETLE:
      return CmpMode::LE;
    case ISD::SETNE:
      return CmpMode::NE;
    }
  }(CondCode.get());

  if (FTZ)
    PTXCmpMode |= NVPTX::PTXCmpMode::FTZ_FLAG;

  return PTXCmpMode;
}

bool NVPTXDAGToDAGISel::SelectSETP_F16X2(SDNode *N) {
  unsigned PTXCmpMode =
      getPTXCmpMode(*cast<CondCodeSDNode>(N->getOperand(2)), useF32FTZ());
  SDLoc DL(N);
  SDNode *SetP = CurDAG->getMachineNode(
      NVPTX::SETP_f16x2rr, DL, MVT::i1, MVT::i1, N->getOperand(0),
      N->getOperand(1), CurDAG->getTargetConstant(PTXCmpMode, DL, MVT::i32));
  ReplaceNode(N, SetP);
  return true;
}

bool NVPTXDAGToDAGISel::SelectSETP_BF16X2(SDNode *N) {
  unsigned PTXCmpMode =
      getPTXCmpMode(*cast<CondCodeSDNode>(N->getOperand(2)), useF32FTZ());
  SDLoc DL(N);
  SDNode *SetP = CurDAG->getMachineNode(
      NVPTX::SETP_bf16x2rr, DL, MVT::i1, MVT::i1, N->getOperand(0),
      N->getOperand(1), CurDAG->getTargetConstant(PTXCmpMode, DL, MVT::i32));
  ReplaceNode(N, SetP);
  return true;
}

void NVPTXDAGToDAGISel::SelectF32X2Op(SDNode *N) {
  unsigned Opcode;
  switch (N->getOpcode()) {
  case NVPTXISD::FADD_F32X2:
    Opcode = NVPTX::FADD_F32X2;
    break;
  case NVPTXISD::FSUB_F32X2:
    Opcode = NVPTX::FSUB_F32X2;
    break;
  case NVPTXISD::FMUL_F32X2:
    Opcode = NVPTX::FMUL_F32X2;
    break;
  case NVPTXISD::FMA_F32X2:
    Opcode = NVPTX::FMA_F32X2;
    break;
  default:
    llvm_unreachable("Unexpected opcode!");
  }
  SDLoc DL(N);
  SmallVector<SDValue> NewOps(N->ops());
  SDNode *NewNode = CurDAG->getMachineNode(Opcode, DL, MVT::i64, NewOps);
  ReplaceNode(N, NewNode);
}

// Find all instances of extract_vector_elt that use this v2f16 vector
// and coalesce them into a scattering move instruction.
bool NVPTXDAGToDAGISel::tryEXTRACT_VECTOR_ELEMENT(SDNode *N) {
  SDValue Vector = N->getOperand(0);

  // We only care about 16x2 as it's the only real vector type we
  // need to deal with.
  MVT VT = Vector.getSimpleValueType();
  if (!Isv2x16VT(VT))
    return false;
  // Find and record all uses of this vector that extract element 0 or 1.
  SmallVector<SDNode *, 4> E0, E1;
  for (auto *U : Vector.getNode()->users()) {
    if (U->getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      continue;
    if (U->getOperand(0) != Vector)
      continue;
    if (const ConstantSDNode *IdxConst =
            dyn_cast<ConstantSDNode>(U->getOperand(1))) {
      if (IdxConst->getZExtValue() == 0)
        E0.push_back(U);
      else if (IdxConst->getZExtValue() == 1)
        E1.push_back(U);
      else
        llvm_unreachable("Invalid vector index.");
    }
  }

  // There's no point scattering f16x2 if we only ever access one
  // element of it.
  if (E0.empty() || E1.empty())
    return false;

  // Merge (f16 extractelt(V, 0), f16 extractelt(V,1))
  // into f16,f16 SplitF16x2(V)
  MVT EltVT = VT.getVectorElementType();
  SDNode *ScatterOp =
      CurDAG->getMachineNode(NVPTX::I32toV2I16, SDLoc(N), EltVT, EltVT, Vector);
  for (auto *Node : E0)
    ReplaceUses(SDValue(Node, 0), SDValue(ScatterOp, 0));
  for (auto *Node : E1)
    ReplaceUses(SDValue(Node, 0), SDValue(ScatterOp, 1));

  return true;
}

static unsigned int getCodeAddrSpace(MemSDNode *N) {
  const Value *Src = N->getMemOperand()->getValue();

  if (!Src)
    return NVPTX::AddressSpace::Generic;

  if (auto *PT = dyn_cast<PointerType>(Src->getType())) {
    switch (PT->getAddressSpace()) {
    case llvm::ADDRESS_SPACE_LOCAL:
      return NVPTX::AddressSpace::Local;
    case llvm::ADDRESS_SPACE_GLOBAL:
      return NVPTX::AddressSpace::Global;
    case llvm::ADDRESS_SPACE_SHARED:
      return NVPTX::AddressSpace::Shared;
    case llvm::ADDRESS_SPACE_GENERIC:
      return NVPTX::AddressSpace::Generic;
    case llvm::ADDRESS_SPACE_PARAM:
      return NVPTX::AddressSpace::Param;
    case llvm::ADDRESS_SPACE_CONST:
      return NVPTX::AddressSpace::Const;
    default: break;
    }
  }
  return NVPTX::AddressSpace::Generic;
}

namespace {

struct OperationOrderings {
  NVPTX::Ordering InstructionOrdering, FenceOrdering;
  OperationOrderings(NVPTX::Ordering IO = NVPTX::Ordering::NotAtomic,
                     NVPTX::Ordering FO = NVPTX::Ordering::NotAtomic)
      : InstructionOrdering(IO), FenceOrdering(FO) {}
};

static OperationOrderings
getOperationOrderings(MemSDNode *N, const NVPTXSubtarget *Subtarget) {
  AtomicOrdering Ordering = N->getSuccessOrdering();
  auto CodeAddrSpace = getCodeAddrSpace(N);

  bool HasMemoryOrdering = Subtarget->hasMemoryOrdering();
  bool HasRelaxedMMIO = Subtarget->hasRelaxedMMIO();

  // clang-format off

  // Lowering for Load/Store Operations (note: AcquireRelease Loads or Stores error).
  // Note: uses of Relaxed in the Atomic column of this table refer
  // to LLVM AtomicOrdering::Monotonic.
  //
  // | Atomic  | Volatile | Statespace         | PTX sm_60- | PTX sm_70+                   |
  // |---------|----------|--------------------|------------|------------------------------|
  // | No      | No       | All                | plain      | .weak                        |
  // | No      | Yes      | Generic,Shared,    | .volatile  | .volatile                    |
  // |         |          | Global [0]         |            |                              |
  // | No      | Yes      | Local,Const,Param  | plain [1]  | .weak [1]                    |
  // | Unorder | Yes/No   | All                | == Relaxed | == Relaxed                   |
  // | Relaxed | No       | Generic,Shared,    | .volatile  | <atomic sem>                 |
  // |         |          | Global [0]         |            |                              |
  // | Other   | No       | Generic,Shared,    | Error [2]  | <atomic sem>                 |
  // |         |          | Global [0]         |            |                              |
  // | Yes     | No       | Local,Const,Param  | plain [1]  | .weak [1]                    |
  // | Relaxed | Yes      | Generic,Shared [0] | .volatile  | .volatile                    |
  // | Relaxed | Yes      | Global [0]         | .volatile  | .mmio.relaxed.sys (PTX 8.2+) |
  // |         |          |                    |            |  or .volatile (PTX 8.1-)     |
  // | Relaxed | Yes      | Local,Const,Param  | plain [1]  | .weak [1]                    |
  // | Other   | Yes      | Generic, Shared,   | Error [2]  | <atomic sem> [3]             |
  // |         |          | / Global [0]       |            |                              |

  // Lowering of CUDA C++ SequentiallyConsistent Operations and Fences to PTX
  // by following the ABI proven sound in:
  //   Lustig et al, A Formal Analysis of the NVIDIA PTX Memory Consistency Model, ASPLOS’19.
  //   https://dl.acm.org/doi/pdf/10.1145/3297858.3304043
  //
  // | CUDA C++ Atomic Operation or Atomic Fence            | PTX Atomic Operation or Fence |
  // |------------------------------------------------------|-------------------------------|
  // | cuda::atomic_thread_fence                            | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) |                               |
  // |------------------------------------------------------|-------------------------------|
  // | cuda::atomic_load                                    | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) | ld.acquire.<scope>;           |
  // |------------------------------------------------------|-------------------------------|  
  // | cuda::atomic_store                                   | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) | st.release.<scope>;           |
  // |------------------------------------------------------|-------------------------------|
  // | cuda::atomic_fetch_<op>                              | fence.sc.<scope>;             |
  // |   (memory_order_seq_cst, cuda::thread_scope_<scope>) | atom.acq_rel.<scope>;         |

  // clang-format on

  // [0]: volatile and atomics are only supported on global or shared
  //      memory locations, accessed via generic/shared/global pointers.
  //      MMIO is only supported on global memory locations,
  //      accessed via generic/global pointers.
  // TODO: Implement MMIO access via generic pointer to global.
  //       Currently implemented for global pointers only.

  // [1]: Lowering volatile/atomic operations to non-volatile/non-atomic
  //      PTX instructions fails to preserve their C++ side-effects.
  //
  //      Example (https://github.com/llvm/llvm-project/issues/62057):
  //
  //          void example() {
  //              std::atomic<bool> True = true;
  //              while (True.load(std::memory_order_relaxed));
  //          }
  //
  //      A C++ program that calls "example" is well-defined: the infinite loop
  //      performs an atomic operation. By lowering volatile/atomics to
  //      "weak" memory operations, we are transforming the above into:
  //
  //          void undefined_behavior() {
  //              bool True = true;
  //              while (True);
  //          }
  //
  //      which exhibits undefined behavior in both C++ and PTX.
  //
  //      Calling "example" in CUDA C++ compiled for sm_60- exhibits undefined
  //      behavior due to lack of Independent Forward Progress. Lowering these
  //      to weak memory operations in sm_60- is therefore fine.
  //
  //      TODO: lower atomic and volatile operations to memory locations
  //      in local, const, and param to two PTX instructions in sm_70+:
  //        - the "weak" memory instruction we are currently lowering to, and
  //        - some other instruction that preserves the side-effect, e.g.,
  //          a dead dummy volatile load.
  if (CodeAddrSpace == NVPTX::AddressSpace::Local ||
      CodeAddrSpace == NVPTX::AddressSpace::Const ||
      CodeAddrSpace == NVPTX::AddressSpace::Param) {
    return NVPTX::Ordering::NotAtomic;
  }

  // [2]: Atomics with Ordering different than Unordered or Relaxed are not
  //      supported on sm_60 and older; this includes volatile atomics.
  if (!(Ordering == AtomicOrdering::NotAtomic ||
        Ordering == AtomicOrdering::Unordered ||
        Ordering == AtomicOrdering::Monotonic) &&
      !HasMemoryOrdering) {
    report_fatal_error(
        formatv("PTX does not support \"atomic\" for orderings different than"
                "\"NotAtomic\" or \"Monotonic\" for sm_60 or older, but order "
                "is: \"{}\".",
                toIRString(Ordering)));
  }

  // [3]: TODO: these should eventually use .mmio<.atomic sem>; for now we drop
  // the volatile semantics and preserve the atomic ones.

  // PTX volatile and PTX atomics are not available for statespace that differ
  // from .generic, .global, or .shared. The behavior of PTX volatile and PTX
  // atomics is undefined if the generic address does not refer to a .global or
  // .shared memory location.
  bool AddrGenericOrGlobalOrShared =
      (CodeAddrSpace == NVPTX::AddressSpace::Generic ||
       CodeAddrSpace == NVPTX::AddressSpace::Global ||
       CodeAddrSpace == NVPTX::AddressSpace::Shared);
  if (!AddrGenericOrGlobalOrShared)
    return NVPTX::Ordering::NotAtomic;

  bool UseRelaxedMMIO =
      HasRelaxedMMIO && CodeAddrSpace == NVPTX::AddressSpace::Global;

  switch (Ordering) {
  case AtomicOrdering::NotAtomic:
    return N->isVolatile() ? NVPTX::Ordering::Volatile
                           : NVPTX::Ordering::NotAtomic;
  case AtomicOrdering::Unordered:
    // We lower unordered in the exact same way as 'monotonic' to respect
    // LLVM IR atomicity requirements.
  case AtomicOrdering::Monotonic:
    if (N->isVolatile())
      return UseRelaxedMMIO ? NVPTX::Ordering::RelaxedMMIO
                            : NVPTX::Ordering::Volatile;
    else
      return HasMemoryOrdering ? NVPTX::Ordering::Relaxed
                               : NVPTX::Ordering::Volatile;
  // case AtomicOrdering::Consume: // If LLVM ever provides this, lower it to
  // Acquire.
  case AtomicOrdering::Acquire:
    if (!N->readMem())
      report_fatal_error(
          formatv("PTX only supports Acquire Ordering on reads: {}",
                  N->getOperationName()));
    return NVPTX::Ordering::Acquire;
  case AtomicOrdering::Release:
    if (!N->writeMem())
      report_fatal_error(
          formatv("PTX only supports Release Ordering on writes: {}",
                  N->getOperationName()));
    return NVPTX::Ordering::Release;
  case AtomicOrdering::AcquireRelease: {
    report_fatal_error(
        formatv("NVPTX does not support AcquireRelease Ordering on "
                "read-modify-write "
                "yet and PTX does not support it on loads or stores: {}",
                N->getOperationName()));
  }
  case AtomicOrdering::SequentiallyConsistent: {
    // LLVM-IR SequentiallyConsistent atomics map to a two-instruction PTX
    // sequence including a "fence.sc.sco" and the memory instruction with an
    // Ordering that differs from "sc": acq, rel, or acq_rel, depending on
    // whether the memory operation is a read, write, or read-modify-write.
    //
    // This sets the ordering of the fence to SequentiallyConsistent, and
    // sets the corresponding ordering for the instruction.
    NVPTX::Ordering InstrOrder;
    if (N->readMem())
      InstrOrder = NVPTX::Ordering::Acquire;
    else if (N->writeMem())
      InstrOrder = NVPTX::Ordering::Release;
    else
      report_fatal_error(
          formatv("NVPTX does not support SequentiallyConsistent Ordering on "
                  "read-modify-writes yet: {}",
                  N->getOperationName()));
    return OperationOrderings(InstrOrder,
                              NVPTX::Ordering::SequentiallyConsistent);
  }
  }
  report_fatal_error(
      formatv("NVPTX backend does not support AtomicOrdering \"{}\" yet.",
              toIRString(Ordering)));
}

} // namespace

NVPTX::Scope NVPTXDAGToDAGISel::getOperationScope(MemSDNode *N,
                                                  NVPTX::Ordering O) const {
  switch (O) {
  case NVPTX::Ordering::NotAtomic:
  case NVPTX::Ordering::Volatile: // Non-atomic volatile operations
    // NVPTX uses Thread scope as the scope of non-atomic operations.
    return NVPTX::Scope::Thread;
  case NVPTX::Ordering::RelaxedMMIO:
    // RelaxedMMIO operations are always system scope.
    // If a RelaxedMMIO order was generated from an atomic volatile operation
    // with a smaller thread scope, we bump it here to system scope.
    return NVPTX::Scope::System;
  case NVPTX::Ordering::Relaxed:
  case NVPTX::Ordering::Acquire:
  case NVPTX::Ordering::Release:
  case NVPTX::Ordering::AcquireRelease:
  case NVPTX::Ordering::SequentiallyConsistent:
    auto S = Scopes[N->getSyncScopeID()];

    // Atomic operations must have a scope greater than thread.
    if (S == NVPTX::Scope::Thread)
      report_fatal_error(
          formatv("Atomics need scope > \"{}\".", ScopeToString(S)));

    // If scope is cluster, clusters must be supported.
    if (S == NVPTX::Scope::Cluster)
      Subtarget->failIfClustersUnsupported("cluster scope");

    // If operation is volatile, then its scope is system.
    return N->isVolatile() ? NVPTX::Scope::System : S;
  }
  llvm_unreachable("unhandled ordering");
}

static bool canLowerToLDG(MemSDNode *N, const NVPTXSubtarget &Subtarget,
                          unsigned CodeAddrSpace, MachineFunction *F) {
  // We use ldg (i.e. ld.global.nc) for invariant loads from the global address
  // space.
  //
  // We have two ways of identifying invariant loads: Loads may be explicitly
  // marked as invariant, or we may infer them to be invariant.
  //
  // We currently infer invariance for loads from
  //  - constant global variables, and
  //  - kernel function pointer params that are noalias (i.e. __restrict) and
  //    never written to.
  //
  // TODO: Perform a more powerful invariance analysis (ideally IPO, and ideally
  // not during the SelectionDAG phase).
  //
  // TODO: Infer invariance only at -O2.  We still want to use ldg at -O0 for
  // explicitly invariant loads because these are how clang tells us to use ldg
  // when the user uses a builtin.
  if (!Subtarget.hasLDG() || CodeAddrSpace != NVPTX::AddressSpace::Global)
    return false;

  if (N->isInvariant())
    return true;

  bool IsKernelFn = isKernelFunction(F->getFunction());

  // We use getUnderlyingObjects() here instead of getUnderlyingObject() mainly
  // because the former looks through phi nodes while the latter does not. We
  // need to look through phi nodes to handle pointer induction variables.
  SmallVector<const Value *, 8> Objs;
  getUnderlyingObjects(N->getMemOperand()->getValue(), Objs);

  return all_of(Objs, [&](const Value *V) {
    if (auto *A = dyn_cast<const Argument>(V))
      return IsKernelFn && A->onlyReadsMemory() && A->hasNoAliasAttr();
    if (auto *GV = dyn_cast<const GlobalVariable>(V))
      return GV->isConstant();
    return false;
  });
}

static unsigned int getFenceOp(NVPTX::Ordering O, NVPTX::Scope S,
                               NVPTXSubtarget const *T) {
  if (S == NVPTX::Scope::Cluster)
    T->failIfClustersUnsupported(".cluster scope fence");

  // Fall back to .acq_rel if .acquire, .release is not supported.
  if (!T->hasSplitAcquireAndReleaseFences() &&
      (O == NVPTX::Ordering::Acquire || O == NVPTX::Ordering::Release))
    O = NVPTX::Ordering::AcquireRelease;

  switch (O) {
  case NVPTX::Ordering::Acquire:
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acquire_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acquire_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_acquire_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acquire_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
      report_fatal_error(
          formatv("Unsupported scope \"{}\" for acquire/release/acq_rel fence.",
                  ScopeToString(S)));
    }
    break;
  case NVPTX::Ordering::Release:
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_release_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_release_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_release_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_release_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
      report_fatal_error(
          formatv("Unsupported scope \"{}\" for acquire/release/acq_rel fence.",
                  ScopeToString(S)));
    }
    break;
  case NVPTX::Ordering::AcquireRelease: {
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acq_rel_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acq_rel_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_acq_rel_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_acq_rel_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
      report_fatal_error(
          formatv("Unsupported scope \"{}\" for acquire/release/acq_rel fence.",
                  ScopeToString(S)));
    }
    break;
  }
  case NVPTX::Ordering::SequentiallyConsistent: {
    switch (S) {
    case NVPTX::Scope::System:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_seq_cst_sys
                                    : NVPTX::INT_MEMBAR_SYS;
    case NVPTX::Scope::Block:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_seq_cst_cta
                                    : NVPTX::INT_MEMBAR_CTA;
    case NVPTX::Scope::Cluster:
      return NVPTX::atomic_thread_fence_seq_cst_cluster;
    case NVPTX::Scope::Device:
      return T->hasMemoryOrdering() ? NVPTX::atomic_thread_fence_seq_cst_gpu
                                    : NVPTX::INT_MEMBAR_GL;
    case NVPTX::Scope::Thread:
      report_fatal_error(formatv("Unsupported scope \"{}\" for seq_cst fence.",
                                 ScopeToString(S)));
    }
    break;
  }
  case NVPTX::Ordering::NotAtomic:
  case NVPTX::Ordering::Relaxed:
  case NVPTX::Ordering::Volatile:
  case NVPTX::Ordering::RelaxedMMIO:
    report_fatal_error(
        formatv("Unsupported \"{}\" ordering and \"{}\" scope for fence.",
                OrderingToString(O), ScopeToString(S)));
  }
  llvm_unreachable("unhandled ordering");
}

// Returns Memory Order and Scope of a memory instruction, and
// inserts any fence before the instruction that's required to
// implement its memory ordering.
std::pair<NVPTX::Ordering, NVPTX::Scope>
NVPTXDAGToDAGISel::insertMemoryInstructionFence(SDLoc DL, SDValue &Chain,
                                                MemSDNode *N) {
  auto [InstructionOrdering, FenceOrdering] =
      getOperationOrderings(N, Subtarget);
  auto Scope = getOperationScope(N, InstructionOrdering);

  // If a fence is required before the operation, insert it:
  switch (NVPTX::Ordering(FenceOrdering)) {
  case NVPTX::Ordering::NotAtomic:
    break;
  case NVPTX::Ordering::SequentiallyConsistent: {
    auto Op = getFenceOp(FenceOrdering, Scope, Subtarget);
    Chain = SDValue(CurDAG->getMachineNode(Op, DL, MVT::Other, Chain), 0);
    break;
  }
  default:
    report_fatal_error(
        formatv("Unexpected fence ordering: \"{}\".",
                OrderingToString(NVPTX::Ordering(FenceOrdering))));
  }
  return {InstructionOrdering, Scope};
}

bool NVPTXDAGToDAGISel::tryIntrinsicNoChain(SDNode *N) {
  unsigned IID = N->getConstantOperandVal(0);
  switch (IID) {
  default:
    return false;
  case Intrinsic::nvvm_texsurf_handle_internal:
    SelectTexSurfHandle(N);
    return true;
  }
}

void NVPTXDAGToDAGISel::SelectTexSurfHandle(SDNode *N) {
  // Op 0 is the intrinsic ID
  SDValue Wrapper = N->getOperand(1);
  SDValue GlobalVal = Wrapper.getOperand(0);
  ReplaceNode(N, CurDAG->getMachineNode(NVPTX::texsurf_handles, SDLoc(N),
                                        MVT::i64, GlobalVal));
}

void NVPTXDAGToDAGISel::SelectAddrSpaceCast(SDNode *N) {
  SDValue Src = N->getOperand(0);
  AddrSpaceCastSDNode *CastN = cast<AddrSpaceCastSDNode>(N);
  unsigned SrcAddrSpace = CastN->getSrcAddressSpace();
  unsigned DstAddrSpace = CastN->getDestAddressSpace();
  SDLoc DL(N);
  assert(SrcAddrSpace != DstAddrSpace &&
         "addrspacecast must be between different address spaces");

  if (DstAddrSpace == ADDRESS_SPACE_GENERIC) {
    // Specific to generic

    if (TM.is64Bit() && TM.getPointerSizeInBits(SrcAddrSpace) == 32) {
      SDValue CvtNone =
          CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE, DL, MVT::i32);
      SDNode *Cvt = CurDAG->getMachineNode(NVPTX::CVT_u64_u32, DL, MVT::i64,
                                           Src, CvtNone);
      Src = SDValue(Cvt, 0);
    }

    unsigned Opc;
    switch (SrcAddrSpace) {
    default: report_fatal_error("Bad address space in addrspacecast");
    case ADDRESS_SPACE_GLOBAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_global_64 : NVPTX::cvta_global;
      break;
    case ADDRESS_SPACE_SHARED:
      Opc = TM.is64Bit() ? NVPTX::cvta_shared_64 : NVPTX::cvta_shared;
      break;
    case ADDRESS_SPACE_CONST:
      Opc = TM.is64Bit() ? NVPTX::cvta_const_64 : NVPTX::cvta_const;
      break;
    case ADDRESS_SPACE_LOCAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_local_64 : NVPTX::cvta_local;
      break;
    }
    ReplaceNode(N, CurDAG->getMachineNode(Opc, DL, N->getValueType(0), Src));
    return;
  } else {
    // Generic to specific
    if (SrcAddrSpace != 0)
      report_fatal_error("Cannot cast between two non-generic address spaces");
    unsigned Opc;
    switch (DstAddrSpace) {
    default: report_fatal_error("Bad address space in addrspacecast");
    case ADDRESS_SPACE_GLOBAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_global_64 : NVPTX::cvta_to_global;
      break;
    case ADDRESS_SPACE_SHARED:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_shared_64 : NVPTX::cvta_to_shared;
      break;
    case ADDRESS_SPACE_CONST:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_const_64 : NVPTX::cvta_to_const;
      break;
    case ADDRESS_SPACE_LOCAL:
      Opc = TM.is64Bit() ? NVPTX::cvta_to_local_64 : NVPTX::cvta_to_local;
      break;
    case ADDRESS_SPACE_PARAM:
      Opc = TM.is64Bit() ? NVPTX::IMOV64rr : NVPTX::IMOV32rr;
      break;
    }

    SDNode *CVTA = CurDAG->getMachineNode(Opc, DL, N->getValueType(0), Src);
    if (TM.is64Bit() && TM.getPointerSizeInBits(DstAddrSpace) == 32) {
      SDValue CvtNone =
          CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE, DL, MVT::i32);
      CVTA = CurDAG->getMachineNode(NVPTX::CVT_u32_u64, DL, MVT::i32,
                                    SDValue(CVTA, 0), CvtNone);
    }

    ReplaceNode(N, CVTA);
    return;
  }
}

// Helper function template to reduce amount of boilerplate code for
// opcode selection.
static std::optional<unsigned>
pickOpcodeForVT(MVT::SimpleValueType VT, unsigned Opcode_i8,
                unsigned Opcode_i16, unsigned Opcode_i32,
                std::optional<unsigned> Opcode_i64, unsigned Opcode_f32,
                std::optional<unsigned> Opcode_f64) {
  switch (VT) {
  case MVT::i1:
  case MVT::i8:
    return Opcode_i8;
  case MVT::i16:
    return Opcode_i16;
  case MVT::i32:
    return Opcode_i32;
  case MVT::i64:
    return Opcode_i64;
  case MVT::f16:
  case MVT::bf16:
    return Opcode_i16;
  case MVT::v2f16:
  case MVT::v2bf16:
  case MVT::v2i16:
  case MVT::v4i8:
    return Opcode_i32;
  case MVT::f32:
    return Opcode_f32;
  case MVT::f64:
    return Opcode_f64;
  default:
    return std::nullopt;
  }
}

static int getLdStRegType(EVT VT) {
  if (VT.isFloatingPoint())
    switch (VT.getSimpleVT().SimpleTy) {
    case MVT::f16:
    case MVT::bf16:
    case MVT::v2f16:
    case MVT::v2bf16:
      return NVPTX::PTXLdStInstCode::Untyped;
    default:
      return NVPTX::PTXLdStInstCode::Float;
    }
  else
    return NVPTX::PTXLdStInstCode::Unsigned;
}

bool NVPTXDAGToDAGISel::tryLoad(SDNode *N) {
  MemSDNode *LD = cast<MemSDNode>(N);
  assert(LD->readMem() && "Expected load");

  // do not support pre/post inc/dec
  LoadSDNode *PlainLoad = dyn_cast<LoadSDNode>(N);
  if (PlainLoad && PlainLoad->isIndexed())
    return false;

  EVT LoadedVT = LD->getMemoryVT();
  if (!LoadedVT.isSimple())
    return false;

  // Address Space Setting
  unsigned int CodeAddrSpace = getCodeAddrSpace(LD);
  if (canLowerToLDG(LD, *Subtarget, CodeAddrSpace, MF)) {
    return tryLDGLDU(N);
  }
  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(LD->getAddressSpace());

  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, LD);

  // Type Setting: fromType + fromTypeWidth
  //
  // Sign   : ISD::SEXTLOAD
  // Unsign : ISD::ZEXTLOAD, ISD::NON_EXTLOAD or ISD::EXTLOAD and the
  //          type is integer
  // Float  : ISD::NON_EXTLOAD or ISD::EXTLOAD and the type is float
  MVT SimpleVT = LoadedVT.getSimpleVT();
  MVT ScalarVT = SimpleVT.getScalarType();
  // Read at least 8 bits (predicates are stored as 8-bit values)
  unsigned FromTypeWidth = std::max(8U, (unsigned)ScalarVT.getSizeInBits());
  unsigned int FromType;

  // Vector Setting
  unsigned VecType = NVPTX::PTXLdStInstCode::Scalar;
  if (SimpleVT.isVector()) {
    assert((Isv2x16VT(LoadedVT) || LoadedVT == MVT::v4i8) &&
           "Unexpected vector type");
    // v2f16/v2bf16/v2i16 is loaded using ld.b32
    FromTypeWidth = 32;
  }

  if (PlainLoad && (PlainLoad->getExtensionType() == ISD::SEXTLOAD))
    FromType = NVPTX::PTXLdStInstCode::Signed;
  else
    FromType = getLdStRegType(ScalarVT);

  // Create the machine instruction DAG
  SDValue N1 = N->getOperand(1);
  SDValue Addr;
  SDValue Offset, Base;
  std::optional<unsigned> Opcode;
  MVT::SimpleValueType TargetVT = LD->getSimpleValueType(0).SimpleTy;

  SmallVector<SDValue, 12> Ops({getI32Imm(Ordering, DL), getI32Imm(Scope, DL),
                                getI32Imm(CodeAddrSpace, DL),
                                getI32Imm(VecType, DL), getI32Imm(FromType, DL),
                                getI32Imm(FromTypeWidth, DL)});

  if (SelectDirectAddr(N1, Addr)) {
    Opcode = pickOpcodeForVT(TargetVT, NVPTX::LD_i8_avar, NVPTX::LD_i16_avar,
                             NVPTX::LD_i32_avar, NVPTX::LD_i64_avar,
                             NVPTX::LD_f32_avar, NVPTX::LD_f64_avar);
    if (!Opcode)
      return false;
    Ops.append({Addr, Chain});
  } else if (PointerSize == 64 ? SelectADDRsi64(N1.getNode(), N1, Base, Offset)
                               : SelectADDRsi(N1.getNode(), N1, Base, Offset)) {
    Opcode = pickOpcodeForVT(TargetVT, NVPTX::LD_i8_asi, NVPTX::LD_i16_asi,
                             NVPTX::LD_i32_asi, NVPTX::LD_i64_asi,
                             NVPTX::LD_f32_asi, NVPTX::LD_f64_asi);
    if (!Opcode)
      return false;
    Ops.append({Base, Offset, Chain});
  } else if (PointerSize == 64 ? SelectADDRri64(N1.getNode(), N1, Base, Offset)
                               : SelectADDRri(N1.getNode(), N1, Base, Offset)) {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(TargetVT, NVPTX::LD_i8_ari_64, NVPTX::LD_i16_ari_64,
                          NVPTX::LD_i32_ari_64, NVPTX::LD_i64_ari_64,
                          NVPTX::LD_f32_ari_64, NVPTX::LD_f64_ari_64);
    else
      Opcode = pickOpcodeForVT(TargetVT, NVPTX::LD_i8_ari, NVPTX::LD_i16_ari,
                               NVPTX::LD_i32_ari, NVPTX::LD_i64_ari,
                               NVPTX::LD_f32_ari, NVPTX::LD_f64_ari);
    if (!Opcode)
      return false;
    Ops.append({Base, Offset, Chain});
  } else {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(TargetVT, NVPTX::LD_i8_areg_64, NVPTX::LD_i16_areg_64,
                          NVPTX::LD_i32_areg_64, NVPTX::LD_i64_areg_64,
                          NVPTX::LD_f32_areg_64, NVPTX::LD_f64_areg_64);
    else
      Opcode = pickOpcodeForVT(TargetVT, NVPTX::LD_i8_areg, NVPTX::LD_i16_areg,
                               NVPTX::LD_i32_areg, NVPTX::LD_i64_areg,
                               NVPTX::LD_f32_areg, NVPTX::LD_f64_areg);
    if (!Opcode)
      return false;
    Ops.append({N1, Chain});
  }

  SDNode *NVPTXLD =
      CurDAG->getMachineNode(*Opcode, DL, TargetVT, MVT::Other, Ops);
  if (!NVPTXLD)
    return false;

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(NVPTXLD), {MemRef});

  ReplaceNode(N, NVPTXLD);
  return true;
}

static bool isVectorElementTypeUpsized(EVT EltVT) {
  // Despite vectors like v8i8, v16i8, v8i16 being within the bit-limit for
  // total load/store size, PTX syntax only supports v2/v4. Thus, we can't use
  // vectorized loads/stores with the actual element type for i8/i16 as that
  // would require v8/v16 variants that do not exist.
  // In order to load/store such vectors efficiently, in Type Legalization
  // we split the vector into word-sized chunks (v2x16/v4i8). Now, we will
  // lower to PTX as vectors of b32.
  return Isv2x16VT(EltVT) || EltVT == MVT::v4i8;
}

bool NVPTXDAGToDAGISel::tryLoadVector(SDNode *N) {
  MemSDNode *MemSD = cast<MemSDNode>(N);
  EVT LoadedVT = MemSD->getMemoryVT();
  if (!LoadedVT.isSimple())
    return false;

  // Address Space Setting
  unsigned int CodeAddrSpace = getCodeAddrSpace(MemSD);
  if (canLowerToLDG(MemSD, *Subtarget, CodeAddrSpace, MF)) {
    return tryLDGLDU(N);
  }
  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(MemSD->getAddressSpace());

  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, MemSD);

  // Vector Setting
  MVT SimpleVT = LoadedVT.getSimpleVT();

  // Type Setting: fromType + fromTypeWidth
  //
  // Sign   : ISD::SEXTLOAD
  // Unsign : ISD::ZEXTLOAD, ISD::NON_EXTLOAD or ISD::EXTLOAD and the
  //          type is integer
  // Float  : ISD::NON_EXTLOAD or ISD::EXTLOAD and the type is float
  MVT ScalarVT = SimpleVT.getScalarType();
  // Read at least 8 bits (predicates are stored as 8-bit values)
  unsigned FromTypeWidth = std::max(8U, (unsigned)ScalarVT.getSizeInBits());
  unsigned int FromType;
  // The last operand holds the original LoadSDNode::getExtensionType() value
  unsigned ExtensionType = cast<ConstantSDNode>(
      N->getOperand(N->getNumOperands() - 1))->getZExtValue();
  if (ExtensionType == ISD::SEXTLOAD)
    FromType = NVPTX::PTXLdStInstCode::Signed;
  else
    FromType = getLdStRegType(ScalarVT);

  unsigned VecType;

  switch (N->getOpcode()) {
  case NVPTXISD::LoadV2:
    VecType = NVPTX::PTXLdStInstCode::V2;
    break;
  case NVPTXISD::LoadV4:
    VecType = NVPTX::PTXLdStInstCode::V4;
    break;
  default:
    return false;
  }

  EVT EltVT = N->getValueType(0);

  if (isVectorElementTypeUpsized(EltVT)) {
    EltVT = MVT::i32;
    FromType = NVPTX::PTXLdStInstCode::Untyped;
    FromTypeWidth = 32;
  }

  SDValue Op1 = N->getOperand(1);
  SDValue Addr, Offset, Base;
  std::optional<unsigned> Opcode;
  SDNode *LD;

  SmallVector<SDValue, 12> Ops({getI32Imm(Ordering, DL), getI32Imm(Scope, DL),
                                getI32Imm(CodeAddrSpace, DL),
                                getI32Imm(VecType, DL), getI32Imm(FromType, DL),
                                getI32Imm(FromTypeWidth, DL)});

  if (SelectDirectAddr(Op1, Addr)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case NVPTXISD::LoadV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::LDV_i8_v2_avar, NVPTX::LDV_i16_v2_avar,
                               NVPTX::LDV_i32_v2_avar, NVPTX::LDV_i64_v2_avar,
                               NVPTX::LDV_f32_v2_avar, NVPTX::LDV_f64_v2_avar);
      break;
    case NVPTXISD::LoadV4:
      Opcode =
          pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v4_avar,
                          NVPTX::LDV_i16_v4_avar, NVPTX::LDV_i32_v4_avar,
                          std::nullopt, NVPTX::LDV_f32_v4_avar, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    Ops.append({Addr, Chain});
  } else if (PointerSize == 64
                 ? SelectADDRsi64(Op1.getNode(), Op1, Base, Offset)
                 : SelectADDRsi(Op1.getNode(), Op1, Base, Offset)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case NVPTXISD::LoadV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::LDV_i8_v2_asi, NVPTX::LDV_i16_v2_asi,
                               NVPTX::LDV_i32_v2_asi, NVPTX::LDV_i64_v2_asi,
                               NVPTX::LDV_f32_v2_asi, NVPTX::LDV_f64_v2_asi);
      break;
    case NVPTXISD::LoadV4:
      Opcode =
          pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v4_asi,
                          NVPTX::LDV_i16_v4_asi, NVPTX::LDV_i32_v4_asi,
                          std::nullopt, NVPTX::LDV_f32_v4_asi, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    Ops.append({Base, Offset, Chain});
  } else if (PointerSize == 64
                 ? SelectADDRri64(Op1.getNode(), Op1, Base, Offset)
                 : SelectADDRri(Op1.getNode(), Op1, Base, Offset)) {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::LoadV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                            NVPTX::LDV_i8_v2_ari_64, NVPTX::LDV_i16_v2_ari_64,
                            NVPTX::LDV_i32_v2_ari_64, NVPTX::LDV_i64_v2_ari_64,
                            NVPTX::LDV_f32_v2_ari_64, NVPTX::LDV_f64_v2_ari_64);
        break;
      case NVPTXISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v4_ari_64,
            NVPTX::LDV_i16_v4_ari_64, NVPTX::LDV_i32_v4_ari_64, std::nullopt,
            NVPTX::LDV_f32_v4_ari_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::LoadV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::LDV_i8_v2_ari, NVPTX::LDV_i16_v2_ari,
                                 NVPTX::LDV_i32_v2_ari, NVPTX::LDV_i64_v2_ari,
                                 NVPTX::LDV_f32_v2_ari, NVPTX::LDV_f64_v2_ari);
        break;
      case NVPTXISD::LoadV4:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v4_ari,
                            NVPTX::LDV_i16_v4_ari, NVPTX::LDV_i32_v4_ari,
                            std::nullopt, NVPTX::LDV_f32_v4_ari, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    Ops.append({Base, Offset, Chain});
  } else {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::LoadV2:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v2_areg_64,
            NVPTX::LDV_i16_v2_areg_64, NVPTX::LDV_i32_v2_areg_64,
            NVPTX::LDV_i64_v2_areg_64, NVPTX::LDV_f32_v2_areg_64,
            NVPTX::LDV_f64_v2_areg_64);
        break;
      case NVPTXISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v4_areg_64,
            NVPTX::LDV_i16_v4_areg_64, NVPTX::LDV_i32_v4_areg_64, std::nullopt,
            NVPTX::LDV_f32_v4_areg_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::LoadV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v2_areg,
                            NVPTX::LDV_i16_v2_areg, NVPTX::LDV_i32_v2_areg,
                            NVPTX::LDV_i64_v2_areg, NVPTX::LDV_f32_v2_areg,
                            NVPTX::LDV_f64_v2_areg);
        break;
      case NVPTXISD::LoadV4:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::LDV_i8_v4_areg,
                            NVPTX::LDV_i16_v4_areg, NVPTX::LDV_i32_v4_areg,
                            std::nullopt, NVPTX::LDV_f32_v4_areg, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    Ops.append({Op1, Chain});
  }
  LD = CurDAG->getMachineNode(*Opcode, DL, N->getVTList(), Ops);

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(LD), {MemRef});

  ReplaceNode(N, LD);
  return true;
}

bool NVPTXDAGToDAGISel::tryLDGLDU(SDNode *N) {
  auto *Mem = cast<MemSDNode>(N);

  // If this is an LDG intrinsic, the address is the third operand. If its an
  // LDG/LDU SD node (from custom vector handling), then its the second operand
  SDValue Op1 = N->getOperand(N->getOpcode() == ISD::INTRINSIC_W_CHAIN ? 2 : 1);

  EVT OrigType = N->getValueType(0);
  EVT EltVT = Mem->getMemoryVT();
  unsigned NumElts = 1;
  if (EltVT.isVector()) {
    NumElts = EltVT.getVectorNumElements();
    EltVT = EltVT.getVectorElementType();
    // vectors of 8/16bits type are loaded/stored as multiples of v4i8/v2x16
    // elements.
    if ((EltVT == MVT::f16 && OrigType == MVT::v2f16) ||
        (EltVT == MVT::bf16 && OrigType == MVT::v2bf16) ||
        (EltVT == MVT::i16 && OrigType == MVT::v2i16) ||
        (EltVT == MVT::i8 && OrigType == MVT::v4i8)) {
      assert(NumElts % OrigType.getVectorNumElements() == 0 &&
             "NumElts must be divisible by the number of elts in subvectors");
      EltVT = OrigType;
      NumElts /= OrigType.getVectorNumElements();
    }
  }

  // Build the "promoted" result VTList for the load. If we are really loading
  // i8s, then the return type will be promoted to i16 since we do not expose
  // 8-bit registers in NVPTX.
  EVT NodeVT = (EltVT == MVT::i8) ? MVT::i16 : EltVT;
  SmallVector<EVT, 5> InstVTs;
  for (unsigned i = 0; i != NumElts; ++i) {
    InstVTs.push_back(NodeVT);
  }
  InstVTs.push_back(MVT::Other);
  SDVTList InstVTList = CurDAG->getVTList(InstVTs);
  SDValue Chain = N->getOperand(0);

  std::optional<unsigned> Opcode;
  SDLoc DL(N);
  SDNode *LD;
  SDValue Base, Offset, Addr;

  if (SelectDirectAddr(Op1, Addr)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case ISD::LOAD:
      Opcode = pickOpcodeForVT(
          EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_GLOBAL_i8avar,
          NVPTX::INT_PTX_LDG_GLOBAL_i16avar, NVPTX::INT_PTX_LDG_GLOBAL_i32avar,
          NVPTX::INT_PTX_LDG_GLOBAL_i64avar, NVPTX::INT_PTX_LDG_GLOBAL_f32avar,
          NVPTX::INT_PTX_LDG_GLOBAL_f64avar);
      break;
    case ISD::INTRINSIC_W_CHAIN:
      Opcode = pickOpcodeForVT(
          EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_GLOBAL_i8avar,
          NVPTX::INT_PTX_LDU_GLOBAL_i16avar, NVPTX::INT_PTX_LDU_GLOBAL_i32avar,
          NVPTX::INT_PTX_LDU_GLOBAL_i64avar, NVPTX::INT_PTX_LDU_GLOBAL_f32avar,
          NVPTX::INT_PTX_LDU_GLOBAL_f64avar);
      break;
    case NVPTXISD::LoadV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::INT_PTX_LDG_G_v2i8_ELE_avar,
                               NVPTX::INT_PTX_LDG_G_v2i16_ELE_avar,
                               NVPTX::INT_PTX_LDG_G_v2i32_ELE_avar,
                               NVPTX::INT_PTX_LDG_G_v2i64_ELE_avar,
                               NVPTX::INT_PTX_LDG_G_v2f32_ELE_avar,
                               NVPTX::INT_PTX_LDG_G_v2f64_ELE_avar);
      break;
    case NVPTXISD::LDUV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::INT_PTX_LDU_G_v2i8_ELE_avar,
                               NVPTX::INT_PTX_LDU_G_v2i16_ELE_avar,
                               NVPTX::INT_PTX_LDU_G_v2i32_ELE_avar,
                               NVPTX::INT_PTX_LDU_G_v2i64_ELE_avar,
                               NVPTX::INT_PTX_LDU_G_v2f32_ELE_avar,
                               NVPTX::INT_PTX_LDU_G_v2f64_ELE_avar);
      break;
    case NVPTXISD::LoadV4:
      Opcode = pickOpcodeForVT(
          EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_G_v4i8_ELE_avar,
          NVPTX::INT_PTX_LDG_G_v4i16_ELE_avar,
          NVPTX::INT_PTX_LDG_G_v4i32_ELE_avar, std::nullopt,
          NVPTX::INT_PTX_LDG_G_v4f32_ELE_avar, std::nullopt);
      break;
    case NVPTXISD::LDUV4:
      Opcode = pickOpcodeForVT(
          EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_G_v4i8_ELE_avar,
          NVPTX::INT_PTX_LDU_G_v4i16_ELE_avar,
          NVPTX::INT_PTX_LDU_G_v4i32_ELE_avar, std::nullopt,
          NVPTX::INT_PTX_LDU_G_v4f32_ELE_avar, std::nullopt);
      break;
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { Addr, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, InstVTList, Ops);
  } else if (TM.is64Bit() ? SelectADDRri64(Op1.getNode(), Op1, Base, Offset)
                          : SelectADDRri(Op1.getNode(), Op1, Base, Offset)) {
    if (TM.is64Bit()) {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i8ari64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i16ari64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i32ari64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i64ari64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_f32ari64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_f64ari64);
        break;
      case ISD::INTRINSIC_W_CHAIN:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i8ari64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i16ari64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i32ari64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i64ari64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_f32ari64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_f64ari64);
        break;
      case NVPTXISD::LoadV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     NVPTX::INT_PTX_LDG_G_v2i8_ELE_ari64,
                                     NVPTX::INT_PTX_LDG_G_v2i16_ELE_ari64,
                                     NVPTX::INT_PTX_LDG_G_v2i32_ELE_ari64,
                                     NVPTX::INT_PTX_LDG_G_v2i64_ELE_ari64,
                                     NVPTX::INT_PTX_LDG_G_v2f32_ELE_ari64,
                                     NVPTX::INT_PTX_LDG_G_v2f64_ELE_ari64);
        break;
      case NVPTXISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     NVPTX::INT_PTX_LDU_G_v2i8_ELE_ari64,
                                     NVPTX::INT_PTX_LDU_G_v2i16_ELE_ari64,
                                     NVPTX::INT_PTX_LDU_G_v2i32_ELE_ari64,
                                     NVPTX::INT_PTX_LDU_G_v2i64_ELE_ari64,
                                     NVPTX::INT_PTX_LDU_G_v2f32_ELE_ari64,
                                     NVPTX::INT_PTX_LDU_G_v2f64_ELE_ari64);
        break;
      case NVPTXISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_G_v4i8_ELE_ari64,
            NVPTX::INT_PTX_LDG_G_v4i16_ELE_ari64,
            NVPTX::INT_PTX_LDG_G_v4i32_ELE_ari64, std::nullopt,
            NVPTX::INT_PTX_LDG_G_v4f32_ELE_ari64, std::nullopt);
        break;
      case NVPTXISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_G_v4i8_ELE_ari64,
            NVPTX::INT_PTX_LDU_G_v4i16_ELE_ari64,
            NVPTX::INT_PTX_LDU_G_v4i32_ELE_ari64, std::nullopt,
            NVPTX::INT_PTX_LDU_G_v4f32_ELE_ari64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_GLOBAL_i8ari,
            NVPTX::INT_PTX_LDG_GLOBAL_i16ari, NVPTX::INT_PTX_LDG_GLOBAL_i32ari,
            NVPTX::INT_PTX_LDG_GLOBAL_i64ari, NVPTX::INT_PTX_LDG_GLOBAL_f32ari,
            NVPTX::INT_PTX_LDG_GLOBAL_f64ari);
        break;
      case ISD::INTRINSIC_W_CHAIN:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_GLOBAL_i8ari,
            NVPTX::INT_PTX_LDU_GLOBAL_i16ari, NVPTX::INT_PTX_LDU_GLOBAL_i32ari,
            NVPTX::INT_PTX_LDU_GLOBAL_i64ari, NVPTX::INT_PTX_LDU_GLOBAL_f32ari,
            NVPTX::INT_PTX_LDU_GLOBAL_f64ari);
        break;
      case NVPTXISD::LoadV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDG_G_v2i8_ELE_ari32,
                                 NVPTX::INT_PTX_LDG_G_v2i16_ELE_ari32,
                                 NVPTX::INT_PTX_LDG_G_v2i32_ELE_ari32,
                                 NVPTX::INT_PTX_LDG_G_v2i64_ELE_ari32,
                                 NVPTX::INT_PTX_LDG_G_v2f32_ELE_ari32,
                                 NVPTX::INT_PTX_LDG_G_v2f64_ELE_ari32);
        break;
      case NVPTXISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDU_G_v2i8_ELE_ari32,
                                 NVPTX::INT_PTX_LDU_G_v2i16_ELE_ari32,
                                 NVPTX::INT_PTX_LDU_G_v2i32_ELE_ari32,
                                 NVPTX::INT_PTX_LDU_G_v2i64_ELE_ari32,
                                 NVPTX::INT_PTX_LDU_G_v2f32_ELE_ari32,
                                 NVPTX::INT_PTX_LDU_G_v2f64_ELE_ari32);
        break;
      case NVPTXISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_G_v4i8_ELE_ari32,
            NVPTX::INT_PTX_LDG_G_v4i16_ELE_ari32,
            NVPTX::INT_PTX_LDG_G_v4i32_ELE_ari32, std::nullopt,
            NVPTX::INT_PTX_LDG_G_v4f32_ELE_ari32, std::nullopt);
        break;
      case NVPTXISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_G_v4i8_ELE_ari32,
            NVPTX::INT_PTX_LDU_G_v4i16_ELE_ari32,
            NVPTX::INT_PTX_LDU_G_v4i32_ELE_ari32, std::nullopt,
            NVPTX::INT_PTX_LDU_G_v4f32_ELE_ari32, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = {Base, Offset, Chain};
    LD = CurDAG->getMachineNode(*Opcode, DL, InstVTList, Ops);
  } else {
    if (TM.is64Bit()) {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i8areg64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i16areg64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i32areg64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i64areg64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_f32areg64,
                                 NVPTX::INT_PTX_LDG_GLOBAL_f64areg64);
        break;
      case ISD::INTRINSIC_W_CHAIN:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i8areg64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i16areg64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i32areg64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i64areg64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_f32areg64,
                                 NVPTX::INT_PTX_LDU_GLOBAL_f64areg64);
        break;
      case NVPTXISD::LoadV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     NVPTX::INT_PTX_LDG_G_v2i8_ELE_areg64,
                                     NVPTX::INT_PTX_LDG_G_v2i16_ELE_areg64,
                                     NVPTX::INT_PTX_LDG_G_v2i32_ELE_areg64,
                                     NVPTX::INT_PTX_LDG_G_v2i64_ELE_areg64,
                                     NVPTX::INT_PTX_LDG_G_v2f32_ELE_areg64,
                                     NVPTX::INT_PTX_LDG_G_v2f64_ELE_areg64);
        break;
      case NVPTXISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                     NVPTX::INT_PTX_LDU_G_v2i8_ELE_areg64,
                                     NVPTX::INT_PTX_LDU_G_v2i16_ELE_areg64,
                                     NVPTX::INT_PTX_LDU_G_v2i32_ELE_areg64,
                                     NVPTX::INT_PTX_LDU_G_v2i64_ELE_areg64,
                                     NVPTX::INT_PTX_LDU_G_v2f32_ELE_areg64,
                                     NVPTX::INT_PTX_LDU_G_v2f64_ELE_areg64);
        break;
      case NVPTXISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_G_v4i8_ELE_areg64,
            NVPTX::INT_PTX_LDG_G_v4i16_ELE_areg64,
            NVPTX::INT_PTX_LDG_G_v4i32_ELE_areg64, std::nullopt,
            NVPTX::INT_PTX_LDG_G_v4f32_ELE_areg64, std::nullopt);
        break;
      case NVPTXISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_G_v4i8_ELE_areg64,
            NVPTX::INT_PTX_LDU_G_v4i16_ELE_areg64,
            NVPTX::INT_PTX_LDU_G_v4i32_ELE_areg64, std::nullopt,
            NVPTX::INT_PTX_LDU_G_v4f32_ELE_areg64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case ISD::LOAD:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i8areg,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i16areg,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i32areg,
                                 NVPTX::INT_PTX_LDG_GLOBAL_i64areg,
                                 NVPTX::INT_PTX_LDG_GLOBAL_f32areg,
                                 NVPTX::INT_PTX_LDG_GLOBAL_f64areg);
        break;
      case ISD::INTRINSIC_W_CHAIN:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i8areg,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i16areg,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i32areg,
                                 NVPTX::INT_PTX_LDU_GLOBAL_i64areg,
                                 NVPTX::INT_PTX_LDU_GLOBAL_f32areg,
                                 NVPTX::INT_PTX_LDU_GLOBAL_f64areg);
        break;
      case NVPTXISD::LoadV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDG_G_v2i8_ELE_areg32,
                                 NVPTX::INT_PTX_LDG_G_v2i16_ELE_areg32,
                                 NVPTX::INT_PTX_LDG_G_v2i32_ELE_areg32,
                                 NVPTX::INT_PTX_LDG_G_v2i64_ELE_areg32,
                                 NVPTX::INT_PTX_LDG_G_v2f32_ELE_areg32,
                                 NVPTX::INT_PTX_LDG_G_v2f64_ELE_areg32);
        break;
      case NVPTXISD::LDUV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::INT_PTX_LDU_G_v2i8_ELE_areg32,
                                 NVPTX::INT_PTX_LDU_G_v2i16_ELE_areg32,
                                 NVPTX::INT_PTX_LDU_G_v2i32_ELE_areg32,
                                 NVPTX::INT_PTX_LDU_G_v2i64_ELE_areg32,
                                 NVPTX::INT_PTX_LDU_G_v2f32_ELE_areg32,
                                 NVPTX::INT_PTX_LDU_G_v2f64_ELE_areg32);
        break;
      case NVPTXISD::LoadV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDG_G_v4i8_ELE_areg32,
            NVPTX::INT_PTX_LDG_G_v4i16_ELE_areg32,
            NVPTX::INT_PTX_LDG_G_v4i32_ELE_areg32, std::nullopt,
            NVPTX::INT_PTX_LDG_G_v4f32_ELE_areg32, std::nullopt);
        break;
      case NVPTXISD::LDUV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::INT_PTX_LDU_G_v4i8_ELE_areg32,
            NVPTX::INT_PTX_LDU_G_v4i16_ELE_areg32,
            NVPTX::INT_PTX_LDU_G_v4i32_ELE_areg32, std::nullopt,
            NVPTX::INT_PTX_LDU_G_v4f32_ELE_areg32, std::nullopt);
        break;
      }
    }
    if (!Opcode)
      return false;
    SDValue Ops[] = { Op1, Chain };
    LD = CurDAG->getMachineNode(*Opcode, DL, InstVTList, Ops);
  }

  // For automatic generation of LDG (through SelectLoad[Vector], not the
  // intrinsics), we may have an extending load like:
  //
  //   i32,ch = load<LD1[%data1(addrspace=1)], zext from i8> t0, t7, undef:i64
  //
  // In this case, the matching logic above will select a load for the original
  // memory type (in this case, i8) and our types will not match (the node needs
  // to return an i32 in this case). Our LDG/LDU nodes do not support the
  // concept of sign-/zero-extension, so emulate it here by adding an explicit
  // CVT instruction. Ptxas should clean up any redundancies here.

  LoadSDNode *LdNode = dyn_cast<LoadSDNode>(N);

  if (OrigType != EltVT &&
      (LdNode || (OrigType.isFloatingPoint() && EltVT.isFloatingPoint()))) {
    // We have an extending-load. The instruction we selected operates on the
    // smaller type, but the SDNode we are replacing has the larger type. We
    // need to emit a CVT to make the types match.
    unsigned CvtOpc =
        GetConvertOpcode(OrigType.getSimpleVT(), EltVT.getSimpleVT(), LdNode);

    // For each output value, apply the manual sign/zero-extension and make sure
    // all users of the load go through that CVT.
    for (unsigned i = 0; i != NumElts; ++i) {
      SDValue Res(LD, i);
      SDValue OrigVal(N, i);

      SDNode *CvtNode =
        CurDAG->getMachineNode(CvtOpc, DL, OrigType, Res,
                               CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE,
                                                         DL, MVT::i32));
      ReplaceUses(OrigVal, SDValue(CvtNode, 0));
    }
  }

  ReplaceNode(N, LD);
  return true;
}

bool NVPTXDAGToDAGISel::tryStore(SDNode *N) {
  MemSDNode *ST = cast<MemSDNode>(N);
  assert(ST->writeMem() && "Expected store");
  StoreSDNode *PlainStore = dyn_cast<StoreSDNode>(N);
  AtomicSDNode *AtomicStore = dyn_cast<AtomicSDNode>(N);
  assert((PlainStore || AtomicStore) && "Expected store");

  // do not support pre/post inc/dec
  if (PlainStore && PlainStore->isIndexed())
    return false;

  EVT StoreVT = ST->getMemoryVT();
  if (!StoreVT.isSimple())
    return false;

  // Address Space Setting
  unsigned int CodeAddrSpace = getCodeAddrSpace(ST);
  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(ST->getAddressSpace());

  SDLoc DL(N);
  SDValue Chain = ST->getChain();
  auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, ST);

  // Vector Setting
  MVT SimpleVT = StoreVT.getSimpleVT();
  unsigned VecType = NVPTX::PTXLdStInstCode::Scalar;

  // Type Setting: toType + toTypeWidth
  // - for integer type, always use 'u'
  MVT ScalarVT = SimpleVT.getScalarType();
  unsigned ToTypeWidth = ScalarVT.getSizeInBits();
  if (SimpleVT.isVector()) {
    assert((Isv2x16VT(StoreVT) || StoreVT == MVT::v4i8) &&
           "Unexpected vector type");
    // v2x16 is stored using st.b32
    ToTypeWidth = 32;
  }

  unsigned int ToType = getLdStRegType(ScalarVT);

  // Create the machine instruction DAG
  SDValue Value = PlainStore ? PlainStore->getValue() : AtomicStore->getVal();
  SDValue BasePtr = ST->getBasePtr();
  SDValue Addr;
  SDValue Offset, Base;
  std::optional<unsigned> Opcode;
  MVT::SimpleValueType SourceVT =
      Value.getNode()->getSimpleValueType(0).SimpleTy;

  SmallVector<SDValue, 12> Ops(
      {Value, getI32Imm(Ordering, DL), getI32Imm(Scope, DL),
       getI32Imm(CodeAddrSpace, DL), getI32Imm(VecType, DL),
       getI32Imm(ToType, DL), getI32Imm(ToTypeWidth, DL)});

  if (SelectDirectAddr(BasePtr, Addr)) {
    Opcode = pickOpcodeForVT(SourceVT, NVPTX::ST_i8_avar, NVPTX::ST_i16_avar,
                             NVPTX::ST_i32_avar, NVPTX::ST_i64_avar,
                             NVPTX::ST_f32_avar, NVPTX::ST_f64_avar);
    if (!Opcode)
      return false;
    Ops.append({Addr, Chain});
  } else if (PointerSize == 64
                 ? SelectADDRsi64(BasePtr.getNode(), BasePtr, Base, Offset)
                 : SelectADDRsi(BasePtr.getNode(), BasePtr, Base, Offset)) {
    Opcode = pickOpcodeForVT(SourceVT, NVPTX::ST_i8_asi, NVPTX::ST_i16_asi,
                             NVPTX::ST_i32_asi, NVPTX::ST_i64_asi,
                             NVPTX::ST_f32_asi, NVPTX::ST_f64_asi);
    if (!Opcode)
      return false;
    Ops.append({Base, Offset, Chain});
  } else if (PointerSize == 64
                 ? SelectADDRri64(BasePtr.getNode(), BasePtr, Base, Offset)
                 : SelectADDRri(BasePtr.getNode(), BasePtr, Base, Offset)) {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(SourceVT, NVPTX::ST_i8_ari_64, NVPTX::ST_i16_ari_64,
                          NVPTX::ST_i32_ari_64, NVPTX::ST_i64_ari_64,
                          NVPTX::ST_f32_ari_64, NVPTX::ST_f64_ari_64);
    else
      Opcode = pickOpcodeForVT(SourceVT, NVPTX::ST_i8_ari, NVPTX::ST_i16_ari,
                               NVPTX::ST_i32_ari, NVPTX::ST_i64_ari,
                               NVPTX::ST_f32_ari, NVPTX::ST_f64_ari);
    if (!Opcode)
      return false;
    Ops.append({Base, Offset, Chain});
  } else {
    if (PointerSize == 64)
      Opcode =
          pickOpcodeForVT(SourceVT, NVPTX::ST_i8_areg_64, NVPTX::ST_i16_areg_64,
                          NVPTX::ST_i32_areg_64, NVPTX::ST_i64_areg_64,
                          NVPTX::ST_f32_areg_64, NVPTX::ST_f64_areg_64);
    else
      Opcode = pickOpcodeForVT(SourceVT, NVPTX::ST_i8_areg, NVPTX::ST_i16_areg,
                               NVPTX::ST_i32_areg, NVPTX::ST_i64_areg,
                               NVPTX::ST_f32_areg, NVPTX::ST_f64_areg);
    if (!Opcode)
      return false;
    Ops.append({BasePtr, Chain});
  }

  SDNode *NVPTXST = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, Ops);

  if (!NVPTXST)
    return false;

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(NVPTXST), {MemRef});
  ReplaceNode(N, NVPTXST);
  return true;
}

bool NVPTXDAGToDAGISel::tryStoreVector(SDNode *N) {
  SDValue Op1 = N->getOperand(1);
  SDValue Addr, Offset, Base;
  std::optional<unsigned> Opcode;
  SDNode *ST;
  EVT EltVT = Op1.getValueType();
  MemSDNode *MemSD = cast<MemSDNode>(N);
  EVT StoreVT = MemSD->getMemoryVT();

  // Address Space Setting
  unsigned CodeAddrSpace = getCodeAddrSpace(MemSD);
  if (CodeAddrSpace == NVPTX::AddressSpace::Const) {
    report_fatal_error("Cannot store to pointer that points to constant "
                       "memory space");
  }
  unsigned int PointerSize =
      CurDAG->getDataLayout().getPointerSizeInBits(MemSD->getAddressSpace());

  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  auto [Ordering, Scope] = insertMemoryInstructionFence(DL, Chain, MemSD);

  // Type Setting: toType + toTypeWidth
  // - for integer type, always use 'u'
  assert(StoreVT.isSimple() && "Store value is not simple");
  MVT ScalarVT = StoreVT.getSimpleVT().getScalarType();
  unsigned ToTypeWidth = ScalarVT.getSizeInBits();
  unsigned ToType = getLdStRegType(ScalarVT);

  SmallVector<SDValue, 12> Ops;
  SDValue N2;
  unsigned VecType;

  switch (N->getOpcode()) {
  case NVPTXISD::StoreV2:
    VecType = NVPTX::PTXLdStInstCode::V2;
    Ops.append({N->getOperand(1), N->getOperand(2)});
    N2 = N->getOperand(3);
    break;
  case NVPTXISD::StoreV4:
    VecType = NVPTX::PTXLdStInstCode::V4;
    Ops.append({N->getOperand(1), N->getOperand(2), N->getOperand(3),
                N->getOperand(4)});
    N2 = N->getOperand(5);
    break;
  default:
    return false;
  }

  if (isVectorElementTypeUpsized(EltVT)) {
    EltVT = MVT::i32;
    ToType = NVPTX::PTXLdStInstCode::Untyped;
    ToTypeWidth = 32;
  }

  Ops.append({getI32Imm(Ordering, DL), getI32Imm(Scope, DL),
              getI32Imm(CodeAddrSpace, DL), getI32Imm(VecType, DL),
              getI32Imm(ToType, DL), getI32Imm(ToTypeWidth, DL)});

  if (SelectDirectAddr(N2, Addr)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case NVPTXISD::StoreV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::STV_i8_v2_avar, NVPTX::STV_i16_v2_avar,
                               NVPTX::STV_i32_v2_avar, NVPTX::STV_i64_v2_avar,
                               NVPTX::STV_f32_v2_avar, NVPTX::STV_f64_v2_avar);
      break;
    case NVPTXISD::StoreV4:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::STV_i8_v4_avar, NVPTX::STV_i16_v4_avar,
                               NVPTX::STV_i32_v4_avar, std::nullopt,
                               NVPTX::STV_f32_v4_avar, std::nullopt);
      break;
    }
    Ops.push_back(Addr);
  } else if (PointerSize == 64 ? SelectADDRsi64(N2.getNode(), N2, Base, Offset)
                               : SelectADDRsi(N2.getNode(), N2, Base, Offset)) {
    switch (N->getOpcode()) {
    default:
      return false;
    case NVPTXISD::StoreV2:
      Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                               NVPTX::STV_i8_v2_asi, NVPTX::STV_i16_v2_asi,
                               NVPTX::STV_i32_v2_asi, NVPTX::STV_i64_v2_asi,
                               NVPTX::STV_f32_v2_asi, NVPTX::STV_f64_v2_asi);
      break;
    case NVPTXISD::StoreV4:
      Opcode =
          pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::STV_i8_v4_asi,
                          NVPTX::STV_i16_v4_asi, NVPTX::STV_i32_v4_asi,
                          std::nullopt, NVPTX::STV_f32_v4_asi, std::nullopt);
      break;
    }
    Ops.append({Base, Offset});
  } else if (PointerSize == 64 ? SelectADDRri64(N2.getNode(), N2, Base, Offset)
                               : SelectADDRri(N2.getNode(), N2, Base, Offset)) {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::StoreV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                            NVPTX::STV_i8_v2_ari_64, NVPTX::STV_i16_v2_ari_64,
                            NVPTX::STV_i32_v2_ari_64, NVPTX::STV_i64_v2_ari_64,
                            NVPTX::STV_f32_v2_ari_64, NVPTX::STV_f64_v2_ari_64);
        break;
      case NVPTXISD::StoreV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::STV_i8_v4_ari_64,
            NVPTX::STV_i16_v4_ari_64, NVPTX::STV_i32_v4_ari_64, std::nullopt,
            NVPTX::STV_f32_v4_ari_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::StoreV2:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::STV_i8_v2_ari, NVPTX::STV_i16_v2_ari,
                                 NVPTX::STV_i32_v2_ari, NVPTX::STV_i64_v2_ari,
                                 NVPTX::STV_f32_v2_ari, NVPTX::STV_f64_v2_ari);
        break;
      case NVPTXISD::StoreV4:
        Opcode = pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy,
                                 NVPTX::STV_i8_v4_ari, NVPTX::STV_i16_v4_ari,
                                 NVPTX::STV_i32_v4_ari, std::nullopt,
                                 NVPTX::STV_f32_v4_ari, std::nullopt);
        break;
      }
    }
    Ops.append({Base, Offset});
  } else {
    if (PointerSize == 64) {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::StoreV2:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::STV_i8_v2_areg_64,
            NVPTX::STV_i16_v2_areg_64, NVPTX::STV_i32_v2_areg_64,
            NVPTX::STV_i64_v2_areg_64, NVPTX::STV_f32_v2_areg_64,
            NVPTX::STV_f64_v2_areg_64);
        break;
      case NVPTXISD::StoreV4:
        Opcode = pickOpcodeForVT(
            EltVT.getSimpleVT().SimpleTy, NVPTX::STV_i8_v4_areg_64,
            NVPTX::STV_i16_v4_areg_64, NVPTX::STV_i32_v4_areg_64, std::nullopt,
            NVPTX::STV_f32_v4_areg_64, std::nullopt);
        break;
      }
    } else {
      switch (N->getOpcode()) {
      default:
        return false;
      case NVPTXISD::StoreV2:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::STV_i8_v2_areg,
                            NVPTX::STV_i16_v2_areg, NVPTX::STV_i32_v2_areg,
                            NVPTX::STV_i64_v2_areg, NVPTX::STV_f32_v2_areg,
                            NVPTX::STV_f64_v2_areg);
        break;
      case NVPTXISD::StoreV4:
        Opcode =
            pickOpcodeForVT(EltVT.getSimpleVT().SimpleTy, NVPTX::STV_i8_v4_areg,
                            NVPTX::STV_i16_v4_areg, NVPTX::STV_i32_v4_areg,
                            std::nullopt, NVPTX::STV_f32_v4_areg, std::nullopt);
        break;
      }
    }
    Ops.push_back(N2);
  }

  if (!Opcode)
    return false;

  Ops.push_back(Chain);

  ST = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, Ops);

  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(ST), {MemRef});

  ReplaceNode(N, ST);
  return true;
}

bool NVPTXDAGToDAGISel::tryLoadParam(SDNode *Node) {
  SDValue Chain = Node->getOperand(0);
  SDValue Offset = Node->getOperand(2);
  SDValue Glue = Node->getOperand(3);
  SDLoc DL(Node);
  MemSDNode *Mem = cast<MemSDNode>(Node);

  unsigned VecSize;
  switch (Node->getOpcode()) {
  default:
    return false;
  case NVPTXISD::LoadParam:
    VecSize = 1;
    break;
  case NVPTXISD::LoadParamV2:
    VecSize = 2;
    break;
  case NVPTXISD::LoadParamV4:
    VecSize = 4;
    break;
  }

  EVT EltVT = Node->getValueType(0);
  EVT MemVT = Mem->getMemoryVT();

  std::optional<unsigned> Opcode;

  switch (VecSize) {
  default:
    return false;
  case 1:
    Opcode = pickOpcodeForVT(MemVT.getSimpleVT().SimpleTy,
                             NVPTX::LoadParamMemI8, NVPTX::LoadParamMemI16,
                             NVPTX::LoadParamMemI32, NVPTX::LoadParamMemI64,
                             NVPTX::LoadParamMemF32, NVPTX::LoadParamMemF64);
    break;
  case 2:
    Opcode =
        pickOpcodeForVT(MemVT.getSimpleVT().SimpleTy, NVPTX::LoadParamMemV2I8,
                        NVPTX::LoadParamMemV2I16, NVPTX::LoadParamMemV2I32,
                        NVPTX::LoadParamMemV2I64, NVPTX::LoadParamMemV2F32,
                        NVPTX::LoadParamMemV2F64);
    break;
  case 4:
    Opcode =
        pickOpcodeForVT(MemVT.getSimpleVT().SimpleTy, NVPTX::LoadParamMemV4I8,
                        NVPTX::LoadParamMemV4I16, NVPTX::LoadParamMemV4I32,
                        std::nullopt, NVPTX::LoadParamMemV4F32, std::nullopt);
    break;
  }
  if (!Opcode)
    return false;

  SDVTList VTs;
  if (VecSize == 1) {
    VTs = CurDAG->getVTList(EltVT, MVT::Other, MVT::Glue);
  } else if (VecSize == 2) {
    VTs = CurDAG->getVTList(EltVT, EltVT, MVT::Other, MVT::Glue);
  } else {
    EVT EVTs[] = { EltVT, EltVT, EltVT, EltVT, MVT::Other, MVT::Glue };
    VTs = CurDAG->getVTList(EVTs);
  }

  unsigned OffsetVal = Offset->getAsZExtVal();

  SmallVector<SDValue, 2> Ops(
      {CurDAG->getTargetConstant(OffsetVal, DL, MVT::i32), Chain, Glue});

  ReplaceNode(Node, CurDAG->getMachineNode(*Opcode, DL, VTs, Ops));
  return true;
}

bool NVPTXDAGToDAGISel::tryStoreRetval(SDNode *N) {
  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  SDValue Offset = N->getOperand(1);
  unsigned OffsetVal = Offset->getAsZExtVal();
  MemSDNode *Mem = cast<MemSDNode>(N);

  // How many elements do we have?
  unsigned NumElts = 1;
  switch (N->getOpcode()) {
  default:
    return false;
  case NVPTXISD::StoreRetval:
    NumElts = 1;
    break;
  case NVPTXISD::StoreRetvalV2:
    NumElts = 2;
    break;
  case NVPTXISD::StoreRetvalV4:
    NumElts = 4;
    break;
  }

  // Build vector of operands
  SmallVector<SDValue, 6> Ops;
  for (unsigned i = 0; i < NumElts; ++i)
    Ops.push_back(N->getOperand(i + 2));
  Ops.append({CurDAG->getTargetConstant(OffsetVal, DL, MVT::i32), Chain});

  // Determine target opcode
  // If we have an i1, use an 8-bit store. The lowering code in
  // NVPTXISelLowering will have already emitted an upcast.
  std::optional<unsigned> Opcode = 0;
  switch (NumElts) {
  default:
    return false;
  case 1:
    Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                             NVPTX::StoreRetvalI8, NVPTX::StoreRetvalI16,
                             NVPTX::StoreRetvalI32, NVPTX::StoreRetvalI64,
                             NVPTX::StoreRetvalF32, NVPTX::StoreRetvalF64);
    if (Opcode == NVPTX::StoreRetvalI8) {
      // Fine tune the opcode depending on the size of the operand.
      // This helps to avoid creating redundant COPY instructions in
      // InstrEmitter::AddRegisterOperand().
      switch (Ops[0].getSimpleValueType().SimpleTy) {
      default:
        break;
      case MVT::i32:
        Opcode = NVPTX::StoreRetvalI8TruncI32;
        break;
      case MVT::i64:
        Opcode = NVPTX::StoreRetvalI8TruncI64;
        break;
      }
    }
    break;
  case 2:
    Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                             NVPTX::StoreRetvalV2I8, NVPTX::StoreRetvalV2I16,
                             NVPTX::StoreRetvalV2I32, NVPTX::StoreRetvalV2I64,
                             NVPTX::StoreRetvalV2F32, NVPTX::StoreRetvalV2F64);
    break;
  case 4:
    Opcode = pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                             NVPTX::StoreRetvalV4I8, NVPTX::StoreRetvalV4I16,
                             NVPTX::StoreRetvalV4I32, std::nullopt,
                             NVPTX::StoreRetvalV4F32, std::nullopt);
    break;
  }
  if (!Opcode)
    return false;

  SDNode *Ret = CurDAG->getMachineNode(*Opcode, DL, MVT::Other, Ops);
  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Ret), {MemRef});

  ReplaceNode(N, Ret);
  return true;
}

// Helpers for constructing opcode (ex: NVPTX::StoreParamV4F32_iiri)
#define getOpcV2H(ty, opKind0, opKind1)                                        \
  NVPTX::StoreParamV2##ty##_##opKind0##opKind1

#define getOpcV2H1(ty, opKind0, isImm1)                                        \
  (isImm1) ? getOpcV2H(ty, opKind0, i) : getOpcV2H(ty, opKind0, r)

#define getOpcodeForVectorStParamV2(ty, isimm)                                 \
  (isimm[0]) ? getOpcV2H1(ty, i, isimm[1]) : getOpcV2H1(ty, r, isimm[1])

#define getOpcV4H(ty, opKind0, opKind1, opKind2, opKind3)                      \
  NVPTX::StoreParamV4##ty##_##opKind0##opKind1##opKind2##opKind3

#define getOpcV4H3(ty, opKind0, opKind1, opKind2, isImm3)                      \
  (isImm3) ? getOpcV4H(ty, opKind0, opKind1, opKind2, i)                       \
           : getOpcV4H(ty, opKind0, opKind1, opKind2, r)

#define getOpcV4H2(ty, opKind0, opKind1, isImm2, isImm3)                       \
  (isImm2) ? getOpcV4H3(ty, opKind0, opKind1, i, isImm3)                       \
           : getOpcV4H3(ty, opKind0, opKind1, r, isImm3)

#define getOpcV4H1(ty, opKind0, isImm1, isImm2, isImm3)                        \
  (isImm1) ? getOpcV4H2(ty, opKind0, i, isImm2, isImm3)                        \
           : getOpcV4H2(ty, opKind0, r, isImm2, isImm3)

#define getOpcodeForVectorStParamV4(ty, isimm)                                 \
  (isimm[0]) ? getOpcV4H1(ty, i, isimm[1], isimm[2], isimm[3])                 \
             : getOpcV4H1(ty, r, isimm[1], isimm[2], isimm[3])

#define getOpcodeForVectorStParam(n, ty, isimm)                                \
  (n == 2) ? getOpcodeForVectorStParamV2(ty, isimm)                            \
           : getOpcodeForVectorStParamV4(ty, isimm)

static unsigned pickOpcodeForVectorStParam(SmallVector<SDValue, 8> &Ops,
                                           unsigned NumElts,
                                           MVT::SimpleValueType MemTy,
                                           SelectionDAG *CurDAG, SDLoc DL) {
  // Determine which inputs are registers and immediates make new operators
  // with constant values
  SmallVector<bool, 4> IsImm(NumElts, false);
  for (unsigned i = 0; i < NumElts; i++) {
    IsImm[i] = (isa<ConstantSDNode>(Ops[i]) || isa<ConstantFPSDNode>(Ops[i]));
    if (IsImm[i]) {
      SDValue Imm = Ops[i];
      if (MemTy == MVT::f32 || MemTy == MVT::f64) {
        const ConstantFPSDNode *ConstImm = cast<ConstantFPSDNode>(Imm);
        const ConstantFP *CF = ConstImm->getConstantFPValue();
        Imm = CurDAG->getTargetConstantFP(*CF, DL, Imm->getValueType(0));
      } else {
        const ConstantSDNode *ConstImm = cast<ConstantSDNode>(Imm);
        const ConstantInt *CI = ConstImm->getConstantIntValue();
        Imm = CurDAG->getTargetConstant(*CI, DL, Imm->getValueType(0));
      }
      Ops[i] = Imm;
    }
  }

  // Get opcode for MemTy, size, and register/immediate operand ordering
  switch (MemTy) {
  case MVT::i8:
    return getOpcodeForVectorStParam(NumElts, I8, IsImm);
  case MVT::i16:
    return getOpcodeForVectorStParam(NumElts, I16, IsImm);
  case MVT::i32:
    return getOpcodeForVectorStParam(NumElts, I32, IsImm);
  case MVT::i64:
    assert(NumElts == 2 && "MVT too large for NumElts > 2");
    return getOpcodeForVectorStParamV2(I64, IsImm);
  case MVT::f32:
    return getOpcodeForVectorStParam(NumElts, F32, IsImm);
  case MVT::f64:
    assert(NumElts == 2 && "MVT too large for NumElts > 2");
    return getOpcodeForVectorStParamV2(F64, IsImm);

  // These cases don't support immediates, just use the all register version
  // and generate moves.
  case MVT::i1:
    return (NumElts == 2) ? NVPTX::StoreParamV2I8_rr
                          : NVPTX::StoreParamV4I8_rrrr;
  case MVT::f16:
  case MVT::bf16:
    return (NumElts == 2) ? NVPTX::StoreParamV2I16_rr
                          : NVPTX::StoreParamV4I16_rrrr;
  case MVT::v2f16:
  case MVT::v2bf16:
  case MVT::v2i16:
  case MVT::v4i8:
    return (NumElts == 2) ? NVPTX::StoreParamV2I32_rr
                          : NVPTX::StoreParamV4I32_rrrr;
  default:
    llvm_unreachable("Cannot select st.param for unknown MemTy");
  }
}

bool NVPTXDAGToDAGISel::tryStoreParam(SDNode *N) {
  SDLoc DL(N);
  SDValue Chain = N->getOperand(0);
  SDValue Param = N->getOperand(1);
  unsigned ParamVal = Param->getAsZExtVal();
  SDValue Offset = N->getOperand(2);
  unsigned OffsetVal = Offset->getAsZExtVal();
  MemSDNode *Mem = cast<MemSDNode>(N);
  SDValue Glue = N->getOperand(N->getNumOperands() - 1);

  // How many elements do we have?
  unsigned NumElts;
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Unexpected opcode");
  case NVPTXISD::StoreParamU32:
  case NVPTXISD::StoreParamS32:
  case NVPTXISD::StoreParam:
    NumElts = 1;
    break;
  case NVPTXISD::StoreParamV2:
    NumElts = 2;
    break;
  case NVPTXISD::StoreParamV4:
    NumElts = 4;
    break;
  }

  // Build vector of operands
  SmallVector<SDValue, 8> Ops;
  for (unsigned i = 0; i < NumElts; ++i)
    Ops.push_back(N->getOperand(i + 3));
  Ops.append({CurDAG->getTargetConstant(ParamVal, DL, MVT::i32),
              CurDAG->getTargetConstant(OffsetVal, DL, MVT::i32), Chain, Glue});

  // Determine target opcode
  // If we have an i1, use an 8-bit store. The lowering code in
  // NVPTXISelLowering will have already emitted an upcast.
  std::optional<unsigned> Opcode;
  switch (N->getOpcode()) {
  default:
    switch (NumElts) {
    default:
      llvm_unreachable("Unexpected NumElts");
    case 1: {
      MVT::SimpleValueType MemTy = Mem->getMemoryVT().getSimpleVT().SimpleTy;
      SDValue Imm = Ops[0];
      if (MemTy != MVT::f16 && MemTy != MVT::v2f16 &&
          (isa<ConstantSDNode>(Imm) || isa<ConstantFPSDNode>(Imm))) {
        // Convert immediate to target constant
        if (MemTy == MVT::f32 || MemTy == MVT::f64) {
          const ConstantFPSDNode *ConstImm = cast<ConstantFPSDNode>(Imm);
          const ConstantFP *CF = ConstImm->getConstantFPValue();
          Imm = CurDAG->getTargetConstantFP(*CF, DL, Imm->getValueType(0));
        } else {
          const ConstantSDNode *ConstImm = cast<ConstantSDNode>(Imm);
          const ConstantInt *CI = ConstImm->getConstantIntValue();
          Imm = CurDAG->getTargetConstant(*CI, DL, Imm->getValueType(0));
        }
        Ops[0] = Imm;
        // Use immediate version of store param
        Opcode = pickOpcodeForVT(MemTy, NVPTX::StoreParamI8_i,
                                 NVPTX::StoreParamI16_i, NVPTX::StoreParamI32_i,
                                 NVPTX::StoreParamI64_i, NVPTX::StoreParamF32_i,
                                 NVPTX::StoreParamF64_i);
      } else
        Opcode =
            pickOpcodeForVT(Mem->getMemoryVT().getSimpleVT().SimpleTy,
                            NVPTX::StoreParamI8_r, NVPTX::StoreParamI16_r,
                            NVPTX::StoreParamI32_r, NVPTX::StoreParamI64_r,
                            NVPTX::StoreParamF32_r, NVPTX::StoreParamF64_r);
      if (Opcode == NVPTX::StoreParamI8_r) {
        // Fine tune the opcode depending on the size of the operand.
        // This helps to avoid creating redundant COPY instructions in
        // InstrEmitter::AddRegisterOperand().
        switch (Ops[0].getSimpleValueType().SimpleTy) {
        default:
          break;
        case MVT::i32:
          Opcode = NVPTX::StoreParamI8TruncI32_r;
          break;
        case MVT::i64:
          Opcode = NVPTX::StoreParamI8TruncI64_r;
          break;
        }
      }
      break;
    }
    case 2:
    case 4: {
      MVT::SimpleValueType MemTy = Mem->getMemoryVT().getSimpleVT().SimpleTy;
      Opcode = pickOpcodeForVectorStParam(Ops, NumElts, MemTy, CurDAG, DL);
      break;
    }
    }
    break;
  // Special case: if we have a sign-extend/zero-extend node, insert the
  // conversion instruction first, and use that as the value operand to
  // the selected StoreParam node.
  case NVPTXISD::StoreParamU32: {
    Opcode = NVPTX::StoreParamI32_r;
    SDValue CvtNone = CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE, DL,
                                                MVT::i32);
    SDNode *Cvt = CurDAG->getMachineNode(NVPTX::CVT_u32_u16, DL,
                                         MVT::i32, Ops[0], CvtNone);
    Ops[0] = SDValue(Cvt, 0);
    break;
  }
  case NVPTXISD::StoreParamS32: {
    Opcode = NVPTX::StoreParamI32_r;
    SDValue CvtNone = CurDAG->getTargetConstant(NVPTX::PTXCvtMode::NONE, DL,
                                                MVT::i32);
    SDNode *Cvt = CurDAG->getMachineNode(NVPTX::CVT_s32_s16, DL,
                                         MVT::i32, Ops[0], CvtNone);
    Ops[0] = SDValue(Cvt, 0);
    break;
  }
  }

  SDVTList RetVTs = CurDAG->getVTList(MVT::Other, MVT::Glue);
  SDNode *Ret = CurDAG->getMachineNode(*Opcode, DL, RetVTs, Ops);
  MachineMemOperand *MemRef = cast<MemSDNode>(N)->getMemOperand();
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Ret), {MemRef});

  ReplaceNode(N, Ret);
  return true;
}

/// SelectBFE - Look for instruction sequences that can be made more efficient
/// by using the 'bfe' (bit-field extract) PTX instruction
bool NVPTXDAGToDAGISel::tryBFE(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue Len;
  SDValue Start;
  SDValue Val;
  bool IsSigned = false;

  if (N->getOpcode() == ISD::AND) {
    // Canonicalize the operands
    // We want 'and %val, %mask'
    if (isa<ConstantSDNode>(LHS) && !isa<ConstantSDNode>(RHS)) {
      std::swap(LHS, RHS);
    }

    ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(RHS);
    if (!Mask) {
      // We need a constant mask on the RHS of the AND
      return false;
    }

    // Extract the mask bits
    uint64_t MaskVal = Mask->getZExtValue();
    if (!isMask_64(MaskVal)) {
      // We *could* handle shifted masks here, but doing so would require an
      // 'and' operation to fix up the low-order bits so we would trade
      // shr+and for bfe+and, which has the same throughput
      return false;
    }

    // How many bits are in our mask?
    int64_t NumBits = countr_one(MaskVal);
    Len = CurDAG->getTargetConstant(NumBits, DL, MVT::i32);

    if (LHS.getOpcode() == ISD::SRL || LHS.getOpcode() == ISD::SRA) {
      // We have a 'srl/and' pair, extract the effective start bit and length
      Val = LHS.getNode()->getOperand(0);
      Start = LHS.getNode()->getOperand(1);
      ConstantSDNode *StartConst = dyn_cast<ConstantSDNode>(Start);
      if (StartConst) {
        uint64_t StartVal = StartConst->getZExtValue();
        // How many "good" bits do we have left?  "good" is defined here as bits
        // that exist in the original value, not shifted in.
        int64_t GoodBits = Start.getValueSizeInBits() - StartVal;
        if (NumBits > GoodBits) {
          // Do not handle the case where bits have been shifted in. In theory
          // we could handle this, but the cost is likely higher than just
          // emitting the srl/and pair.
          return false;
        }
        Start = CurDAG->getTargetConstant(StartVal, DL, MVT::i32);
      } else {
        // Do not handle the case where the shift amount (can be zero if no srl
        // was found) is not constant. We could handle this case, but it would
        // require run-time logic that would be more expensive than just
        // emitting the srl/and pair.
        return false;
      }
    } else {
      // Do not handle the case where the LHS of the and is not a shift. While
      // it would be trivial to handle this case, it would just transform
      // 'and' -> 'bfe', but 'and' has higher-throughput.
      return false;
    }
  } else if (N->getOpcode() == ISD::SRL || N->getOpcode() == ISD::SRA) {
    if (LHS->getOpcode() == ISD::AND) {
      ConstantSDNode *ShiftCnst = dyn_cast<ConstantSDNode>(RHS);
      if (!ShiftCnst) {
        // Shift amount must be constant
        return false;
      }

      uint64_t ShiftAmt = ShiftCnst->getZExtValue();

      SDValue AndLHS = LHS->getOperand(0);
      SDValue AndRHS = LHS->getOperand(1);

      // Canonicalize the AND to have the mask on the RHS
      if (isa<ConstantSDNode>(AndLHS)) {
        std::swap(AndLHS, AndRHS);
      }

      ConstantSDNode *MaskCnst = dyn_cast<ConstantSDNode>(AndRHS);
      if (!MaskCnst) {
        // Mask must be constant
        return false;
      }

      uint64_t MaskVal = MaskCnst->getZExtValue();
      uint64_t NumZeros;
      uint64_t NumBits;
      if (isMask_64(MaskVal)) {
        NumZeros = 0;
        // The number of bits in the result bitfield will be the number of
        // trailing ones (the AND) minus the number of bits we shift off
        NumBits = llvm::countr_one(MaskVal) - ShiftAmt;
      } else if (isShiftedMask_64(MaskVal)) {
        NumZeros = llvm::countr_zero(MaskVal);
        unsigned NumOnes = llvm::countr_one(MaskVal >> NumZeros);
        // The number of bits in the result bitfield will be the number of
        // trailing zeros plus the number of set bits in the mask minus the
        // number of bits we shift off
        NumBits = NumZeros + NumOnes - ShiftAmt;
      } else {
        // This is not a mask we can handle
        return false;
      }

      if (ShiftAmt < NumZeros) {
        // Handling this case would require extra logic that would make this
        // transformation non-profitable
        return false;
      }

      Val = AndLHS;
      Start = CurDAG->getTargetConstant(ShiftAmt, DL, MVT::i32);
      Len = CurDAG->getTargetConstant(NumBits, DL, MVT::i32);
    } else if (LHS->getOpcode() == ISD::SHL) {
      // Here, we have a pattern like:
      //
      // (sra (shl val, NN), MM)
      // or
      // (srl (shl val, NN), MM)
      //
      // If MM >= NN, we can efficiently optimize this with bfe
      Val = LHS->getOperand(0);

      SDValue ShlRHS = LHS->getOperand(1);
      ConstantSDNode *ShlCnst = dyn_cast<ConstantSDNode>(ShlRHS);
      if (!ShlCnst) {
        // Shift amount must be constant
        return false;
      }
      uint64_t InnerShiftAmt = ShlCnst->getZExtValue();

      SDValue ShrRHS = RHS;
      ConstantSDNode *ShrCnst = dyn_cast<ConstantSDNode>(ShrRHS);
      if (!ShrCnst) {
        // Shift amount must be constant
        return false;
      }
      uint64_t OuterShiftAmt = ShrCnst->getZExtValue();

      // To avoid extra codegen and be profitable, we need Outer >= Inner
      if (OuterShiftAmt < InnerShiftAmt) {
        return false;
      }

      // If the outer shift is more than the type size, we have no bitfield to
      // extract (since we also check that the inner shift is <= the outer shift
      // then this also implies that the inner shift is < the type size)
      if (OuterShiftAmt >= Val.getValueSizeInBits()) {
        return false;
      }

      Start = CurDAG->getTargetConstant(OuterShiftAmt - InnerShiftAmt, DL,
                                        MVT::i32);
      Len = CurDAG->getTargetConstant(Val.getValueSizeInBits() - OuterShiftAmt,
                                      DL, MVT::i32);

      if (N->getOpcode() == ISD::SRA) {
        // If we have a arithmetic right shift, we need to use the signed bfe
        // variant
        IsSigned = true;
      }
    } else {
      // No can do...
      return false;
    }
  } else {
    // No can do...
    return false;
  }


  unsigned Opc;
  // For the BFE operations we form here from "and" and "srl", always use the
  // unsigned variants.
  if (Val.getValueType() == MVT::i32) {
    if (IsSigned) {
      Opc = NVPTX::BFE_S32rii;
    } else {
      Opc = NVPTX::BFE_U32rii;
    }
  } else if (Val.getValueType() == MVT::i64) {
    if (IsSigned) {
      Opc = NVPTX::BFE_S64rii;
    } else {
      Opc = NVPTX::BFE_U64rii;
    }
  } else {
    // We cannot handle this type
    return false;
  }

  SDValue Ops[] = {
    Val, Start, Len
  };

  ReplaceNode(N, CurDAG->getMachineNode(Opc, DL, N->getVTList(), Ops));
  return true;
}

// Select bf16/bf16v2 FADD, FSUB, FMUL as fma on targets with only fma
bool NVPTXDAGToDAGISel::tryBF16ArithToFMA(SDNode *N) {
  EVT VT = SDValue(N, 0).getValueType();
  if (VT.getScalarType() != MVT::bf16)
    return false;

  const NVPTXSubtarget *STI = TM.getSubtargetImpl();
  if (STI->hasNativeBF16Support(N->getOpcode()))
    return false;

  const bool IsVec = VT.isVector();
  assert(!IsVec || VT.getVectorNumElements() == 2);
  SDLoc DL(N);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SmallVector<SDValue, 3> Operands;
  auto GetConstant = [&](float Value) -> SDValue {
    // BF16 immediates must be legalized to integer register values
    APFloat APF(Value);
    bool LosesInfo;
    APF.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &LosesInfo);
    assert(!LosesInfo);
    if (IsVec) {
      auto API = APF.bitcastToAPInt();
      API = API.concat(API);
      auto Const = CurDAG->getTargetConstant(API, DL, MVT::i32);
      return SDValue(CurDAG->getMachineNode(NVPTX::IMOV32ri, DL, VT, Const), 0);
    }
    auto Const = CurDAG->getTargetConstantFP(APF, DL, VT);
    return SDValue(CurDAG->getMachineNode(NVPTX::BFMOV16ri, DL, VT, Const), 0);
  };

  switch (N->getOpcode()) {
  case ISD::FADD:
    // add(a, b) -> fma(a, 1.0, b)
    Operands = {N0, GetConstant(1.0), N1};
    break;
  case ISD::FSUB:
    // sub(a, b) -> fma(b, -1.0, a)
    Operands = {N1, GetConstant(-1.0), N0};
    break;
  case ISD::FMUL:
    // mul(a, b) -> fma(a, b, -0.0)
    // NOTE: The identity is -0, not 0, because -0 + 0 == 0 for floats
    Operands = {N0, N1, GetConstant(-0.0)};
    break;
  default:
    llvm_unreachable("Unexpected opcode");
  };

  int Opcode = IsVec ? NVPTX::BFMA16x2rrr : NVPTX::BFMA16rrr;
  MachineSDNode *FMA = CurDAG->getMachineNode(Opcode, DL, VT, Operands);
  ReplaceNode(N, FMA);
  return true;
}

static inline bool isAddLike(const SDValue V) {
  return V.getOpcode() == ISD::ADD ||
         (V->getOpcode() == ISD::OR && V->getFlags().hasDisjoint());
}

// SelectDirectAddr - Match a direct address for DAG.
// A direct address could be a globaladdress or externalsymbol.
bool NVPTXDAGToDAGISel::SelectDirectAddr(SDValue N, SDValue &Address) {
  // Return true if TGA or ES.
  if (N.getOpcode() == ISD::TargetGlobalAddress ||
      N.getOpcode() == ISD::TargetExternalSymbol) {
    Address = N;
    return true;
  }
  if (N.getOpcode() == NVPTXISD::Wrapper) {
    Address = N.getOperand(0);
    return true;
  }
  // addrspacecast(MoveParam(arg_symbol) to addrspace(PARAM)) -> arg_symbol
  if (AddrSpaceCastSDNode *CastN = dyn_cast<AddrSpaceCastSDNode>(N)) {
    if (CastN->getSrcAddressSpace() == ADDRESS_SPACE_GENERIC &&
        CastN->getDestAddressSpace() == ADDRESS_SPACE_PARAM &&
        CastN->getOperand(0).getOpcode() == NVPTXISD::MoveParam)
      return SelectDirectAddr(CastN->getOperand(0).getOperand(0), Address);
  }
  return false;
}

// symbol+offset
bool NVPTXDAGToDAGISel::SelectADDRsi_imp(SDNode *OpNode, SDValue Addr,
                                         SDValue &Base, SDValue &Offset,
                                         MVT VT) {
  std::function<std::optional<uint64_t>(SDValue, uint64_t)>
      FindRootAddressAndTotalOffset =
          [&](SDValue Addr,
              uint64_t AccumulatedOffset) -> std::optional<uint64_t> {
    if (isAddLike(Addr)) {
      if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
        SDValue PossibleBaseAddr = Addr.getOperand(0);
        AccumulatedOffset += CN->getZExtValue();
        if (SelectDirectAddr(PossibleBaseAddr, Base))
          return AccumulatedOffset;
        return FindRootAddressAndTotalOffset(PossibleBaseAddr,
                                             AccumulatedOffset);
      }
    }
    return std::nullopt;
  };
  if (auto AccumulatedOffset = FindRootAddressAndTotalOffset(Addr, 0)) {
    Offset = CurDAG->getTargetConstant(*AccumulatedOffset, SDLoc(OpNode), VT);
    return true;
  }
  return false;
}

// symbol+offset
bool NVPTXDAGToDAGISel::SelectADDRsi(SDNode *OpNode, SDValue Addr,
                                     SDValue &Base, SDValue &Offset) {
  return SelectADDRsi_imp(OpNode, Addr, Base, Offset, MVT::i32);
}

// symbol+offset
bool NVPTXDAGToDAGISel::SelectADDRsi64(SDNode *OpNode, SDValue Addr,
                                       SDValue &Base, SDValue &Offset) {
  return SelectADDRsi_imp(OpNode, Addr, Base, Offset, MVT::i64);
}

// register+offset
bool NVPTXDAGToDAGISel::SelectADDRri_imp(SDNode *OpNode, SDValue Addr,
                                         SDValue &Base, SDValue &Offset,
                                         MVT VT) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
    Offset = CurDAG->getTargetConstant(0, SDLoc(OpNode), VT);
    return true;
  }
  if (Addr.getOpcode() == ISD::TargetExternalSymbol ||
      Addr.getOpcode() == ISD::TargetGlobalAddress)
    return false; // direct calls.

  if (isAddLike(Addr)) {
    if (SelectDirectAddr(Addr.getOperand(0), Addr)) {
      return false;
    }
    if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Addr.getOperand(1))) {
      if (FrameIndexSDNode *FIN =
              dyn_cast<FrameIndexSDNode>(Addr.getOperand(0)))
        // Constant offset from frame ref.
        Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), VT);
      else
        Base = Addr.getOperand(0);

      // Offset must fit in a 32-bit signed int in PTX [register+offset] address
      // mode
      if (!CN->getAPIntValue().isSignedIntN(32))
        return false;

      Offset = CurDAG->getSignedTargetConstant(CN->getSExtValue(),
                                               SDLoc(OpNode), MVT::i32);
      return true;
    }
  }
  return false;
}

// register+offset
bool NVPTXDAGToDAGISel::SelectADDRri(SDNode *OpNode, SDValue Addr,
                                     SDValue &Base, SDValue &Offset) {
  return SelectADDRri_imp(OpNode, Addr, Base, Offset, MVT::i32);
}

// register+offset
bool NVPTXDAGToDAGISel::SelectADDRri64(SDNode *OpNode, SDValue Addr,
                                       SDValue &Base, SDValue &Offset) {
  return SelectADDRri_imp(OpNode, Addr, Base, Offset, MVT::i64);
}

bool NVPTXDAGToDAGISel::ChkMemSDNodeAddressSpace(SDNode *N,
                                                 unsigned int spN) const {
  const Value *Src = nullptr;
  if (MemSDNode *mN = dyn_cast<MemSDNode>(N)) {
    if (spN == 0 && mN->getMemOperand()->getPseudoValue())
      return true;
    Src = mN->getMemOperand()->getValue();
  }
  if (!Src)
    return false;
  if (auto *PT = dyn_cast<PointerType>(Src->getType()))
    return (PT->getAddressSpace() == spN);
  return false;
}

/// SelectInlineAsmMemoryOperand - Implement addressing mode selection for
/// inline asm expressions.
bool NVPTXDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, InlineAsm::ConstraintCode ConstraintID,
    std::vector<SDValue> &OutOps) {
  SDValue Op0, Op1;
  switch (ConstraintID) {
  default:
    return true;
  case InlineAsm::ConstraintCode::m: // memory
    if (SelectDirectAddr(Op, Op0)) {
      OutOps.push_back(Op0);
      OutOps.push_back(CurDAG->getTargetConstant(0, SDLoc(Op), MVT::i32));
      return false;
    }
    if (SelectADDRri(Op.getNode(), Op, Op0, Op1)) {
      OutOps.push_back(Op0);
      OutOps.push_back(Op1);
      return false;
    }
    break;
  }
  return true;
}

void NVPTXDAGToDAGISel::SelectV2I64toI128(SDNode *N) {
  // Lower a CopyToReg with two 64-bit inputs
  // Dst:i128, lo:i64, hi:i64
  //
  // CopyToReg Dst, lo, hi;
  //
  // ==>
  //
  // tmp = V2I64toI128 {lo, hi};
  // CopyToReg Dst, tmp;
  SDValue Dst = N->getOperand(1);
  SDValue Lo = N->getOperand(2);
  SDValue Hi = N->getOperand(3);

  SDLoc DL(N);
  SDNode *Mov =
      CurDAG->getMachineNode(NVPTX::V2I64toI128, DL, MVT::i128, {Lo, Hi});

  SmallVector<SDValue, 4> NewOps(N->getNumOperands() - 1);
  NewOps[0] = N->getOperand(0);
  NewOps[1] = Dst;
  NewOps[2] = SDValue(Mov, 0);
  if (N->getNumOperands() == 5)
    NewOps[3] = N->getOperand(4);
  SDValue NewValue = CurDAG->getNode(ISD::CopyToReg, DL, SmallVector<EVT>(N->values()), NewOps);

  ReplaceNode(N, NewValue.getNode());
}

void NVPTXDAGToDAGISel::SelectI128toV2I64(SDNode *N) {
  // Lower CopyFromReg from a 128-bit regs to two 64-bit regs
  // Dst:i128, Src:i128
  //
  // {lo, hi} = CopyFromReg Src
  //
  // ==>
  //
  // {lo, hi} = I128toV2I64 Src
  //
  SDValue Ch = N->getOperand(0);
  SDValue Src = N->getOperand(1);
  SDValue Glue = N->getOperand(2);
  SDLoc DL(N);

  // Add Glue and Ch to the operands and results to avoid break the execution
  // order
  SDNode *Mov = CurDAG->getMachineNode(
      NVPTX::I128toV2I64, DL,
      {MVT::i64, MVT::i64, Ch.getValueType(), Glue.getValueType()},
      {Src, Ch, Glue});

  ReplaceNode(N, Mov);
}

void NVPTXDAGToDAGISel::SelectI64ToV2I32(SDNode *N) {
  SDValue Ch = N->getOperand(0);
  SDValue Src = N->getOperand(1);
  SDLoc DL(N);

  SDNode *Mov = CurDAG->getMachineNode(NVPTX::I64toV2I32, DL,
                                       {MVT::i32, MVT::i32, Ch.getValueType()},
                                       {Src, Ch});
  ReplaceNode(N, Mov);
}

/// GetConvertOpcode - Returns the CVT_ instruction opcode that implements a
/// conversion from \p SrcTy to \p DestTy.
unsigned NVPTXDAGToDAGISel::GetConvertOpcode(MVT DestTy, MVT SrcTy,
                                             LoadSDNode *LdNode) {
  bool IsSigned = LdNode && LdNode->getExtensionType() == ISD::SEXTLOAD;
  switch (SrcTy.SimpleTy) {
  default:
    llvm_unreachable("Unhandled source type");
  case MVT::i8:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i16:
      return IsSigned ? NVPTX::CVT_s16_s8 : NVPTX::CVT_u16_u8;
    case MVT::i32:
      return IsSigned ? NVPTX::CVT_s32_s8 : NVPTX::CVT_u32_u8;
    case MVT::i64:
      return IsSigned ? NVPTX::CVT_s64_s8 : NVPTX::CVT_u64_u8;
    }
  case MVT::i16:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i8:
      return IsSigned ? NVPTX::CVT_s8_s16 : NVPTX::CVT_u8_u16;
    case MVT::i32:
      return IsSigned ? NVPTX::CVT_s32_s16 : NVPTX::CVT_u32_u16;
    case MVT::i64:
      return IsSigned ? NVPTX::CVT_s64_s16 : NVPTX::CVT_u64_u16;
    }
  case MVT::i32:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i8:
      return IsSigned ? NVPTX::CVT_s8_s32 : NVPTX::CVT_u8_u32;
    case MVT::i16:
      return IsSigned ? NVPTX::CVT_s16_s32 : NVPTX::CVT_u16_u32;
    case MVT::i64:
      return IsSigned ? NVPTX::CVT_s64_s32 : NVPTX::CVT_u64_u32;
    }
  case MVT::i64:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::i8:
      return IsSigned ? NVPTX::CVT_s8_s64 : NVPTX::CVT_u8_u64;
    case MVT::i16:
      return IsSigned ? NVPTX::CVT_s16_s64 : NVPTX::CVT_u16_u64;
    case MVT::i32:
      return IsSigned ? NVPTX::CVT_s32_s64 : NVPTX::CVT_u32_u64;
    }
  case MVT::f16:
    switch (DestTy.SimpleTy) {
    default:
      llvm_unreachable("Unhandled dest type");
    case MVT::f32:
      return NVPTX::CVT_f32_f16;
    case MVT::f64:
      return NVPTX::CVT_f64_f16;
    }
  }
}

bool NVPTXDAGToDAGISel::tryFence(SDNode *N) {
  SDLoc DL(N);
  assert(N->getOpcode() == ISD::ATOMIC_FENCE);
  unsigned int FenceOp =
      getFenceOp(NVPTX::Ordering(N->getConstantOperandVal(1)),
                 Scopes[N->getConstantOperandVal(2)], Subtarget);
  SDValue Chain = N->getOperand(0);
  SDNode *FenceNode = CurDAG->getMachineNode(FenceOp, DL, MVT::Other, Chain);
  ReplaceNode(N, FenceNode);
  return true;
}

NVPTXScopes::NVPTXScopes(LLVMContext &C) {
  Scopes[C.getOrInsertSyncScopeID("singlethread")] = NVPTX::Scope::Thread;
  Scopes[C.getOrInsertSyncScopeID("")] = NVPTX::Scope::System;
  Scopes[C.getOrInsertSyncScopeID("block")] = NVPTX::Scope::Block;
  Scopes[C.getOrInsertSyncScopeID("cluster")] = NVPTX::Scope::Cluster;
  Scopes[C.getOrInsertSyncScopeID("device")] = NVPTX::Scope::Device;
}

NVPTX::Scope NVPTXScopes::operator[](SyncScope::ID ID) const {
  if (Scopes.empty())
    llvm_unreachable("NVPTX Scopes must be initialized before calling "
                     "NVPTXScopes::operator[]");

  auto S = Scopes.find(ID);
  if (S == Scopes.end()) {
    // TODO:
    // - Add API to LLVMContext to get the name of a single scope.
    // - Use that API here to print an error containing the name
    //   of this Unknown ID.
    report_fatal_error(formatv("Could not find scope ID={}.", int(ID)));
  }
  return S->second;
}

bool NVPTXScopes::empty() const { return Scopes.size() == 0; }

#define CP_ASYNC_BULK_TENSOR_OPCODE(dir, dim, mode, is_s32, suffix)            \
  (is_s32                                                                      \
       ? NVPTX::CP_ASYNC_BULK_TENSOR_##dir##_##dim##_SHARED32_##mode##suffix   \
       : NVPTX::CP_ASYNC_BULK_TENSOR_##dir##_##dim##_##mode##suffix)

#define CP_ASYNC_BULK_TENSOR_OPCODE_S2G_IMPL(op, dim, mode, is_ch, is_s32)     \
  (is_ch ? (CP_ASYNC_BULK_TENSOR_OPCODE(op, dim, mode, is_s32, _CH))           \
         : (CP_ASYNC_BULK_TENSOR_OPCODE(op, dim, mode, is_s32, )))

#define GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(dim, mode, is_reduce, is_ch,       \
                                            is_s32)                            \
  (is_reduce                                                                   \
       ? (CP_ASYNC_BULK_TENSOR_OPCODE_S2G_IMPL(RED, dim, mode, is_ch, is_s32)) \
       : (CP_ASYNC_BULK_TENSOR_OPCODE_S2G_IMPL(S2G, dim, mode, is_ch,          \
                                               is_s32)))

#define GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(dim, mode, is_mc, is_ch, is_s32)   \
  [&]() -> auto {                                                              \
    if (is_mc && is_ch)                                                        \
      return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, _MC_CH);      \
    if (is_ch)                                                                 \
      return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, _CH);         \
    if (is_mc)                                                                 \
      return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, _MC);         \
    return CP_ASYNC_BULK_TENSOR_OPCODE(G2S, dim, mode, is_s32, );              \
  }()

#define GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(dim, mode, is_ch)             \
  (is_ch ? NVPTX::CP_ASYNC_BULK_TENSOR_PREFETCH_##dim##_##mode##_CH            \
         : NVPTX::CP_ASYNC_BULK_TENSOR_PREFETCH_##dim##_##mode)

static unsigned GetCpAsyncBulkTensorS2GOpcode(size_t Dim, bool IsShared32,
                                              bool IsCacheHint, bool IsIm2Col,
                                              bool IsReduce = false) {
  if (IsIm2Col) {
    switch (Dim) {
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(3D, IM2COL, IsReduce,
                                                 IsCacheHint, IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(4D, IM2COL, IsReduce,
                                                 IsCacheHint, IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(5D, IM2COL, IsReduce,
                                                 IsCacheHint, IsShared32);
    default:
      llvm_unreachable("Invalid Dimension in im2col mode for "
                       "GetCpAsyncBulkTensorS2GOpcode.");
    }
  } else {
    switch (Dim) {
    case 1:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(1D, TILE, IsReduce,
                                                 IsCacheHint, IsShared32);
    case 2:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(2D, TILE, IsReduce,
                                                 IsCacheHint, IsShared32);
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(3D, TILE, IsReduce,
                                                 IsCacheHint, IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(4D, TILE, IsReduce,
                                                 IsCacheHint, IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_S2G(5D, TILE, IsReduce,
                                                 IsCacheHint, IsShared32);
    default:
      llvm_unreachable(
          "Invalid Dimension in tile mode for GetCpAsyncBulkTensorS2GOpcode.");
    }
  }
}

static unsigned GetCpAsyncBulkTensorG2SOpcode(size_t Dim, bool IsShared32,
                                              bool IsMultiCast,
                                              bool IsCacheHint, bool IsIm2Col) {
  if (IsIm2Col) {
    switch (Dim) {
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(3D, IM2COL, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(4D, IM2COL, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(5D, IM2COL, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    default:
      llvm_unreachable("Invalid Dimension in im2col mode for "
                       "GetCpAsyncBulkTensorG2SOpcode.");
    }
  } else {
    switch (Dim) {
    case 1:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(1D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 2:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(2D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(3D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(4D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_G2S(5D, TILE, IsMultiCast,
                                                 IsCacheHint, IsShared32);
    default:
      llvm_unreachable(
          "Invalid Dimension in tile mode for GetCpAsyncBulkTensorG2SOpcode.");
    }
  }
}

static unsigned GetCpAsyncBulkTensorPrefetchOpcode(size_t Dim, bool IsCacheHint,
                                                   bool IsIm2Col) {
  if (IsIm2Col) {
    switch (Dim) {
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(3D, IM2COL, IsCacheHint);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(4D, IM2COL, IsCacheHint);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(5D, IM2COL, IsCacheHint);
    default:
      llvm_unreachable("Invalid Dimension in im2col mode for "
                       "GetCpAsyncBulkTensorPrefetchOpcode.");
    }
  } else {
    switch (Dim) {
    case 1:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(1D, TILE, IsCacheHint);
    case 2:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(2D, TILE, IsCacheHint);
    case 3:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(3D, TILE, IsCacheHint);
    case 4:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(4D, TILE, IsCacheHint);
    case 5:
      return GET_CP_ASYNC_BULK_TENSOR_OPCODE_PREFETCH(5D, TILE, IsCacheHint);
    default:
      llvm_unreachable("Invalid Dimension in tile mode for "
                       "GetCpAsyncBulkTensorPrefetchOpcode.");
    }
  }
}

static size_t GetDimsFromIntrinsic(unsigned IID) {
  switch (IID) {
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_3d:
    return 3;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_4d:
    return 4;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_5d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_5d:
    return 5;
  default:
    llvm_unreachable("Invalid im2col intrinsic in GetDimsFromIntrinsic.");
  }
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkTensorG2SCommon(SDNode *N,
                                                         bool IsIm2Col) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // {dst, mbar, src, dims{d0...dN}, im2col_offsets{dims-2}
  // multicast, cache_hint,
  // multicast_flag, cache_hint_flag}
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {7 + dims + im2col_offsets}
  size_t NumOps = N->getNumOperands();
  size_t NumDims = IsIm2Col ? GetDimsFromIntrinsic(N->getConstantOperandVal(1))
                            : (NumOps - 9);
  // Offsets is always 'NumDims - 2' and only for im2col mode
  size_t NumOffsets = IsIm2Col ? (NumDims - 2) : 0;
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  bool IsMultiCast = N->getConstantOperandVal(NumOps - 2) == 1;
  size_t NumBaseArgs = NumDims + NumOffsets + 3; // for {dst, mbar, src}
  size_t MultiCastIdx = NumBaseArgs + 2;         // for Chain and IID

  SDLoc DL(N);
  SmallVector<SDValue, 8> Ops(N->ops().slice(2, NumBaseArgs));

  // Push MultiCast operand, if available
  if (IsMultiCast)
    Ops.push_back(N->getOperand(MultiCastIdx));

  // Push CacheHint operand, if available
  if (IsCacheHint)
    Ops.push_back(N->getOperand(MultiCastIdx + 1));

  // Finally, the chain operand
  Ops.push_back(N->getOperand(0));

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode = GetCpAsyncBulkTensorG2SOpcode(
      NumDims, IsShared32, IsMultiCast, IsCacheHint, IsIm2Col);
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkTensorS2GCommon(SDNode *N,
                                                         bool IsIm2Col) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // src, dst, dims{d0...dN}, cache_hint, cache_hint_flag
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {4 + dims}
  size_t NumOps = N->getNumOperands();
  size_t NumDims = NumOps - 6;
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  size_t NumArgs = NumDims + (IsCacheHint ? 3 : 2); // src, dst, cache_hint

  SDLoc DL(N);
  SmallVector<SDValue, 8> Ops(N->ops().slice(2, NumArgs));
  Ops.push_back(N->getOperand(0)); // Chain operand

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode =
      GetCpAsyncBulkTensorS2GOpcode(NumDims, IsShared32, IsCacheHint, IsIm2Col);
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkTensorPrefetchCommon(SDNode *N,
                                                              bool IsIm2Col) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // {src, dims{d0...dN}, im2col_offsets{dims-2}
  // cache_hint, cache_hint_flag}
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {3 + dims + im2col_offsets}
  size_t NumOps = N->getNumOperands();
  size_t NumDims = IsIm2Col ? GetDimsFromIntrinsic(N->getConstantOperandVal(1))
                            : (NumOps - 5);
  // Offsets is always 'NumDims - 2' and only for im2col mode
  size_t NumOffsets = IsIm2Col ? (NumDims - 2) : 0;
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  size_t NumArgs = NumDims + NumOffsets + (IsCacheHint ? 2 : 1);

  SDLoc DL(N);
  SmallVector<SDValue, 12> Ops(N->ops().slice(2, NumArgs));
  Ops.push_back(N->getOperand(0)); // Chain operand

  unsigned Opcode =
      GetCpAsyncBulkTensorPrefetchOpcode(NumDims, IsCacheHint, IsIm2Col);
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkTensorReduceCommon(SDNode *N,
                                                            unsigned RedOp,
                                                            bool IsIm2Col) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // src, dst, dims{d0...dN}, cache_hint, cache_hint_flag
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {4 + dims}
  size_t NumOps = N->getNumOperands();
  size_t NumDims = NumOps - 6;
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  size_t NumArgs = NumDims + (IsCacheHint ? 3 : 2); // src, dst, cache_hint

  SDLoc DL(N);
  SmallVector<SDValue, 12> Ops(N->ops().slice(2, NumArgs));
  Ops.push_back(getI32Imm(RedOp, DL)); // Reduction Op
  Ops.push_back(N->getOperand(0));     // Chain operand

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode = GetCpAsyncBulkTensorS2GOpcode(
      NumDims, IsShared32, IsCacheHint, IsIm2Col, /*IsReduce=*/true);
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkS2G(SDNode *N) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // dst, src, size, cache_hint, cache_hint_flag
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {5}
  size_t NumOps = N->getNumOperands();
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  size_t NumArgs = IsCacheHint ? 4 : 3; // src, dst, size, cache_hint

  SDLoc DL(N);
  SmallVector<SDValue, 8> Ops(N->ops().slice(2, NumArgs));
  Ops.push_back(N->getOperand(0)); // Chain operand

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode;
  if (IsCacheHint)
    Opcode = IsShared32 ? NVPTX::CP_ASYNC_BULK_S2G_SHARED32_CH
                        : NVPTX::CP_ASYNC_BULK_S2G_CH;
  else
    Opcode = IsShared32 ? NVPTX::CP_ASYNC_BULK_S2G_SHARED32
                        : NVPTX::CP_ASYNC_BULK_S2G;
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkG2S(SDNode *N) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // {dst, mbar, src, size, multicast, cache_hint,
  // multicast_flag, cache_hint_flag}
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {8}
  size_t NumOps = N->getNumOperands();
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  bool IsMultiCast = N->getConstantOperandVal(NumOps - 2) == 1;
  size_t NumBaseArgs = 4;                // dst, mbar, src, size
  size_t MultiCastIdx = NumBaseArgs + 2; // for Chain and IID

  SDLoc DL(N);
  SmallVector<SDValue, 8> Ops(N->ops().slice(2, NumBaseArgs));

  // Push MultiCast operand, if available
  if (IsMultiCast)
    Ops.push_back(N->getOperand(MultiCastIdx));

  // Push CacheHint operand, if available
  if (IsCacheHint)
    Ops.push_back(N->getOperand(MultiCastIdx + 1));

  // Finally, the chain operand
  Ops.push_back(N->getOperand(0));

  bool IsShared32 =
      CurDAG->getDataLayout().getPointerSizeInBits(ADDRESS_SPACE_SHARED) == 32;
  unsigned Opcode = [&]() {
    if (IsMultiCast && IsCacheHint)
      return IsShared32 ? NVPTX::CP_ASYNC_BULK_G2S_SHARED32_MC_CH
                        : NVPTX::CP_ASYNC_BULK_G2S_MC_CH;
    if (IsMultiCast)
      return IsShared32 ? NVPTX::CP_ASYNC_BULK_G2S_SHARED32_MC
                        : NVPTX::CP_ASYNC_BULK_G2S_MC;
    if (IsCacheHint)
      return IsShared32 ? NVPTX::CP_ASYNC_BULK_G2S_SHARED32_CH
                        : NVPTX::CP_ASYNC_BULK_G2S_CH;
    return IsShared32 ? NVPTX::CP_ASYNC_BULK_G2S_SHARED32
                      : NVPTX::CP_ASYNC_BULK_G2S;
  }();
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

void NVPTXDAGToDAGISel::SelectCpAsyncBulkPrefetchL2(SDNode *N) {
  // We have {Chain, Intrinsic-ID} followed by the actual intrisic args:
  // src, size, cache_hint, cache_hint_flag
  // NumOperands = {Chain, IID} + {Actual intrinsic args}
  //             = {2}          + {4}
  size_t NumOps = N->getNumOperands();
  bool IsCacheHint = N->getConstantOperandVal(NumOps - 1) == 1;
  size_t NumArgs = IsCacheHint ? 3 : 2; // src, size, cache_hint

  SDLoc DL(N);
  SmallVector<SDValue, 4> Ops(N->ops().slice(2, NumArgs));
  Ops.push_back(N->getOperand(0)); // Chain operand
  
  unsigned Opcode = IsCacheHint 
  ?  NVPTX::CP_ASYNC_BULK_PREFETCH_CH
  :  NVPTX::CP_ASYNC_BULK_PREFETCH;
  ReplaceNode(N, CurDAG->getMachineNode(Opcode, DL, N->getVTList(), Ops));
}

bool NVPTXDAGToDAGISel::tryIntrinsicVoid(SDNode *N) {
  unsigned IID = N->getConstantOperandVal(1);
  using TMARedTy = llvm::nvvm::TMAReductionOp;
  auto CastTy = [](TMARedTy Op) { return static_cast<unsigned>(Op); };
  switch (IID) {
  default:
    return false;
  case Intrinsic::nvvm_cp_async_bulk_global_to_shared_cluster:
    SelectCpAsyncBulkG2S(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_shared_cta_to_global:
    SelectCpAsyncBulkS2G(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_prefetch_L2:
    SelectCpAsyncBulkPrefetchL2(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_tile_5d:
    SelectCpAsyncBulkTensorS2GCommon(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_s2g_im2col_5d:
    SelectCpAsyncBulkTensorS2GCommon(N, /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_tile_5d:
    SelectCpAsyncBulkTensorG2SCommon(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_g2s_im2col_5d:
    SelectCpAsyncBulkTensorG2SCommon(N, /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_tile_5d:
    SelectCpAsyncBulkTensorPrefetchCommon(N);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_prefetch_im2col_5d:
    SelectCpAsyncBulkTensorPrefetchCommon(N, /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::ADD));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_add_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::ADD),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MIN));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_min_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MIN),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MAX));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_max_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::MAX),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::INC));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_inc_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::INC),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::DEC));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_dec_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::DEC),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::AND));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_and_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::AND),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::OR));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_or_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::OR),
                                        /*IsIm2Col=*/true);
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_1d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_2d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_tile_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::XOR));
    return true;
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_3d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_4d:
  case Intrinsic::nvvm_cp_async_bulk_tensor_reduce_xor_im2col_5d:
    SelectCpAsyncBulkTensorReduceCommon(N, CastTy(TMARedTy::XOR),
                                        /*IsIm2Col=*/true);
    return true;
  }
}
