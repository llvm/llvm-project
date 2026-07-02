//===- SelectionDAGDumper.cpp - Implement SelectionDAG::dump() ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAG::dump method and friends.
//
//===----------------------------------------------------------------------===//

#include "SDNodeDbgValue.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/ModuleSlotTracker.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Printable.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>

using namespace llvm;

static cl::opt<bool>
VerboseDAGDumping("dag-dump-verbose", cl::Hidden,
                  cl::desc("Display more information when dumping selection "
                           "DAG nodes."));

static cl::opt<bool>
    PrintSDNodeAddrs("print-sdnode-addrs", cl::Hidden,
                     cl::desc("Print addresses of SDNodes when dumping"));

namespace {

// Keep the names in one byte blob so the constant table has no pointers or
// relocations. Dense integer tables map built-in opcodes to the blob.
struct FixedOperationNameData {
#define DAG_NODE_NAME(OPCODE, NAME) char OPCODE[sizeof(NAME)];
#include "llvm/CodeGen/SelectionDAGOperationNames.def"
#undef DAG_NODE_NAME
};

constexpr FixedOperationNameData FixedOperationNames = {
#define DAG_NODE_NAME(OPCODE, NAME) NAME,
#include "llvm/CodeGen/SelectionDAGOperationNames.def"
#undef DAG_NODE_NAME
};

static_assert(sizeof(FixedOperationNames) <=
              std::numeric_limits<uint16_t>::max());
static_assert(alignof(FixedOperationNameData) == alignof(char));
static_assert(std::is_standard_layout_v<FixedOperationNameData>);

#define DAG_NODE_NAME(OPCODE, NAME)                                            \
  static_assert(ISD::OPCODE < ISD::BUILTIN_OP_END);                            \
  static_assert(sizeof(NAME) - 1 <= std::numeric_limits<uint8_t>::max());
#include "llvm/CodeGen/SelectionDAGOperationNames.def"
#undef DAG_NODE_NAME

template <typename ValueType, bool StoreLengths>
constexpr std::array<ValueType, ISD::BUILTIN_OP_END>
makeFixedOperationNameTable() {
  std::array<ValueType, ISD::BUILTIN_OP_END> Result{};
#define DAG_NODE_NAME(OPCODE, NAME)                                            \
  Result[ISD::OPCODE] = static_cast<ValueType>(                                \
      StoreLengths ? sizeof(NAME) - 1                                          \
                   : offsetof(FixedOperationNameData, OPCODE));
#include "llvm/CodeGen/SelectionDAGOperationNames.def"
#undef DAG_NODE_NAME
  return Result;
}

constexpr auto FixedOperationNameOffsets =
    makeFixedOperationNameTable<uint16_t, false>();
constexpr auto FixedOperationNameLengths =
    makeFixedOperationNameTable<uint8_t, true>();
static_assert(sizeof(FixedOperationNameOffsets) ==
              ISD::BUILTIN_OP_END * sizeof(uint16_t));
static_assert(sizeof(FixedOperationNameLengths) ==
              ISD::BUILTIN_OP_END * sizeof(uint8_t));

} // namespace

std::string SDNode::getOperationName(const SelectionDAG *G) const {
  const unsigned Opcode = getOpcode();
  if (Opcode < FixedOperationNameLengths.size()) {
    const uint8_t Length = FixedOperationNameLengths[Opcode];
    if (Length != 0) {
      const char *Names = reinterpret_cast<const char *>(&FixedOperationNames);
      return std::string(Names + FixedOperationNameOffsets[Opcode], Length);
    }
  }

  switch (Opcode) {
  default:
    if (Opcode < ISD::BUILTIN_OP_END)
      return "<<Unknown DAG Node>>";
    if (isMachineOpcode()) {
      if (G)
        if (const TargetInstrInfo *TII = G->getSubtarget().getInstrInfo())
          if (getMachineOpcode() < TII->getNumOpcodes())
            return std::string(TII->getName(getMachineOpcode()));
      return "<<Unknown Machine Node #" + utostr(Opcode) + ">>";
    }
    if (G) {
      const SelectionDAGTargetInfo &TSI = G->getSelectionDAGInfo();
      if (const char *Name = TSI.getTargetNodeName(Opcode))
        return Name;
      const TargetLowering &TLI = G->getTargetLoweringInfo();
      const char *Name = TLI.getTargetNodeName(Opcode);
      if (Name)
        return Name;
      return "<<Unknown Target Node #" + utostr(Opcode) + ">>";
    }
    return "<<Unknown Node #" + utostr(Opcode) + ">>";

#ifndef NDEBUG
  case ISD::DELETED_NODE:
    return "<<Deleted Node!>>";
#endif

  case ISD::Constant:
    if (cast<ConstantSDNode>(this)->isOpaque())
      return "OpaqueConstant";
    return "Constant";
  case ISD::INTRINSIC_WO_CHAIN:
  case ISD::INTRINSIC_VOID:
  case ISD::INTRINSIC_W_CHAIN: {
    unsigned OpNo = getOpcode() == ISD::INTRINSIC_WO_CHAIN ? 0 : 1;
    unsigned IID = getOperand(OpNo)->getAsZExtVal();
    if (IID < Intrinsic::num_intrinsics)
      return Intrinsic::getBaseName((Intrinsic::ID)IID).str();
    if (!G)
      return "Unknown intrinsic";
    llvm_unreachable("Invalid intrinsic ID");
  }

  case ISD::TargetConstant:
    if (cast<ConstantSDNode>(this)->isOpaque())
      return "OpaqueTargetConstant";
    return "TargetConstant";

  case ISD::CONDCODE:
    switch (cast<CondCodeSDNode>(this)->get()) {
    default:
      llvm_unreachable("Unknown setcc condition!");
    case ISD::SETOEQ:
      return "setoeq";
    case ISD::SETOGT:
      return "setogt";
    case ISD::SETOGE:
      return "setoge";
    case ISD::SETOLT:
      return "setolt";
    case ISD::SETOLE:
      return "setole";
    case ISD::SETONE:
      return "setone";

    case ISD::SETO:
      return "seto";
    case ISD::SETUO:
      return "setuo";
    case ISD::SETUEQ:
      return "setueq";
    case ISD::SETUGT:
      return "setugt";
    case ISD::SETUGE:
      return "setuge";
    case ISD::SETULT:
      return "setult";
    case ISD::SETULE:
      return "setule";
    case ISD::SETUNE:
      return "setune";

    case ISD::SETEQ:
      return "seteq";
    case ISD::SETGT:
      return "setgt";
    case ISD::SETGE:
      return "setge";
    case ISD::SETLT:
      return "setlt";
    case ISD::SETLE:
      return "setle";
    case ISD::SETNE:
      return "setne";

    case ISD::SETTRUE:
      return "settrue";
    case ISD::SETTRUE2:
      return "settrue2";
    case ISD::SETFALSE:
      return "setfalse";
    case ISD::SETFALSE2:
      return "setfalse2";
    }

    // Vector Predication
#define BEGIN_REGISTER_VP_SDNODE(SDID, LEGALARG, NAME, ...)                    \
  case ISD::SDID:                                                              \
    return #NAME;
#include "llvm/IR/VPIntrinsics.def"
  }
}

const char *SDNode::getIndexedModeName(ISD::MemIndexedMode AM) {
  switch (AM) {
  default:              return "";
  case ISD::PRE_INC:    return "<pre-inc>";
  case ISD::PRE_DEC:    return "<pre-dec>";
  case ISD::POST_INC:   return "<post-inc>";
  case ISD::POST_DEC:   return "<post-dec>";
  }
}

static Printable PrintNodeId(const SDNode &Node) {
  return Printable([&Node](raw_ostream &OS) {
#ifndef NDEBUG
    static const raw_ostream::Colors Color[] = {
        raw_ostream::BLACK,  raw_ostream::RED,  raw_ostream::GREEN,
        raw_ostream::YELLOW, raw_ostream::BLUE, raw_ostream::MAGENTA,
        raw_ostream::CYAN,
    };
    OS.changeColor(Color[Node.PersistentId % std::size(Color)]);
    OS << 't' << Node.PersistentId;
    OS.resetColor();
#else
    OS << (const void*)&Node;
#endif
  });
}

// Print the MMO with more information from the SelectionDAG.
static void printMemOperand(raw_ostream &OS, const MachineMemOperand &MMO,
                            const MachineFunction *MF, const Module *M,
                            const MachineFrameInfo *MFI,
                            const TargetInstrInfo *TII, LLVMContext &Ctx) {
  ModuleSlotTracker MST(M);
  if (MF)
    MST.incorporateFunction(MF->getFunction());
  SmallVector<StringRef, 0> SSNs;
  MMO.print(OS, MST, SSNs, Ctx, MFI, TII);
}

static void printMemOperand(raw_ostream &OS, const MachineMemOperand &MMO,
                            const SelectionDAG *G) {
  if (G) {
    const MachineFunction *MF = &G->getMachineFunction();
    return printMemOperand(OS, MMO, MF, MF->getFunction().getParent(),
                           &MF->getFrameInfo(),
                           G->getSubtarget().getInstrInfo(), *G->getContext());
  }

  LLVMContext Ctx;
  return printMemOperand(OS, MMO, /*MF=*/nullptr, /*M=*/nullptr,
                         /*MFI=*/nullptr, /*TII=*/nullptr, Ctx);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SDNode::dump() const { dump(nullptr); }

LLVM_DUMP_METHOD void SDNode::dump(const SelectionDAG *G) const {
  print(dbgs(), G);
  dbgs() << '\n';
}
#endif

void SDNode::print_types(raw_ostream &OS, const SelectionDAG *G) const {
  for (unsigned i = 0, e = getNumValues(); i != e; ++i) {
    if (i) OS << ",";
    if (getValueType(i) == MVT::Other)
      OS << "ch";
    else
      OS << getValueType(i).getEVTString();
  }
}

void SDNode::print_details(raw_ostream &OS, const SelectionDAG *G) const {
  if (getFlags().hasNoUnsignedWrap())
    OS << " nuw";

  if (getFlags().hasNoSignedWrap())
    OS << " nsw";

  if (getFlags().hasExact())
    OS << " exact";

  if (getFlags().hasDisjoint())
    OS << " disjoint";

  if (getFlags().hasSameSign())
    OS << " samesign";

  if (getFlags().hasInBounds())
    OS << " inbounds";

  if (getFlags().hasNonNeg())
    OS << " nneg";

  if (getFlags().hasNoNaNs())
    OS << " nnan";

  if (getFlags().hasNoInfs())
    OS << " ninf";

  if (getFlags().hasNoSignedZeros())
    OS << " nsz";

  if (getFlags().hasAllowReciprocal())
    OS << " arcp";

  if (getFlags().hasAllowContract())
    OS << " contract";

  if (getFlags().hasApproximateFuncs())
    OS << " afn";

  if (getFlags().hasAllowReassociation())
    OS << " reassoc";

  if (getFlags().hasNoFPExcept())
    OS << " nofpexcept";

  if (getFlags().hasNoConvergent())
    OS << " noconvergent";

  if (const MachineSDNode *MN = dyn_cast<MachineSDNode>(this)) {
    if (!MN->memoperands_empty()) {
      OS << "<";
      OS << "Mem:";
      for (MachineSDNode::mmo_iterator i = MN->memoperands_begin(),
           e = MN->memoperands_end(); i != e; ++i) {
        printMemOperand(OS, **i, G);
        if (std::next(i) != e)
          OS << " ";
      }
      OS << ">";
    }
  } else if (const ShuffleVectorSDNode *SVN =
               dyn_cast<ShuffleVectorSDNode>(this)) {
    OS << "<";
    for (unsigned i = 0, e = ValueList[0].getVectorNumElements(); i != e; ++i) {
      int Idx = SVN->getMaskElt(i);
      if (i) OS << ",";
      if (Idx < 0)
        OS << "u";
      else
        OS << Idx;
    }
    OS << ">";
  } else if (const ConstantSDNode *CSDN = dyn_cast<ConstantSDNode>(this)) {
    OS << '<' << CSDN->getAPIntValue() << '>';
  } else if (const ConstantFPSDNode *CSDN = dyn_cast<ConstantFPSDNode>(this)) {
    if (&CSDN->getValueAPF().getSemantics() == &APFloat::IEEEsingle())
      OS << '<' << CSDN->getValueAPF().convertToFloat() << '>';
    else if (&CSDN->getValueAPF().getSemantics() == &APFloat::IEEEdouble())
      OS << '<' << CSDN->getValueAPF().convertToDouble() << '>';
    else {
      OS << "<APFloat(";
      CSDN->getValueAPF().bitcastToAPInt().print(OS, false);
      OS << ")>";
    }
  } else if (const GlobalAddressSDNode *GADN =
             dyn_cast<GlobalAddressSDNode>(this)) {
    int64_t offset = GADN->getOffset();
    OS << '<';
    GADN->getGlobal()->printAsOperand(OS);
    OS << '>';
    if (offset > 0)
      OS << " + " << offset;
    else
      OS << " " << offset;
    if (unsigned int TF = GADN->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const FrameIndexSDNode *FIDN = dyn_cast<FrameIndexSDNode>(this)) {
    OS << "<" << FIDN->getIndex() << ">";
  } else if (const JumpTableSDNode *JTDN = dyn_cast<JumpTableSDNode>(this)) {
    OS << "<" << JTDN->getIndex() << ">";
    if (unsigned int TF = JTDN->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(this)){
    int offset = CP->getOffset();
    if (CP->isMachineConstantPoolEntry())
      OS << "<" << *CP->getMachineCPVal() << ">";
    else
      OS << "<" << *CP->getConstVal() << ">";
    if (offset > 0)
      OS << " + " << offset;
    else
      OS << " " << offset;
    if (unsigned int TF = CP->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const TargetIndexSDNode *TI = dyn_cast<TargetIndexSDNode>(this)) {
    OS << "<" << TI->getIndex() << '+' << TI->getOffset() << ">";
    if (unsigned TF = TI->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const BasicBlockSDNode *BBDN = dyn_cast<BasicBlockSDNode>(this)) {
    OS << "<";
    const Value *LBB = (const Value*)BBDN->getBasicBlock()->getBasicBlock();
    if (LBB)
      OS << LBB->getName() << " ";
    OS << (const void*)BBDN->getBasicBlock() << ">";
  } else if (const RegisterSDNode *R = dyn_cast<RegisterSDNode>(this)) {
    OS << ' ' << printReg(R->getReg(),
                          G ? G->getSubtarget().getRegisterInfo() : nullptr);
  } else if (const ExternalSymbolSDNode *ES =
             dyn_cast<ExternalSymbolSDNode>(this)) {
    OS << "'" << ES->getSymbol() << "'";
    if (unsigned int TF = ES->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const SrcValueSDNode *M = dyn_cast<SrcValueSDNode>(this)) {
    if (M->getValue())
      OS << "<" << M->getValue() << ">";
    else
      OS << "<null>";
  } else if (const MDNodeSDNode *MD = dyn_cast<MDNodeSDNode>(this)) {
    if (MD->getMD())
      OS << "<" << MD->getMD() << ">";
    else
      OS << "<null>";
  } else if (const VTSDNode *N = dyn_cast<VTSDNode>(this)) {
    OS << ":" << N->getVT();
  }
  else if (const LoadSDNode *LD = dyn_cast<LoadSDNode>(this)) {
    OS << "<";

    printMemOperand(OS, *LD->getMemOperand(), G);

    bool doExt = true;
    switch (LD->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:  OS << ", anyext"; break;
    case ISD::SEXTLOAD: OS << ", sext"; break;
    case ISD::ZEXTLOAD: OS << ", zext"; break;
    }
    if (doExt)
      OS << " from " << LD->getMemoryVT();

    const char *AM = getIndexedModeName(LD->getAddressingMode());
    if (*AM)
      OS << ", " << AM;

    OS << ">";
  } else if (const StoreSDNode *ST = dyn_cast<StoreSDNode>(this)) {
    OS << "<";
    printMemOperand(OS, *ST->getMemOperand(), G);

    if (ST->isTruncatingStore())
      OS << ", trunc to " << ST->getMemoryVT();

    const char *AM = getIndexedModeName(ST->getAddressingMode());
    if (*AM)
      OS << ", " << AM;

    OS << ">";
  } else if (const MaskedLoadSDNode *MLd = dyn_cast<MaskedLoadSDNode>(this)) {
    OS << "<";

    printMemOperand(OS, *MLd->getMemOperand(), G);

    bool doExt = true;
    switch (MLd->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:  OS << ", anyext"; break;
    case ISD::SEXTLOAD: OS << ", sext"; break;
    case ISD::ZEXTLOAD: OS << ", zext"; break;
    }
    if (doExt)
      OS << " from " << MLd->getMemoryVT();

    const char *AM = getIndexedModeName(MLd->getAddressingMode());
    if (*AM)
      OS << ", " << AM;

    if (MLd->isExpandingLoad())
      OS << ", expanding";

    OS << ">";
  } else if (const MaskedStoreSDNode *MSt = dyn_cast<MaskedStoreSDNode>(this)) {
    OS << "<";
    printMemOperand(OS, *MSt->getMemOperand(), G);

    if (MSt->isTruncatingStore())
      OS << ", trunc to " << MSt->getMemoryVT();

    const char *AM = getIndexedModeName(MSt->getAddressingMode());
    if (*AM)
      OS << ", " << AM;

    if (MSt->isCompressingStore())
      OS << ", compressing";

    OS << ">";
  } else if (const auto *MGather = dyn_cast<MaskedGatherSDNode>(this)) {
    OS << "<";
    printMemOperand(OS, *MGather->getMemOperand(), G);

    bool doExt = true;
    switch (MGather->getExtensionType()) {
    default: doExt = false; break;
    case ISD::EXTLOAD:  OS << ", anyext"; break;
    case ISD::SEXTLOAD: OS << ", sext"; break;
    case ISD::ZEXTLOAD: OS << ", zext"; break;
    }
    if (doExt)
      OS << " from " << MGather->getMemoryVT();

    auto Signed = MGather->isIndexSigned() ? "signed" : "unsigned";
    auto Scaled = MGather->isIndexScaled() ? "scaled" : "unscaled";
    OS << ", " << Signed << " " << Scaled << " offset";

    OS << ">";
  } else if (const auto *MScatter = dyn_cast<MaskedScatterSDNode>(this)) {
    OS << "<";
    printMemOperand(OS, *MScatter->getMemOperand(), G);

    if (MScatter->isTruncatingStore())
      OS << ", trunc to " << MScatter->getMemoryVT();

    auto Signed = MScatter->isIndexSigned() ? "signed" : "unsigned";
    auto Scaled = MScatter->isIndexScaled() ? "scaled" : "unscaled";
    OS << ", " << Signed << " " << Scaled << " offset";

    OS << ">";
  } else if (const MemSDNode *M = dyn_cast<MemSDNode>(this)) {
    OS << "<";
    interleaveComma(M->memoperands(), OS, [&](const MachineMemOperand *MMO) {
      printMemOperand(OS, *MMO, G);
    });
    if (auto *A = dyn_cast<AtomicSDNode>(M))
      if (A->getOpcode() == ISD::ATOMIC_LOAD) {
        bool doExt = true;
        switch (A->getExtensionType()) {
        default: doExt = false; break;
        case ISD::EXTLOAD:  OS << ", anyext"; break;
        case ISD::SEXTLOAD: OS << ", sext"; break;
        case ISD::ZEXTLOAD: OS << ", zext"; break;
        }
        if (doExt)
          OS << " from " << A->getMemoryVT();
      }
    OS << ">";
  } else if (const BlockAddressSDNode *BA =
               dyn_cast<BlockAddressSDNode>(this)) {
    int64_t offset = BA->getOffset();
    OS << "<";
    BA->getBlockAddress()->getFunction()->printAsOperand(OS, false);
    OS << ", ";
    BA->getBlockAddress()->getBasicBlock()->printAsOperand(OS, false);
    OS << ">";
    if (offset > 0)
      OS << " + " << offset;
    else
      OS << " " << offset;
    if (unsigned int TF = BA->getTargetFlags())
      OS << " [TF=" << TF << ']';
  } else if (const AddrSpaceCastSDNode *ASC =
               dyn_cast<AddrSpaceCastSDNode>(this)) {
    OS << '['
       << ASC->getSrcAddressSpace()
       << " -> "
       << ASC->getDestAddressSpace()
       << ']';
  } else if (const auto *AA = dyn_cast<AssertAlignSDNode>(this)) {
    OS << '<' << AA->getAlign().value() << '>';
  }

  if (VerboseDAGDumping) {
    if (unsigned Order = getIROrder())
        OS << " [ORD=" << Order << ']';

    if (getNodeId() != -1)
      OS << " [ID=" << getNodeId() << ']';
    if (!(isa<ConstantSDNode>(this) || (isa<ConstantFPSDNode>(this))))
      OS << " # D:" << isDivergent();

    if (G && !G->GetDbgValues(this).empty()) {
      OS << " [NoOfDbgValues=" << G->GetDbgValues(this).size() << ']';
      for (SDDbgValue *Dbg : G->GetDbgValues(this))
        if (!Dbg->isInvalidated())
          Dbg->print(OS);
    } else if (getHasDebugValue())
      OS << " [NoOfDbgValues>0]";

    if (const auto *MD = G ? G->getPCSections(this) : nullptr) {
      OS << " [pcsections ";
      MD->printAsOperand(OS, G->getMachineFunction().getFunction().getParent());
      OS << ']';
    }

    if (MDNode *MMRA = G ? G->getMMRAMetadata(this) : nullptr) {
      OS << " [mmra ";
      MMRA->printAsOperand(OS,
                           G->getMachineFunction().getFunction().getParent());
      OS << ']';
    }
  }
}

LLVM_DUMP_METHOD void SDDbgValue::print(raw_ostream &OS) const {
  OS << " DbgVal(Order=" << getOrder() << ')';
  if (isInvalidated())
    OS << "(Invalidated)";
  if (isEmitted())
    OS << "(Emitted)";
  OS << "(";
  bool Comma = false;
  for (const SDDbgOperand &Op : getLocationOps()) {
    if (Comma)
      OS << ", ";
    switch (Op.getKind()) {
    case SDDbgOperand::SDNODE:
      if (Op.getSDNode())
        OS << "SDNODE=" << PrintNodeId(*Op.getSDNode()) << ':' << Op.getResNo();
      else
        OS << "SDNODE";
      break;
    case SDDbgOperand::CONST:
      OS << "CONST";
      break;
    case SDDbgOperand::FRAMEIX:
      OS << "FRAMEIX=" << Op.getFrameIx();
      break;
    case SDDbgOperand::VREG:
      OS << "VREG=" << printReg(Op.getVReg());
      break;
    }
    Comma = true;
  }
  OS << ")";
  if (isIndirect()) OS << "(Indirect)";
  if (isVariadic())
    OS << "(Variadic)";
  OS << ":\"" << Var->getName() << '"';
#ifndef NDEBUG
  if (Expr->getNumElements())
    Expr->dump();
#endif
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SDDbgValue::dump() const {
  if (isInvalidated())
    return;
  print(dbgs());
  dbgs() << "\n";
}
#endif

/// Return true if this node is so simple that we should just print it inline
/// if it appears as an operand.
static bool shouldPrintInline(const SDNode &Node, const SelectionDAG *G) {
  // Avoid lots of cluttering when inline printing nodes with associated
  // DbgValues in verbose mode.
  if (VerboseDAGDumping && G && !G->GetDbgValues(&Node).empty())
    return false;
  if (Node.getOpcode() == ISD::EntryToken)
    return false;
  return Node.getNumOperands() == 0;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
static void DumpNodes(const SDNode *N, unsigned indent, const SelectionDAG *G) {
  for (const SDValue &Op : N->op_values()) {
    if (shouldPrintInline(*Op.getNode(), G))
      continue;
    if (Op.getNode()->hasOneUse())
      DumpNodes(Op.getNode(), indent+2, G);
  }

  dbgs().indent(indent);
  N->dump(G);
}

LLVM_DUMP_METHOD void SelectionDAG::dump() const { dump(false); }

LLVM_DUMP_METHOD void SelectionDAG::dump(bool Sorted) const {
  dbgs() << "SelectionDAG has " << AllNodes.size() << " nodes:\n";

  auto dumpEachNode = [this](const SDNode &N) {
    if (!N.hasOneUse() && &N != getRoot().getNode() &&
        (!shouldPrintInline(N, this) || N.use_empty()))
      DumpNodes(&N, 2, this);
  };

  if (Sorted) {
    SmallVector<const SDNode *> SortedNodes;
    SortedNodes.reserve(AllNodes.size());
    getTopologicallyOrderedNodes(SortedNodes);
    for (const SDNode *N : SortedNodes)
      dumpEachNode(*N);
  } else {
    for (const SDNode &N : allnodes())
      dumpEachNode(N);
  }

  if (getRoot().getNode()) DumpNodes(getRoot().getNode(), 2, this);
  dbgs() << "\n";

  if (VerboseDAGDumping) {
    if (DbgBegin() != DbgEnd())
      dbgs() << "SDDbgValues:\n";
    for (auto *Dbg : make_range(DbgBegin(), DbgEnd()))
      Dbg->dump();
    if (ByvalParmDbgBegin() != ByvalParmDbgEnd())
      dbgs() << "Byval SDDbgValues:\n";
    for (auto *Dbg : make_range(ByvalParmDbgBegin(), ByvalParmDbgEnd()))
      Dbg->dump();
  }
  dbgs() << "\n";
}
#endif

void SDNode::printr(raw_ostream &OS, const SelectionDAG *G) const {
  OS << PrintNodeId(*this) << ": ";
  print_types(OS, G);
  OS << " = " << getOperationName(G);
  print_details(OS, G);
}

static bool printOperand(raw_ostream &OS, const SelectionDAG *G,
                         const SDValue Value) {
  if (!Value.getNode()) {
    OS << "<null>";
    return false;
  }

  if (shouldPrintInline(*Value.getNode(), G)) {
    OS << Value->getOperationName(G) << ':';
    Value->print_types(OS, G);
    Value->print_details(OS, G);
    return true;
  }

  OS << PrintNodeId(*Value.getNode());
  if (unsigned RN = Value.getResNo())
    OS << ':' << RN;
  return false;
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
using VisitedSDNodeSet = SmallPtrSet<const SDNode *, 32>;

static void DumpNodesr(raw_ostream &OS, const SDNode *N, unsigned indent,
                       const SelectionDAG *G, VisitedSDNodeSet &once) {
  if (!once.insert(N).second) // If we've been here before, return now.
    return;

  // Dump the current SDNode, but don't end the line yet.
  OS.indent(indent);
  N->printr(OS, G);

  // Having printed this SDNode, walk the children:
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (i) OS << ",";
    OS << " ";

    const SDValue Op = N->getOperand(i);
    bool printedInline = printOperand(OS, G, Op);
    if (printedInline)
      once.insert(Op.getNode());
  }

  OS << "\n";

  // Dump children that have grandchildren on their own line(s).
  for (const SDValue &Op : N->op_values())
    DumpNodesr(OS, Op.getNode(), indent+2, G, once);
}

LLVM_DUMP_METHOD void SDNode::dumpr() const {
  VisitedSDNodeSet once;
  DumpNodesr(dbgs(), this, 0, nullptr, once);
}

LLVM_DUMP_METHOD void SDNode::dumpr(const SelectionDAG *G) const {
  VisitedSDNodeSet once;
  DumpNodesr(dbgs(), this, 0, G, once);
}
#endif

static void printrWithDepthHelper(raw_ostream &OS, const SDNode *N,
                                  const SelectionDAG *G, unsigned depth,
                                  unsigned indent) {
  if (depth == 0)
    return;

  OS.indent(indent);

  N->print(OS, G);

  for (const SDValue &Op : N->op_values()) {
    // Don't follow chain operands.
    if (Op.getValueType() == MVT::Other)
      continue;
    // Don't print children that were fully rendered inline.
    if (shouldPrintInline(*Op.getNode(), G))
      continue;
    OS << '\n';
    printrWithDepthHelper(OS, Op.getNode(), G, depth - 1, indent + 2);
  }
}

void SDNode::printrWithDepth(raw_ostream &OS, const SelectionDAG *G,
                            unsigned depth) const {
  printrWithDepthHelper(OS, this, G, depth, 0);
}

void SDNode::printrFull(raw_ostream &OS, const SelectionDAG *G) const {
  // Don't print impossibly deep things.
  printrWithDepth(OS, G, 10);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void SDNode::dumprWithDepth(const SelectionDAG *G, unsigned depth) const {
  printrWithDepth(dbgs(), G, depth);
}

LLVM_DUMP_METHOD void SDNode::dumprFull(const SelectionDAG *G) const {
  // Don't print impossibly deep things.
  dumprWithDepth(G, 10);
}
#endif

void SDNode::print(raw_ostream &OS, const SelectionDAG *G) const {
  printr(OS, G);
  // Under VerboseDAGDumping divergence will be printed always.
  if (isDivergent() && !VerboseDAGDumping)
    OS << " # D:1";
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i) {
    if (i) OS << ", "; else OS << " ";
    printOperand(OS, G, getOperand(i));
  }
  if (DebugLoc DL = getDebugLoc()) {
    OS << ", ";
    DL.print(OS);
  }
  if (PrintSDNodeAddrs)
    OS << " ; " << this;
}
