//===--------------------- HexagonPostRAHandleQFP.cpp --------------------------
//===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
// For v79 and above, we generate qf operations for HVX which includes vadd,
// vsub and vmpy instructions. These qf operations with qf operands are fast,
// maintain similar accuracy as IEEE and saves power.
//
// However, these qf operands should always be converted back to IEEE format
// when used in non-HVX instructions. This is because of how the qf values
// are stored in memory. qf operands have 4 extra bits. If used in non-HVX
// operations, these bits get dropped resulting in incorrect value being
// used. So, before use in any non-HVX operation we need to convert these
// qf values to IEEE format.
//
// During register allocation, when no more physical registers are available
// the qf operands may be spilled to memory. This instantly causes loss of
// accuracy. This pass prevents that by:
// 1. Inserting qf type to IEEE type conversion instructions before the spill.
// 2. Iterating over the uses of qf def (created before the spill) and
// changing their opcodes to handle IEEE type operands for saturating
// instructions. This is because, the refills will use IEEE type operands, but
// the instructions will still assume qf operands. For non-saturating
// instructions which uses qf, we incorporate a conversion to IEEE before that.
// 3. Iterating over the uses of qf def created by the spill and replacing
// them with appropiate opcode (which uses IEEE operands) for saturating
// instructions. For non-saturating instructions which uses qf,
// we incorporate a conversion to IEEE before that.
// 4. Iterating over the copy instructions and checking their uses,
// inserting conversions from qf to IEEE whenever required. The conversions
// are inserted after their reaching def since there can be multiple defs
// for use in non-SSA form.
//
// To get the use-def chains, we make use of Register DataFlow Graph (RDF),
// since after register allocation SSA form is lost. This can be done during
// spills and fills during Frame Lowering for register allocation. However,
// that was abandoned due to the intermediate state of the code.
// Liveness is preserved in this pass.
//
// NOTE:
// Saturating instructions: Instructions for which transformation involves
// only changing the opcode. Eg. vmpy(qf32, sf) saturates to vmpy(sf, sf) when
// we see that the first operand is now a sf type.
// Non-Saturating instructions: Instructions for which conversion(s) have
// to be inserted. Eg. Vd.f8=Vu.qf16. If the use operand is now hf type,
// we have to insert a conversion qf16 = hf before this instruction.
//
// FIXME tags have been added for potential errors, along with the underlying
// assumption.
// FIXME Implement v81 specific optimizations as below. At the moment, we add
// converts.
// Vd.qf16=Vu.hf
// Vd.qf16=Vu.qf16
// Vd.qf32=Vu.qf32
// Vd.qf32=Vu.sf
//===---------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "llvm/CodeGen/LiveInterval.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RDFGraph.h"
#include "llvm/CodeGen/RDFLiveness.h"
#include "llvm/CodeGen/RDFRegisters.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "handle-qfp"

using namespace llvm;
using namespace rdf;

extern cl::opt<QFloatMode> QFloatModeValue;

cl::opt<bool> DisablePostRAHandleQFloat(
    "disable-handle-qfp", cl::init(false),
    cl::desc("Disable handling of Qfloat spills/refills after register "
             "allocation."));

static cl::opt<bool> EnablePostRAXqfCompliance(
    "enable-postra-xqf-check", cl::init(false),
    cl::desc("Enable ABI compliance for xqf operands post regalloc."));

namespace llvm {
FunctionPass *createHexagonPostRAHandleQFP();
void initializeHexagonPostRAHandleQFPPass(PassRegistry &);
} // namespace llvm

// QF Instructions list which need to be analyzed.
// The value of the key denotes a pair
// pair.first|pair.second = True if IEEE type, false otherwise.
// We only need to change the opcode to handling qf/sf
// misuses for these, or these instructions can be 'saturated'.
DenseMap<unsigned short, std::pair<bool, bool>> QFPSatInstsMap{
    {Hexagon::V6_vadd_qf16_mix, {false, true}},
    {Hexagon::V6_vadd_qf16, {false, false}},
    {Hexagon::V6_vadd_qf32_mix, {false, true}},
    {Hexagon::V6_vadd_qf32, {false, false}},
    {Hexagon::V6_vsub_qf16_mix, {false, true}},
    {Hexagon::V6_vsub_hf_mix, {true, false}},
    {Hexagon::V6_vsub_qf16, {false, false}},
    {Hexagon::V6_vsub_qf32_mix, {false, true}},
    {Hexagon::V6_vsub_sf_mix, {true, false}},
    {Hexagon::V6_vsub_qf32, {false, false}},
    {Hexagon::V6_vmpy_qf16_mix_hf, {false, true}},
    {Hexagon::V6_vmpy_qf16, {false, false}},
    {Hexagon::V6_vmpy_qf32_mix_hf, {false, true}},
    {Hexagon::V6_vmpy_qf32_qf16, {false, false}},
    {Hexagon::V6_vmpy_qf32, {false, false}},
    {Hexagon::V6_vmpy_rt_qf16, {false, true}},
    // These opcodes take a single operand only.
    // Second placeholder op is true always.
    {Hexagon::V6_vabs_qf32_qf32, {false, true}},
    {Hexagon::V6_vabs_qf16_qf16, {false, true}},
    {Hexagon::V6_vneg_qf32_qf32, {false, true}},
    {Hexagon::V6_vneg_qf16_qf16, {false, true}},
    {Hexagon::V6_vilog2_qf32, {false, true}},
    {Hexagon::V6_vilog2_qf16, {false, true}},
    {Hexagon::V6_vconv_qf32_qf32, {false, true}},
    {Hexagon::V6_vconv_qf16_qf16, {false, true}},
};

// This holds the instruction opcodes for which there are
// no 'saturating' opcodes. The only way is to insert
// convert instructions before them.
SmallVector<unsigned short, 5> QFNonSatInstr{
    Hexagon::V6_vconv_hf_qf16, Hexagon::V6_vconv_hf_qf32,
    Hexagon::V6_vconv_sf_qf32,
    // v81 instructions
    Hexagon::V6_vconv_bf_qf32, Hexagon::V6_vconv_f8_qf16};

namespace {
class HexagonPostRAHandleQFP : public MachineFunctionPass {
public:
  static char ID;
  HexagonPostRAHandleQFP() : MachineFunctionPass(ID) {
    PassRegistry &R = *PassRegistry::getPassRegistry();
    initializeHexagonPostRAHandleQFPPass(R);
  }
  StringRef getPassName() const override {
    return "Hexagon handle QFloat spills and refills post RA.";
  }
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addRequired<MachineDominanceFrontierWrapperPass>();
    AU.setPreservesCFG();
  }
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  // QFUses collects the instructions which uses QF operands.
  // These have to be deleted and transformed to opcodes
  // to denote usage of IEEE operands.
  // It might involve changing the order of the Register operands.
  using QFUses = std::map<MachineInstr *, std::pair<bool, bool>>;
  QFUses QFUsesMap;

  // Holds the Register Dataglow Graph.
  DataFlowGraph *DFG = nullptr;

  // Stores spill nodes and their reaching definition instructions
  // which generates the qf operand to be stored.
  std::vector<std::pair<MachineInstr *, NodeAddr<DefNode *>>> SpillMIs;
  // Stores the refill nodes consisting of load instructions.
  std::vector<NodeAddr<DefNode *>> RefillMIs;

  // Stores the type of op.
  enum ConvOperand {
    Undefined = 0x0,
    Lo = 0x1,
    Hi = 0x2,
    HiLo = 0x3,
  };
  // Stores the convert instructions which take qf operands.
  MapVector<MachineInstr *, unsigned> QFNonSatMIs;

  // Stores the qf-generating vmul/vadd/etc. nodes with mutiple reaching defs
  std::set<NodeAddr<StmtNode *>> PossibleMultiReachDefs;
  // Qf generating instructions to ignore. Do not insert conversion instruction
  // to sf/hf from qf, if the instr is present in this list; since that means
  // a conversion has already been inserted after the instruction.
  SmallPtrSet<MachineInstr *, 4> IgnoreInsertConvList;

  // Register type
  enum class RegType { qf32, qf16, qf32_double, qf16_double, ieee, undefined };
  // Stores the copy instructions which their reaching def, along with the op
  // type
  std::map<std::pair<NodeAddr<DefNode *>, NodeAddr<DefNode *>>, RegType>
      QFCopys;

  // Stores the reaching defs of copies whose result has to be converted to IEEE
  DenseMap<MachineInstr *, RegType> ReachDefOfCopies;

  // Stores copies which need to be converted back to qf. The uses of these
  // copies feed to qf type instructions and hence can be converted back to qf
  // type.
  DenseMap<MachineInstr *, std::pair<NodeAddr<DefNode *>, RegType>>
      ConvertToQfCopies;

  // Subregister kill set for a doubletype use. The pair of bool,bool
  // represents the hi and lo subregisters of the double register.
  DenseMap<MachineInstr *, std::pair<bool, bool>> SubRegKillSet;

  const HexagonInstrInfo *HII = nullptr;
  const HexagonRegisterInfo *HRI = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  Liveness *LV = nullptr;
  const HexagonSubtarget *HST = nullptr;

  void collectQFPStackSpill(NodeAddr<StmtNode *> *);
  void collectQFPStackRefill(NodeAddr<StmtNode *> *);
  void collectCopies(NodeAddr<StmtNode *> *);
  bool HandleRefills();
  bool HandleSpills();
  bool HandleCopies();
  bool HandleNonSatInstr();
  bool HandleMultiReachingDefs();
  bool HandleReachDefOfCopies();
  bool HandleConvertToQfCopies();
  RegType HasQfUses(NodeAddr<DefNode *>, MachineInstr *);
  void collectConvQFInstr(NodeAddr<DefNode *> &);
  void collectQFUses(NodeAddr<DefNode *>, MachineInstr *DefMI);
  void conditionallyInsert(MachineInstr &, Register &);

  // Helper functions
  unsigned short getreplacedQFOpcode(unsigned, bool, bool);
  MCPhysReg findAllocatableReg(MachineInstr *MI) const;
  void insertIEEEToQF(MachineInstr *, Register, MachineOperand, bool is32bit);
  void collectLivenessForSubregs(NodeAddr<UseNode *> &);
  void insertInstr(MachineInstr *, unsigned, unsigned, unsigned, RegState);
};
} // namespace

// This class handles spurious vector instrutions which do not
// follow the ABI. For eg, vcombine(qf,qf) takes qf operands
// instead of IEEE type. This diagnostic pass can be used
// as a final verifier for XQF implementation. Turned off by
// default
class XqfPostRADiagnosis {
public:
  XqfPostRADiagnosis(DataFlowGraph &G, Liveness &L, const HexagonInstrInfo *HII)
      : G(&G), L(&L), HII(HII) {}
  // Deleting default constructor to handle misconstruction
  XqfPostRADiagnosis() = delete;

  void runCompliance() const;
  void print_warning(Twine &, MachineInstr *, MachineInstr *) const;

private:
  DataFlowGraph *G = nullptr;
  Liveness *L = nullptr;
  const HexagonInstrInfo *HII = nullptr;
};

void XqfPostRADiagnosis::print_warning(Twine &wstr, MachineInstr *DefMI,
                                       MachineInstr *UseMI) const {
#ifndef NDEBUG
  dbgs() << wstr;
  dbgs() << "\n\tDef:";
  DefMI->dump();
  // dbgs() << "\t" << DefMI->getParent()->getName();
  dbgs() << "\tUse:";
  UseMI->dump();
  // dbgs() << "\t" << UseMI->getParent()->getName();
#endif // NDEBUG
}

// This static function gets all reached uses of a def.
// When it encounters a phi node, it goes over the
// reached uses of the phi node too.
static void getAllRealUses(NodeAddr<DefNode *> DA, NodeSet &UNodeSet,
                           Liveness *L, DataFlowGraph *G,
                           bool comprehensive = false) {
  RegisterRef DR = DA.Addr->getRegRef(*G);
  NodeAddr<StmtNode *> DefStmt = DA.Addr->getOwner(*G);
  MachineInstr *Instr = DefStmt.Addr->getCode();
  auto UseSet = L->getAllReachedUses(DR, DA);

  for (auto UI : UseSet) {
    NodeAddr<UseNode *> UA = G->addr<UseNode *>(UI);

    MachineFunction *MF = Instr->getMF();
    const auto &HRI = MF->getSubtarget<HexagonSubtarget>().getRegisterInfo();
    Register RR = UA.Addr->getRegRef(*G).Id;
    if (HRI->isFakeReg(RR))
      continue;

    if (UA.Addr->getFlags() & NodeAttrs::PhiRef) {
      NodeAddr<PhiNode *> PA = UA.Addr->getOwner(*G);
      NodeId id = PA.Id;
      const Liveness::RefMap &phiUse = L->getRealUses(id);
      for (auto I : phiUse) {
        if (!G->getPRI().alias(RegisterRef(I.first), DR))
          continue;
        auto phiUseSet = I.second;
        for (auto phiUI : phiUseSet) {
          NodeAddr<UseNode *> phiUA = G->addr<UseNode *>(phiUI.first);
          UNodeSet.insert(phiUA.Id);
        }
      }
    } else {
      // FIXME Due to bug in RDF, check if the reaching def of the use
      // reaches this instruction
      if (comprehensive) {
        UNodeSet.insert(UA.Id);
        continue;
      }
      NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*G);
      for (NodeAddr<UseNode *> UA : UseStmt.Addr->members_if(G->IsUse, *G)) {
        NodeId QFPDefNode = UA.Addr->getReachingDef();
        NodeAddr<DefNode *> RegDef = G->addr<DefNode *>(QFPDefNode);
        // FIXME Reaching def computation error
        if (QFPDefNode == 0)
          continue;
        NodeAddr<StmtNode *> RegStmt = RegDef.Addr->getOwner(*G);
        MachineInstr *ReachDefInstr = RegStmt.Addr->getCode();
        if (ReachDefInstr && ReachDefInstr == Instr)
          UNodeSet.insert(UA.Id);
      }
    }
  }
}

void XqfPostRADiagnosis::runCompliance() const {
  NodeAddr<FuncNode *> FA = G->getFunc();
  for (NodeAddr<BlockNode *> BA : FA.Addr->members(*G)) {
    for (auto IA : BA.Addr->members(*G)) {
      if (!G->IsCode<NodeAttrs::Stmt>(IA))
        continue;
      NodeAddr<StmtNode *> SA = IA;
      MachineInstr *DefMI = SA.Addr->getCode();
      if (DefMI->isDebugInstr() || DefMI->isInlineAsm())
        continue;
      auto NodeBase = SA.Addr->members_if(G->IsDef, *G);
      if (NodeBase.empty())
        continue;
      NodeAddr<DefNode *> DfNode = NodeBase.front();

      NodeSet UseSet;
      getAllRealUses(DfNode, UseSet, L, G, true);
      for (auto UI : UseSet) {
        NodeAddr<UseNode *> UA = G->addr<UseNode *>(UI);
        if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
          continue;
        NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*G);
        MachineInstr *UseMI = UseStmt.Addr->getCode();
        if (UseMI->isDebugInstr() || UseMI->isInlineAsm())
          continue;
        unsigned OpNo = UA.Addr->getOp().getOperandNo();
        if (HII->usesQF32Operand(UseMI, OpNo) && !HII->isQFP32Instr(DefMI)) {
          Twine wstr(Twine("Mismatch: sf type used as qf32 at operand ")
                         .concat(Twine(OpNo)));
          print_warning(wstr, DefMI, UseMI);
        } else if (!HII->usesQF32Operand(UseMI, OpNo) &&
                   HII->isQFP32Instr(DefMI)) {
          Twine wstr(Twine("Mismatch: qf32 type used as sf at operand ")
                         .concat(Twine(OpNo)));
          print_warning(wstr, DefMI, UseMI);
        } else if (HII->usesQF16Operand(UseMI, OpNo) &&
                   !HII->isQFP16Instr(DefMI)) {
          Twine wstr(Twine("Mismatch: hf type used as qf16 at operand ")
                         .concat(Twine(OpNo)));
          print_warning(wstr, DefMI, UseMI);
        } else if (!HII->usesQF16Operand(UseMI, OpNo) &&
                   HII->isQFP16Instr(DefMI)) {
          Twine wstr(Twine("Mismatch: qf16 type used as hf at operand ")
                         .concat(Twine(OpNo)));
          print_warning(wstr, DefMI, UseMI);
        }
      }
    }
  }
}

char HexagonPostRAHandleQFP::ID = 0;

namespace llvm {
char &HexagonPostRAHandleQFPID = HexagonPostRAHandleQFP::ID;
}

// Check whether the instruction is added already, if not add it
// along with the Register values and qf type.
// If already added, then check the register values and edit them.
void HexagonPostRAHandleQFP::conditionallyInsert(MachineInstr &MI,
                                                 Register &DefReg) {
  LLVM_DEBUG(dbgs() << "\nCollecting instruction using QF: "; MI.dump());
  // check if the key exists.
  Register Reg1 = MI.getOperand(1).getReg();

  // If the use is a unary operation, make second register point to Defreg
  // This ensures that secondOp is always true
  Register Reg2 = MI.getNumOperands() == 2 ? DefReg : MI.getOperand(2).getReg();

  if (QFUsesMap.find(&MI) != QFUsesMap.end()) {
    auto Entry = QFUsesMap[&MI];
    bool firstOp = ((Reg1 == DefReg) ? true : false) | Entry.first;
    bool secondOp = ((Reg2 == DefReg) ? true : false) | Entry.second;
    QFUsesMap[&MI] = std::make_pair(firstOp, secondOp);

  } else { // encountered first time.
    // Get the default type of the operand:
    // True : IEEE type
    // False : QF type
    auto defaultPair = QFPSatInstsMap[MI.getOpcode()];
    bool firstOp = (Reg1 == DefReg) ? true : defaultPair.first;
    bool secondOp = (Reg2 == DefReg) ? true : defaultPair.second;
    QFUsesMap[&MI] = std::make_pair(firstOp, secondOp);
  }
}

unsigned short HexagonPostRAHandleQFP::getreplacedQFOpcode(unsigned srcOpcode,
                                                           bool firstOp,
                                                           bool secondOp) {
  if (firstOp && secondOp) {
    switch (srcOpcode) {
    case Hexagon::V6_vadd_qf32:
    case Hexagon::V6_vadd_qf32_mix:
      return Hexagon::V6_vadd_sf;
    case Hexagon::V6_vadd_qf16:
    case Hexagon::V6_vadd_qf16_mix:
      return Hexagon::V6_vadd_hf;

    case Hexagon::V6_vsub_qf32:
    case Hexagon::V6_vsub_qf32_mix:
    case Hexagon::V6_vsub_sf_mix:
      return Hexagon::V6_vsub_sf;
    case Hexagon::V6_vsub_qf16:
    case Hexagon::V6_vsub_qf16_mix:
    case Hexagon::V6_vsub_hf_mix:
      return Hexagon::V6_vsub_hf;

    case Hexagon::V6_vmpy_qf32:
      return Hexagon::V6_vmpy_qf32_sf;
    case Hexagon::V6_vmpy_qf16:
    case Hexagon::V6_vmpy_qf16_mix_hf:
      return Hexagon::V6_vmpy_qf16_hf;
    case Hexagon::V6_vmpy_qf32_qf16:
    case Hexagon::V6_vmpy_qf32_mix_hf:
      return Hexagon::V6_vmpy_qf32_hf;

    case Hexagon::V6_vmpy_rt_qf16:
      return Hexagon::V6_vmpy_rt_hf;
    // v81 opcodes start
    case Hexagon::V6_vabs_qf32_qf32:
      return Hexagon::V6_vabs_qf32_sf;
    case Hexagon::V6_vabs_qf16_qf16:
      return Hexagon::V6_vabs_qf16_hf;
    case Hexagon::V6_vneg_qf32_qf32:
      return Hexagon::V6_vneg_qf32_sf;
    case Hexagon::V6_vneg_qf16_qf16:
      return Hexagon::V6_vneg_qf16_hf;
    case Hexagon::V6_vilog2_qf32:
      return Hexagon::V6_vilog2_sf;
    case Hexagon::V6_vilog2_qf16:
      return Hexagon::V6_vilog2_hf;
    case Hexagon::V6_vconv_qf32_qf32:
      return Hexagon::V6_vconv_qf32_sf;
    case Hexagon::V6_vconv_qf16_qf16:
      return Hexagon::V6_vconv_qf16_hf;
      // v81 opcodes end

    default:
      llvm_unreachable("Invalid qf opcode in this scenario!");
    }
  } else if (firstOp) {
    switch (srcOpcode) {
    case Hexagon::V6_vadd_qf32:
      return Hexagon::V6_vadd_qf32_mix; // interchange reqd
    case Hexagon::V6_vadd_qf16:
      return Hexagon::V6_vadd_qf16_mix; // interchange reqd

    case Hexagon::V6_vsub_qf32:
      if (HST->useHVXV81Ops())
        return Hexagon::V6_vsub_sf_mix;
      else if (HST->useHVXV79Ops())
        return Hexagon::V6_vsub_sf; // conv reqd
      else
        llvm_unreachable("Invalid Hexagon Arch for this scenario!");
    case Hexagon::V6_vsub_qf16:
      if (HST->useHVXV81Ops())
        return Hexagon::V6_vsub_hf_mix;
      else if (HST->useHVXV79Ops())
        return Hexagon::V6_vsub_hf; // conv reqd
      else
        llvm_unreachable("Invalid Hexagon Arch for this scenario!");
    case Hexagon::V6_vsub_qf32_mix:
      return Hexagon::V6_vsub_sf;
    case Hexagon::V6_vsub_qf16_mix:
      return Hexagon::V6_vsub_hf;

    // This opcode does not have a mixed type. Hence if one
    // of op1 or op2 is IEEE type and another qf type,
    // send the opcode which takes in both as IEEE type.
    case Hexagon::V6_vmpy_qf32:
      return Hexagon::V6_vmpy_qf32_sf; // conv reqd
    case Hexagon::V6_vmpy_qf16:
      return Hexagon::V6_vmpy_qf16_mix_hf; // interchange reqd
    case Hexagon::V6_vmpy_qf32_qf16:
      return Hexagon::V6_vmpy_qf32_mix_hf; // interchange reqd

    default:
      return srcOpcode;
    }
  } else if (secondOp) {
    switch (srcOpcode) {
    case Hexagon::V6_vadd_qf32:
      return Hexagon::V6_vadd_qf32_mix;
    case Hexagon::V6_vadd_qf16:
      return Hexagon::V6_vadd_qf16_mix;

    case Hexagon::V6_vsub_qf32:
      return Hexagon::V6_vsub_qf32_mix;
    case Hexagon::V6_vsub_qf16:
      return Hexagon::V6_vsub_qf16_mix;
    case Hexagon::V6_vsub_sf_mix:
      return Hexagon::V6_vsub_sf;
    case Hexagon::V6_vsub_hf_mix:
      return Hexagon::V6_vsub_hf;

    case Hexagon::V6_vmpy_qf32:
      return Hexagon::V6_vmpy_qf32_sf; // conv reqd

    case Hexagon::V6_vmpy_qf16:
      return Hexagon::V6_vmpy_qf16_mix_hf;
    case Hexagon::V6_vmpy_qf32_qf16:
      return Hexagon::V6_vmpy_qf32_mix_hf;

    default:
      return srcOpcode;
    }
  } else
    return srcOpcode;
}

// Insert IEEE to Qf conversion instructions
// is32bit: If true, SrcReg holds sf type, else a hf type
void HexagonPostRAHandleQFP::insertIEEEToQF(MachineInstr *MI, Register SrcReg,
                                            MachineOperand SrcOp,
                                            bool is32bit = false) {

  auto MBB = MI->getParent();
  MachineInstrBuilder MIB;
  const DebugLoc &DL = MI->getDebugLoc();

  if (HST->useHVXV81Ops()) {
    auto Op = is32bit ? Hexagon::V6_vconv_qf32_sf : Hexagon::V6_vconv_qf16_hf;
    MIB = BuildMI(*MBB, *MI, DL, HII->get(Op), SrcReg)
              .addReg(SrcReg, RegState::Renamable | RegState::Kill);
    LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
               MIB.getInstr()->dump());

  } else if (HST->useHVXV79Ops()) {
    // Get an available register
    auto V0_Reg = findAllocatableReg(MI);

    MIB = BuildMI(*MBB, *MI, DL, HII->get(Hexagon::V6_vd0), V0_Reg);
    LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
               MIB.getInstr()->dump());
    auto Op = is32bit ? Hexagon::V6_vadd_sf : Hexagon::V6_vadd_hf;
    MIB = BuildMI(*MBB, *MI, DL, HII->get(Op), SrcReg)
              .addReg(SrcReg, RegState::Renamable | RegState::Kill)
              .addReg(V0_Reg, RegState::Kill);
    LLVM_DEBUG(dbgs() << "Inserting new instruction: "; MIB.getInstr()->dump());
  } else
    llvm_unreachable("Not possible to insert qf = hf/sf for this unknown\
      subtarget!");
}

// Create a new instruction which handle sf/hf types to replace
// qf type handling instructions.
bool HexagonPostRAHandleQFP::HandleRefills() {

  bool Changed = false;
  LLVM_DEBUG(dbgs() << "HandleRefills: ");
  std::vector<MachineInstr *> eraseList;

  for (auto It : QFUsesMap) {

    // Separately handle unary qf opcodes
    MachineInstr *MI = It.first;
    auto SrcOpcode = MI->getOpcode();
    auto Pair = It.second;
    auto SrcOp1 = MI->getOperand(1);
    Register DestReg = MI->getOperand(0).getReg();
    auto MBB = MI->getParent();
    MachineInstrBuilder MIB;
    LLVM_DEBUG(dbgs() << "\nProcessing: "; MI->dump());
    const DebugLoc &DL = MI->getDebugLoc();

    // lambda to handle unary qf operations
    // ieee: True if the 1st operand is sf/hf type, false if qf type
    auto HandleUnaryRefill = [&](MachineInstr *MI, bool isIeee) -> bool {
      if (isIeee) {
        auto finalOpcode = getreplacedQFOpcode(SrcOpcode, true, true);
        MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                  .addReg(SrcOp1.getReg(), getRegState(SrcOp1));
        Changed |= true;
        LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
                   MIB.getInstr()->dump());
      } else
        eraseList.push_back(MI);
      return Changed;
    };

    if (MI->getNumOperands() == 2) {
      Changed |= HandleUnaryRefill(It.first, It.second.first);
      continue;
    }
    auto SrcOp2 = MI->getOperand(2);

    // lambda to handle mixed type vsub instructions for v79
    auto HandleSub = [&](auto srcOpcode) -> bool {
      auto ConvOp = (srcOpcode == Hexagon::V6_vsub_qf32)
                        ? Hexagon::V6_vconv_sf_qf32
                        : Hexagon::V6_vconv_hf_qf16;
      auto SubOp = (ConvOp == Hexagon::V6_vconv_sf_qf32) ? Hexagon::V6_vsub_sf
                                                         : Hexagon::V6_vsub_hf;

      Register SrcOp2Reg = SrcOp2.getReg();
      MIB = BuildMI(*MBB, *MI, DL, HII->get(ConvOp), SrcOp2Reg)
                .addReg(SrcOp2Reg, getRegState(SrcOp2) | RegState::Kill);
      LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
                 MIB.getInstr()->dump());
      MIB = BuildMI(*MBB, *MI, DL, HII->get(SubOp), DestReg)
                .addReg(SrcOp1.getReg(), getRegState(SrcOp1))
                .addReg(SrcOp2Reg, getRegState(SrcOp2));
      // If Op2 is not killed, it is used after this instruction.
      // convert it back to original qf form.
      if (!SrcOp2.isKill())
        insertIEEEToQF(&*(++MI->getIterator()), SrcOp2.getReg(), SrcOp2);
      return true;
    };

    // If both operands are sf type, we only need to replace the opcode.
    if (Pair.first == true && Pair.second == true) {
      auto finalOpcode = getreplacedQFOpcode(SrcOpcode, true, true);
      MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                .addReg(SrcOp1.getReg(), getRegState(SrcOp1))
                .addReg(SrcOp2.getReg(), getRegState(SrcOp2));
      Changed |= true;
      LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
                 MIB.getInstr()->dump());

    } else if (Pair.first == true && Pair.second == false) {
      auto finalOpcode = getreplacedQFOpcode(SrcOpcode, true, false);

      // If 2nd op is qf, first op is sf, convert the 2nd
      // op to sf before inserting the vmpy instruction.
      if (SrcOpcode == Hexagon::V6_vmpy_qf32) {
        Register SrcOp2Reg = SrcOp2.getReg();
        MIB = BuildMI(*MBB, *MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32),
                      SrcOp2Reg)
                  .addReg(SrcOp2Reg, getRegState(SrcOp2) | RegState::Kill);
        LLVM_DEBUG(dbgs() << "\nInserting new instruction before: ";
                   MIB.getInstr()->dump());
        MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                  .addReg(SrcOp1.getReg(), getRegState(SrcOp1))
                  .addReg(SrcOp2Reg, getRegState(SrcOp2));
        // If Op2 is not killed convert back to qf, since there
        // are uses for this qf op.
        if (!SrcOp2.isKill())
          insertIEEEToQF(&*(++MI->getIterator()), SrcOp2.getReg(), SrcOp2,
                         true /* sf type reg */);

        // if the opcode is mixed type, we use Op2 as first operand
        // since that takes in qf type. Op1 is taken as second op.
      } else if (finalOpcode == Hexagon::V6_vadd_qf16_mix ||
                 finalOpcode == Hexagon::V6_vadd_qf32_mix ||
                 finalOpcode == Hexagon::V6_vmpy_qf16_mix_hf ||
                 finalOpcode == Hexagon::V6_vmpy_qf32_mix_hf) {
        MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                  .addReg(SrcOp2.getReg(), getRegState(SrcOp2))
                  .addReg(SrcOp1.getReg(), getRegState(SrcOp1));

        // Subtracting is not associative, so if Op1 is sf/hf type and
        // Op2 is qf type, we cannot interchange the operands.
        // For v79, we convert Op2 to IEEE and use the non-mix type
        // instruction for the subtraction.
        // For v81, we have an appropiate opcode with vsub(sf/hf, qf) type
      } else if ((SrcOpcode == Hexagon::V6_vsub_qf32 ||
                  SrcOpcode == Hexagon::V6_vsub_qf16) &&
                 HST->useHVXV79Ops()) {
        Changed |= HandleSub(SrcOpcode);

      } else {
        MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                  .addReg(SrcOp1.getReg(), getRegState(SrcOp1))
                  .addReg(SrcOp2.getReg(), getRegState(SrcOp2));
      }
      LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
                 MIB.getInstr()->dump());
      Changed |= true;
    } else if (Pair.first == false && Pair.second == true) {

      auto finalOpcode = getreplacedQFOpcode(SrcOpcode, false, true);
      // If 2nd op is sf, first op is qf, convert the 1st
      // op to sf before inserting the vmpy instruction.
      if (SrcOpcode == Hexagon::V6_vmpy_qf32) {
        Register SrcOp1Reg = SrcOp1.getReg();
        MIB = BuildMI(*MBB, *MI, DL, HII->get(Hexagon::V6_vconv_sf_qf32),
                      SrcOp1Reg)
                  .addReg(SrcOp1Reg, getRegState(SrcOp1) | RegState::Kill);
        LLVM_DEBUG(dbgs() << "\nInserting new instruction before: ";
                   MIB.getInstr()->dump());
        MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                  .addReg(SrcOp1Reg, getRegState(SrcOp1))
                  .addReg(SrcOp2.getReg(), getRegState(SrcOp2));
        LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
                   MIB.getInstr()->dump());
        // If Op1 is not killed convert back to qf, since there
        // are uses for this qf op.
        if (!SrcOp1.isKill())
          insertIEEEToQF(&*(++MI->getIterator()), SrcOp1.getReg(), SrcOp1,
                         true /*sf type reg*/);
      } else {

        MIB = BuildMI(*MBB, *MI, DL, HII->get(finalOpcode), DestReg)
                  .addReg(SrcOp1.getReg(), getRegState(SrcOp1))
                  .addReg(SrcOp2.getReg(), getRegState(SrcOp2));
        LLVM_DEBUG(dbgs() << "\nInserting new instruction: ";
                   MIB.getInstr()->dump());
      }
      Changed |= true;
    } else {
      // Both the operands of this instructions are valid, so no use of
      // this instruction is to be modified. We need to remove this
      // instruction from the action map QFUsesMap.
      eraseList.push_back(MI);
    }
  }

  for (MachineInstr *delMI : eraseList)
    QFUsesMap.erase(delMI);

  return Changed;
}

// Insert a new instruction.
void HexagonPostRAHandleQFP::insertInstr(MachineInstr *MI, unsigned MIOpcode,
                                         unsigned SrcReg, unsigned DstReg,
                                         RegState Flags) {

  MachineInstrBuilder MIB;
  MachineBasicBlock *MBB = MI->getParent();
  DebugLoc DL = MI->getDebugLoc();
  MachineBasicBlock::iterator MIt = MI;
  auto MINext = ++MI->getIterator();
  if (++MIt == MBB->end())
    MIB = BuildMI(MBB, DL, HII->get(MIOpcode), DstReg).addReg(SrcReg, Flags);
  else
    MIB = BuildMI(*MBB, MINext, DL, HII->get(MIOpcode), DstReg)
              .addReg(SrcReg, Flags);
  LLVM_DEBUG(dbgs() << "\t\tInserting after conv: "; MIB.getInstr()->dump());
}

// Find an available vector register to store 0x0. We have reserved vector
// register v30 to be exempted from being used during register allocation
// for this purpose.
MCPhysReg HexagonPostRAHandleQFP::findAllocatableReg(MachineInstr *MI) const {
  LLVM_DEBUG(dbgs() << "\tUsing V30 register to store a vector of zeroes!");
  return Hexagon::V30;
}

// Insert qf = sf/hf conversions before non-saturating instructions
bool HexagonPostRAHandleQFP::HandleNonSatInstr() {

  for (auto It : QFNonSatMIs) {
    MachineInstr *MI = It.first;
    auto MIOpcode = MI->getOpcode();
    auto Op = MI->getOperand(1);
    Register DefReg = Op.getReg();
    LLVM_DEBUG(dbgs() << "Analyzing convert instruction: "; MI->dump());
    // Handle hf = qf16.
    // Handle f8 = qf16
    if (MIOpcode == Hexagon::V6_vconv_hf_qf16 ||
        MIOpcode == Hexagon::V6_vconv_f8_qf16) {

      insertIEEEToQF(MI, DefReg, Op);
      // TODO Check if there are any reaching def which is qf generating type.
      // That op should be converted to sf/hf
      if (!Op.isKill())
        insertInstr(MI, Hexagon::V6_vconv_hf_qf16, DefReg, DefReg,
                    getRegState(Op) | RegState::Kill);

      // Handle hf = qf.qf.
      // Handle bf = qf.qf
    } else if (MIOpcode == Hexagon::V6_vconv_hf_qf32 ||
               MIOpcode == Hexagon::V6_vconv_bf_qf32) {
      Register DefLo = HRI->getSubReg(DefReg, Hexagon::vsub_lo);
      Register DefHi = HRI->getSubReg(DefReg, Hexagon::vsub_hi);

      if (It.second == ConvOperand::HiLo) {
        insertIEEEToQF(MI, DefLo, Op, true /* sf type */);
        insertIEEEToQF(MI, DefHi, Op, true /* sf type */);

        // Check which subregister is live and convert it
        // and according insert conversion for that subreg
        auto KillState = SubRegKillSet[MI];
        if (!KillState.first)
          insertInstr(MI, Hexagon::V6_vconv_sf_qf32, DefHi, DefHi,
                      getRegState(Op) | RegState::Kill);

        if (!KillState.second)
          insertInstr(MI, Hexagon::V6_vconv_sf_qf32, DefLo, DefLo,
                      getRegState(Op) | RegState::Kill);

      } else if (It.second == ConvOperand::Hi) {
        insertIEEEToQF(MI, DefHi, Op, true /* sf type */);
        if (!Op.isKill())
          insertInstr(MI, Hexagon::V6_vconv_sf_qf32, DefHi, DefHi,
                      getRegState(Op) | RegState::Kill);

      } else { // It.second == ConvOperand::Lo
        insertIEEEToQF(MI, DefLo, Op, true /* sf type */);
        if (!Op.isKill())
          insertInstr(MI, Hexagon::V6_vconv_sf_qf32, DefLo, DefLo,
                      getRegState(Op) | RegState::Kill);
      }
      // Handle sf = qf32.
    } else if (MIOpcode == Hexagon::V6_vconv_sf_qf32) {
      insertIEEEToQF(MI, DefReg, Op, true /* sf type */);
      if (!Op.isKill())
        insertInstr(MI, Hexagon::V6_vconv_sf_qf32, DefReg, DefReg,
                    getRegState(Op) | RegState::Kill);

    } else {
      llvm_unreachable("Unhandled non-saturating instruction!");
    }
  }

  if (QFNonSatMIs.empty())
    return false;
  return true;
}

// Calculates the liveness of subregisters (whether killed or not)
// when double register is used. This is necessary because RDF
// carries liveness of the superreg and not the subregisters individually
void HexagonPostRAHandleQFP::collectLivenessForSubregs(
    NodeAddr<UseNode *> &UsedNode) {
  RegisterRef UR = UsedNode.Addr->getRegRef(*DFG);
  NodeAddr<StmtNode *> UseStmt = UsedNode.Addr->getOwner(*DFG);
  MachineInstr *UseInstr = UseStmt.Addr->getCode();
  auto UseOp = UseInstr->getOperand(1);
  Register UseDefLo = HRI->getSubReg(UseOp.getReg(), Hexagon::vsub_lo);
  Register UseDefHi = HRI->getSubReg(UseOp.getReg(), Hexagon::vsub_hi);

  NodeSet Visited, Defs;
  bool isHiSubRegKilled = true, isLoSubRegKilled = true;
  const auto &P = LV->getAllReachingDefsRec(UR, UsedNode, Visited, Defs);

  if (!P.second)
    return;

  for (auto RD : P.first) {
    NodeAddr<DefNode *> RegDef = DFG->addr<DefNode *>(RD);
    Register RR = RegDef.Addr->getRegRef(*DFG).Id;
    if (HRI->isFakeReg(RR))
      continue;
    NodeAddr<StmtNode *> RegStmt = RegDef.Addr->getOwner(*DFG);
    MachineInstr *ReachDefInstr = RegStmt.Addr->getCode();
    if (ReachDefInstr == nullptr)
      continue;

    // If the reaching def is WReg, then the kill flag in the use is correct
    // since there is no subreg
    Register DefReg = ReachDefInstr->getOperand(0).getReg();
    if (Hexagon::HvxWRRegClass.contains(DefReg)) {
      if (!UseOp.isKill())
        isHiSubRegKilled = isLoSubRegKilled = false;

      // If the reaching ref is VReg, the liveness might be different between
      // each of the subreg. Handle them individually.
      // Find the other uses after this use for the reaching def. If it exists,
      // the subregister is live after the use.
      // NOTE: Assumption: The uses are in order in RDF.
    } else {
      NodeSet UseSet;
      getAllRealUses(RegDef, UseSet, LV, DFG);
      for (auto UIntr : UseSet) {
        NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UIntr);
        NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*DFG);
        MachineInstr *UseMI = UseStmt.Addr->getCode();
        if (UseMI == nullptr)
          continue;
        // When we reach the use set a flag to see if there are other uses
        // after this. If yes, then the register is not killed.
        if (UseMI == UseInstr)
          continue;
        if (HII->isMIBefore(UseInstr, UseMI) && DefReg == UseDefLo) {
          isLoSubRegKilled = false;
          break;
        }
        if (HII->isMIBefore(UseInstr, UseMI) && DefReg == UseDefHi) {
          isHiSubRegKilled = false;
          break;
        }
      }
    }
  }
  SubRegKillSet[UseInstr] = std::make_pair(isHiSubRegKilled, isLoSubRegKilled);
}

// Store all refill instructions.
void HexagonPostRAHandleQFP::collectQFPStackRefill(
    NodeAddr<StmtNode *> *StNode) {
  NodeAddr<DefNode *> DfNode =
      StNode->Addr->members_if(DFG->IsDef, *DFG).front();
  MachineInstr *MI = StNode->Addr->getCode();
  // Check if operand to this instruction is a frame index.
  const MachineOperand &OpFI = MI->getOperand(1);
  if (!OpFI.isFI())
    return;

  // LLVM_DEBUG(dbgs() << "\n[Stack Refill]: Collecting: "; MI->dump());
  RefillMIs.push_back(DfNode);
}

// Iterate over the uses of the qf generating instruction in RDG graph
// If we get a qf to IEEE convert instruction, add it to a list.
void HexagonPostRAHandleQFP::collectConvQFInstr(NodeAddr<DefNode *> &RegDef) {

  NodeSet UseSet;
  NodeAddr<StmtNode *> DefStmt = RegDef.Addr->getOwner(*DFG);
  MachineInstr *DefInstr = DefStmt.Addr->getCode();
  getAllRealUses(RegDef, UseSet, LV, DFG);
  for (auto UI : UseSet) {
    NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UI);
    if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
      continue;
    NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*DFG);
    MachineInstr *QFConvInstr = UseStmt.Addr->getCode();
    if (std::find(QFNonSatInstr.begin(), QFNonSatInstr.end(),
                  QFConvInstr->getOpcode()) != QFNonSatInstr.end()) {

      // The use is a double register type. But the def can be hi/lo or double
      // type. So conversion needs to be inserted only for the type
      // which is in IEEE form.
      auto UseReg = QFConvInstr->getOperand(1).getReg();
      auto DefReg = DefInstr->getOperand(0).getReg();
      if (Hexagon::HvxWRRegClass.contains(UseReg)) {

        collectLivenessForSubregs(UA);
        unsigned Op = ConvOperand::Undefined;
        if (QFNonSatMIs.contains(QFConvInstr))
          Op = QFNonSatMIs[QFConvInstr];

        // Def is double type
        if (Hexagon::HvxWRRegClass.contains(DefReg))
          Op = ConvOperand::HiLo;
        // Def is lo of double type
        else if (DefReg == HRI->getSubReg(UseReg, Hexagon::vsub_lo))
          Op |= ConvOperand::Lo;
        // Def is hi of double type
        else
          Op |= ConvOperand::Hi;
        QFNonSatMIs[QFConvInstr] = Op;
      } else // for other def-use, BothOp is used as default
        QFNonSatMIs[QFConvInstr] = ConvOperand::HiLo;

      IgnoreInsertConvList.insert(DefInstr);
      LLVM_DEBUG(std::string OpType = ""; switch (QFNonSatMIs[QFConvInstr]) {
        case ConvOperand::HiLo:
          OpType = "HiLo Op";
          break;
        case ConvOperand::Lo:
          OpType = "Lo Op";
          break;
        case ConvOperand::Hi:
          OpType = "Hi Op";
          break;
        default:
          OpType = "Undefined";
      } dbgs() << "Collecting convert instruction with type "
               << OpType << " : ";
                 QFConvInstr->dump());
    }
  }
}

// Check if the COPY statements use came from a def which generates
// a qf type. If yes, collect it in a vector. Also, collect copies
// with reaching def other copies (nested copies).
void HexagonPostRAHandleQFP::collectCopies(NodeAddr<StmtNode *> *StNode) {

  NodeAddr<DefNode *> CopyDef =
      StNode->Addr->members_if(DFG->IsDef, *DFG).front();
  MachineInstr *CopyInstr = StNode->Addr->getCode();
  LLVM_DEBUG(dbgs() << "\nAnalyzing copy: "; StNode->Addr->getCode()->dump());

  for (NodeAddr<UseNode *> UA : StNode->Addr->members_if(DFG->IsUse, *DFG)) {
    RegisterRef UR = UA.Addr->getRegRef(*DFG);
    NodeSet Visited, Defs;
    const auto &P = LV->getAllReachingDefsRec(UR, UA, Visited, Defs);
    if (!P.second) {
      LLVM_DEBUG({
        dbgs() << "*** Unable to collect all reaching defs for use ***\n"
               << PrintNode<UseNode *>(UA, *DFG) << '\n';
      });
      continue;
    }

    // Note: there can be multiple reaching defs of the copy
    for (auto RD : P.first) {
      NodeAddr<DefNode *> RegDef = DFG->addr<DefNode *>(RD);
      Register RR = RegDef.Addr->getRegRef(*DFG).Id;
      if (HRI->isFakeReg(RR))
        continue;
      NodeAddr<StmtNode *> RegStmt = RegDef.Addr->getOwner(*DFG);
      MachineInstr *ReachDefInstr = RegStmt.Addr->getCode();
      if (ReachDefInstr == nullptr)
        continue;
      LLVM_DEBUG(dbgs() << "\t[Reaching Def]: "; ReachDefInstr->dump());

      // If the reaching def is a COPY,collect it with reg type ieee
      if (ReachDefInstr->getOpcode() == TargetOpcode::COPY) {
        auto pairKey = std::make_pair(CopyDef, RegDef);
        QFCopys[pairKey] = RegType::ieee;
        continue;
      }

      // If the reaching def is a qf instr, collect the copy.
      // reg type is selected based on the op
      auto RegT = RegType::undefined;
      if (HII->isQFPInstr(ReachDefInstr)) {
        if (HII->isQFP32Instr(ReachDefInstr)) {
          // check whether the copies register is hvxWR or hvxVR type
          // NOTE: Assumption: A copy's reaching def shall not be 2,
          // i.e., for each of the subregister.
          if (Hexagon::HvxWRRegClass.contains(
                  ReachDefInstr->getOperand(0).getReg()))
            RegT = RegType::qf32_double;
          else
            RegT = RegType::qf32;
        } else if (HII->isQFP16Instr(ReachDefInstr)) {
          // Check if qf16 instruction outputs double-wide register
          if (Hexagon::HvxWRRegClass.contains(
                  ReachDefInstr->getOperand(0).getReg())) {
            RegT = RegType::qf16_double;
          } else {
            RegT = RegType::qf16;
          }
        }
      } else {
        // if the copy involves non-qf vector registers collect it too
        Register CopyReg = CopyInstr->getOperand(1).getReg();
        if (Hexagon::HvxWRRegClass.contains(CopyReg) ||
            Hexagon::HvxVRRegClass.contains(CopyReg))
          RegT = RegType::ieee;
        else
          continue;
      }
      auto pairKey = std::make_pair(CopyDef, RegDef);
      QFCopys[pairKey] = RegT;
    }
  }
}

// Inserts an qf instruction to a list. These instruction
// values are spilled to the stack.
void HexagonPostRAHandleQFP::collectQFPStackSpill(
    NodeAddr<StmtNode *> *StNode) {

  MachineInstr *MI = StNode->Addr->getCode();
  LLVM_DEBUG(dbgs() << "\n[Stack Spill]: Analyzing: "; MI->dump());
  // Check if operand to this instruction is a frame index.
  const MachineOperand &OpFI = MI->getOperand(0);
  if (!OpFI.isFI())
    return;

  // Pre-RegAlloc
  //%46:hvxwr = V6_vmpy_qf32_hf %7:hvxvr, %10:hvxvr
  // PS_vstorerw_ai %stack.3, 0, %46:hvxwr :: (store (s2048) into %stack.3,
  // align 128)
  // Post-RegAlloc
  // renamable $w4 = V6_vmpy_qf32_hf killed renamable $v1, renamable $v0
  // PS_vstorerw_ai %stack.3, 0, renamable $w4 :: (store (s2048) into %stack.3,
  // align 128)

  if (!MI->getOperand(2).isReg())
    return;

  // Iterate over the operands of the store instruction to get their reaching
  // defs
  NodeId QFPDefNode = 0;
  for (NodeAddr<UseNode *> UA : StNode->Addr->members_if(DFG->IsUse, *DFG)) {
    QFPDefNode = UA.Addr->getReachingDef();

    // Get the defining instruction node(s)
    NodeAddr<DefNode *> RegDef = DFG->addr<DefNode *>(QFPDefNode);
    assert(QFPDefNode != 0 && "Reaching def computation error");
    NodeAddr<StmtNode *> RegStmt = RegDef.Addr->getOwner(*DFG);
    MachineInstr *ReachDefInstr = RegStmt.Addr->getCode();
    if (ReachDefInstr == nullptr)
      continue;
    LLVM_DEBUG(dbgs() << "[Stack Spill]:\tReaching Def of operand:";
               ReachDefInstr->dump());
    // Reaching Def cannot be a phi instruction.
    if (RegDef.Addr->getFlags() & NodeAttrs::PhiRef)
      continue;

    if (!HII->isQFPInstr(ReachDefInstr))
      continue;

    auto RR = RegDef.Addr->getRegRef(*DFG).Id;
    if (HRI->isFakeReg(RR))
      continue;

    LLVM_DEBUG(dbgs() << "Found a QFPStackSpill via \n"; MI->dump();
               dbgs() << "The corresponding XQF instruction is:\n";
               ReachDefInstr->dump());

    // Collect the spills.
    SpillMIs.push_back(std::make_pair(MI, RegDef));
  }
}

// Find the uses of qf generating instructions and conditionally add them
// to a list.
void HexagonPostRAHandleQFP::collectQFUses(NodeAddr<DefNode *> RegDef,
                                           MachineInstr *DefMI) {

  NodeSet UseSet;
  LLVM_DEBUG(dbgs() << " Finding uses of: "; DefMI->dump(););
  getAllRealUses(RegDef, UseSet, LV, DFG);

  for (auto UI : UseSet) {
    NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UI);
    if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
      continue;
    NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*DFG);
    MachineInstr *UseMI = UseStmt.Addr->getCode();
    LLVM_DEBUG(dbgs() << "\t\t\t[Reached Use of QF operand]: "; UseMI->dump());

    Register UsedReg = UA.Addr->getRegRef(*DFG).Id;
    if (QFPSatInstsMap.find(UseMI->getOpcode()) != QFPSatInstsMap.end()) {
      if (PossibleMultiReachDefs.count(UseStmt) == 0) {
        PossibleMultiReachDefs.insert(UseStmt);
        LLVM_DEBUG(dbgs() << "\n[Collect instr with possible multidef]:";
                   UseMI->dump());
      }
      conditionallyInsert(*UseMI, UsedReg);
    }
  }
}

// Process the list which can have multiple definitions. A possible case
// can be reaching defs to be a copy and a qf-generating instr respectively.
// Only handle the qf-generating instruction by inserting convert to sf/hf
// after it. Additionally, then handle the reached uses of this reaching
// def since the type has changed to sf/hf from qf after the conversion.
bool HexagonPostRAHandleQFP::HandleMultiReachingDefs() {

  bool Changed = false;
  // Note: It may seem this loop can further add to PossibleMultiReachDefs.
  // But it is not expected to since if any instruction has multiple
  // definitions it should already be present in it.
  for (auto It : PossibleMultiReachDefs) {
    MachineInstr *Instr = It.Addr->getCode();
    // get the op type for the original instruction.
    // True is sf/hf, false is qf
    auto Pair = QFUsesMap[Instr];

    unsigned short UseNo = 1;
    // Iterate over the operands
    for (NodeAddr<UseNode *> UA : It.Addr->members_if(DFG->IsUse, *DFG)) {

      // If the type is qf for the operand,
      // we skip since there is no scope for mismatch
      if ((UseNo == 1 && Pair.first == false) ||
          (UseNo == 2 && Pair.second == false)) {
        ++UseNo;
        continue;
      }

      RegisterRef UR = UA.Addr->getRegRef(*DFG);
      NodeSet Visited, Defs;
      const auto &P = LV->getAllReachingDefsRec(UR, UA, Visited, Defs);
      if (!P.second) {
        LLVM_DEBUG({
          dbgs() << "*** Unable to collect all reaching defs for use ***\n"
                 << PrintNode<UseNode *>(UA, *DFG) << '\n';
        });
        continue;
      }

      // Iterate over the reaching defs and process the ones which
      // generate qf. Ignore the ones which have already been handled
      for (auto RD : P.first) {
        NodeAddr<DefNode *> RegDef = DFG->addr<DefNode *>(RD);

        // Ignore fake reaches
        auto RR = RegDef.Addr->getRegRef(*DFG).Id;
        if (HRI->isFakeReg(RR))
          continue;

        NodeAddr<StmtNode *> RegStmt = RegDef.Addr->getOwner(*DFG);
        MachineInstr *ReachDefInstr = RegStmt.Addr->getCode();

        if (ReachDefInstr == nullptr)
          continue;

        if (!HII->isQFPInstr(ReachDefInstr))
          continue;
        if (IgnoreInsertConvList.find(ReachDefInstr) !=
            IgnoreInsertConvList.end())
          continue;
        LLVM_DEBUG(dbgs() << "[Multidef] Handling reaching def:";
                   ReachDefInstr->dump());

        auto *MBB = ReachDefInstr->getParent();
        auto &dl = ReachDefInstr->getDebugLoc();
        auto NextReachMI = ++ReachDefInstr->getIterator();
        auto DefOp = ReachDefInstr->getOperand(0);
        Register OpReg = DefOp.getReg();
        MachineInstrBuilder MIB;

        // For double vector regs, two conversions are inserted. Single
        // conversion for qf32 type
        if (HII->isQFP32Instr(ReachDefInstr)) {
          // if the reaching def is a qf double type
          if (Hexagon::HvxWRRegClass.contains(
                  ReachDefInstr->getOperand(0).getReg())) {
            Register RegLo = HRI->getSubReg(OpReg, Hexagon::vsub_lo);
            Register RegHi = HRI->getSubReg(OpReg, Hexagon::vsub_hi);
            MIB = BuildMI(*MBB, NextReachMI, dl,
                          HII->get(Hexagon::V6_vconv_sf_qf32), RegLo)
                      .addReg(RegLo, RegState::Renamable | RegState::Kill);
            LLVM_DEBUG(dbgs() << "[MultiDef] Inserting convert instruction: ";
                       MIB.getInstr()->dump());
            MIB = BuildMI(*MBB, NextReachMI, dl,
                          HII->get(Hexagon::V6_vconv_sf_qf32), RegHi)
                      .addReg(RegHi, RegState::Renamable | RegState::Kill);
          } else { // If the reaching def is a qf type
            MIB = BuildMI(*MBB, NextReachMI, dl,
                          HII->get(Hexagon::V6_vconv_sf_qf32), OpReg)
                      .addReg(OpReg, RegState::Renamable | RegState::Kill);
          }
        }
        if (HII->isQFP16Instr(ReachDefInstr)) {
          MIB = BuildMI(*MBB, NextReachMI, dl,
                        HII->get(Hexagon::V6_vconv_hf_qf16), OpReg)
                    .addReg(OpReg, RegState::Renamable | RegState::Kill);
        }
        LLVM_DEBUG(dbgs() << "[MultiDef] Inserting convert instruction: ";
                   MIB.getInstr()->dump(); dbgs() << "\tafter instruction: ";
                   ReachDefInstr->dump());

        // find the uses of the newly transformed to sf/hf and handle
        // accordingly. Uses can be vmul/vadd/etc. types or converts which take
        // in qf types.
        collectQFUses(RegDef, ReachDefInstr);
        collectConvQFInstr(RegDef);
        IgnoreInsertConvList.insert(ReachDefInstr);
        Changed = true;
      }
      UseNo++;
    }
  }
  return Changed;
}

bool HexagonPostRAHandleQFP::HandleConvertToQfCopies() {
  if (ConvertToQfCopies.empty())
    return false;

  LLVM_DEBUG(
      dbgs() << "\n*** Inserting convert to qf for selected copies ***\n");

  // Any reached use of the copy should not already be collected to be
  // converted to IEEE. If present, it means that the reached use has
  // other reaching def with type IEEE, other than this copy.
  auto CanTransform = [&](MachineInstr *MI, unsigned OpNo) -> bool {
    if (QFUsesMap.find(MI) != QFUsesMap.end()) {
      auto Entry = QFUsesMap[MI];
      if (OpNo == 1 && Entry.first == true)
        return false;
      if (OpNo == 2 && Entry.second == true)
        return false;
    }
    return true;
  };

  for (auto It : ConvertToQfCopies) {
    NodeSet UseSet;
    getAllRealUses(It.second.first, UseSet, LV, DFG);

    bool transform = true;
    for (auto UI : UseSet) {
      NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UI);
      if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
        continue;
      NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*DFG);
      MachineInstr *UseMI = UseStmt.Addr->getCode();
      unsigned OpNo = UA.Addr->getOp().getOperandNo();

      if (!CanTransform(UseMI, OpNo)) {
        transform = false;
        break;
      }
    }

    if (transform) {

      LLVM_DEBUG(dbgs() << "\n[HandleConvertToQfCopies]\tProcessing Copy:";
                 It.first->dump());
      auto CopyOp = It.first->getOperand(0);
      auto NextMIIter = std::next(It.first->getIterator());
      switch (It.second.second) {
      case RegType::qf32_double: {
        Register DefLo = HRI->getSubReg(CopyOp.getReg(), Hexagon::vsub_lo);
        Register DefHi = HRI->getSubReg(CopyOp.getReg(), Hexagon::vsub_hi);
        insertIEEEToQF(&*NextMIIter, DefLo, CopyOp, /*is32bit=*/true);
        insertIEEEToQF(&*NextMIIter, DefHi, CopyOp, /*is32bit=*/true);
        break;
      }
      case RegType::qf16_double: {
        Register DefLo = HRI->getSubReg(CopyOp.getReg(), Hexagon::vsub_lo);
        Register DefHi = HRI->getSubReg(CopyOp.getReg(), Hexagon::vsub_hi);
        insertIEEEToQF(&*NextMIIter, DefLo, CopyOp, /*is32bit=*/false);
        insertIEEEToQF(&*NextMIIter, DefHi, CopyOp, /*is32bit=*/false);
        break;
      }
      case RegType::qf16:
        insertIEEEToQF(&*NextMIIter, CopyOp.getReg(), CopyOp,
                       /*is32bit=*/false);
        break;
      case RegType::qf32:
        insertIEEEToQF(&*NextMIIter, CopyOp.getReg(), CopyOp, /*is32bit=*/true);
        break;
      default:
        break;
      }
    } else {
      collectQFUses(It.second.first, It.first);
      collectConvQFInstr(It.second.first);
    }
  }
  return true;
}

bool HexagonPostRAHandleQFP::HandleReachDefOfCopies() {
  if (ReachDefOfCopies.empty())
    return false;

  MachineInstrBuilder MIB;
  for (auto It : ReachDefOfCopies) {
    auto *MBB = It.first->getParent();
    auto &dl = It.first->getDebugLoc();
    auto NextMI = ++(It.first)->getIterator();
    auto RegOp = It.first->getOperand(0);
    Register OpReg = RegOp.getReg();

    if (It.second == RegType::qf32)
      MIB =
          BuildMI(*MBB, NextMI, dl, HII->get(Hexagon::V6_vconv_sf_qf32), OpReg)
              .addReg(OpReg, RegState::Renamable | RegState::Kill);
    else if (It.second == RegType::qf16)
      MIB =
          BuildMI(*MBB, NextMI, dl, HII->get(Hexagon::V6_vconv_hf_qf16), OpReg)
              .addReg(OpReg, RegState::Renamable | RegState::Kill);
    else if (It.second == RegType::qf32_double) {
      Register RegLo = HRI->getSubReg(OpReg, Hexagon::vsub_lo);
      Register RegHi = HRI->getSubReg(OpReg, Hexagon::vsub_hi);
      MIB =
          BuildMI(*MBB, NextMI, dl, HII->get(Hexagon::V6_vconv_sf_qf32), RegLo)
              .addReg(RegLo, RegState::Renamable | RegState::Kill);
      LLVM_DEBUG(dbgs() << "Inserting convert instruction: ";
                 MIB.getInstr()->dump());
      MIB =
          BuildMI(*MBB, NextMI, dl, HII->get(Hexagon::V6_vconv_sf_qf32), RegHi)
              .addReg(RegHi, RegState::Renamable | RegState::Kill);
    } else if (It.second == RegType::qf16_double) {
      Register RegLo = HRI->getSubReg(OpReg, Hexagon::vsub_lo);
      Register RegHi = HRI->getSubReg(OpReg, Hexagon::vsub_hi);
      MIB =
          BuildMI(*MBB, NextMI, dl, HII->get(Hexagon::V6_vconv_hf_qf16), RegLo)
              .addReg(RegLo, RegState::Renamable | RegState::Kill);
      LLVM_DEBUG(dbgs() << "Inserting convert instruction: ";
                 MIB.getInstr()->dump());
      MIB =
          BuildMI(*MBB, NextMI, dl, HII->get(Hexagon::V6_vconv_hf_qf16), RegHi)
              .addReg(RegHi, RegState::Renamable | RegState::Kill);
    }
    LLVM_DEBUG(dbgs() << "Inserting convert instruction: ";
               MIB.getInstr()->dump(); dbgs() << "\tafter instruction: ";
               It.first->dump());
  }
  return true;
}

HexagonPostRAHandleQFP::RegType
HexagonPostRAHandleQFP::HasQfUses(NodeAddr<DefNode *> CopyDef,
                                  MachineInstr *CopyMI) {
  NodeSet UseSet;
  getAllRealUses(CopyDef, UseSet, LV, DFG);

  if (UseSet.size() == 0)
    return RegType::undefined;

  bool hasQf16Use = false;
  bool hasQf32Use = false;

  LLVM_DEBUG(dbgs() << "[COPY]\nUses of the copy are: ");
  for (auto UI : UseSet) {
    NodeAddr<UseNode *> UA = DFG->addr<UseNode *>(UI);
    if (UA.Addr->getFlags() & NodeAttrs::PhiRef)
      continue;
    NodeAddr<StmtNode *> UseStmt = UA.Addr->getOwner(*DFG);
    MachineInstr *UseMI = UseStmt.Addr->getCode();
    unsigned OpNo = UA.Addr->getOp().getOperandNo();

    LLVM_DEBUG(dbgs() << "\nCopy's use: "; UseMI->dump());
    // Any reached use should not be a non-qf instruction
    if (!HII->usesQFOperand(UseMI, OpNo))
      return RegType::ieee;

    // Determine the qf type from the use
    if (HII->usesQF16Operand(UseMI, OpNo))
      hasQf16Use = true;
    else if (HII->usesQF32Operand(UseMI, OpNo))
      hasQf32Use = true;

    // Any reached use should not already be converted to IEEE.
    // If present, it means that the reached use has other reaching def
    // other than the copy.
    if (QFUsesMap.find(UseMI) != QFUsesMap.end()) {
      auto Entry = QFUsesMap[UseMI];
      if (OpNo == 1 && Entry.first == true)
        return RegType::ieee;
      if (OpNo == 2 && Entry.second == true)
        return RegType::ieee;
    }
  }

  // Set the output type based on uses
  if (hasQf16Use) {
    // Check if copy destination is double-wide
    if (Hexagon::HvxWRRegClass.contains(CopyMI->getOperand(0).getReg()))
      return RegType::qf16_double;
    else
      return RegType::qf16;
  } else if (hasQf32Use) {
    if (Hexagon::HvxWRRegClass.contains(CopyMI->getOperand(0).getReg()))
      return RegType::qf32_double;
    else
      return RegType::qf32;
  }

  return RegType::undefined;
}

// Go through the collected copies and insert conversion to sf/hf
// conditionally *after their reaching defs*. This is done because there
// can be mutliple reaching defs of the copies. Also, check for the uses
// of the reaching def and handle qf uses too by changing opcode or
// inserting converts.
// Additionally, check for the uses of the copy
// and handle them via changing opcode or inserting converts.
bool HexagonPostRAHandleQFP::HandleCopies() {

  bool Changed = false;

  // If a convert is inserted after a reaching def, add it to ignorelist.
  // This is because this reaching def can be reaching def of other copies
  // due to non-SSA form.
  for (auto It : QFCopys) {

    // Get details of the copy node
    NodeAddr<DefNode *> CopyNode = It.first.first;
    NodeAddr<StmtNode *> StNode = CopyNode.Addr->getOwner(*DFG);
    [[maybe_unused]] auto *CopyMI = StNode.Addr->getCode();
    LLVM_DEBUG(dbgs() << "\nHandling Reaching Defs of COPY: "; CopyMI->dump();
               std::string Type; switch (It.second) {
                 case RegType::qf32_double:
                   Type = "qf32_double";
                   break;
                 case RegType::qf32:
                   Type = "qf32";
                   break;
                 case RegType::qf16:
                   Type = "qf16";
                   break;
                 case RegType::qf16_double:
                   Type = "qf16_double";
                   break;
                 default:
                   Type = "ieee";
               } dbgs() << "\t Type: "
                        << Type << "\n");

    // insert convert to IEEE after the reaching def if it generates qf type
    RegType RTy = It.second;
    if (RTy != RegType::ieee) {

      // get details of the reaching def node
      NodeAddr<DefNode *> ReachDefNode = It.first.second;
      NodeAddr<StmtNode *> StNode = ReachDefNode.Addr->getOwner(*DFG);
      auto *ReachingDef = StNode.Addr->getCode();

      if (IgnoreInsertConvList.find(ReachingDef) != IgnoreInsertConvList.end())
        continue;

      // Collect the reaching defs to be processed later.
      ReachDefOfCopies.insert(std::make_pair(ReachingDef, RTy));

      // Process the reached uses of the reaching def now for
      // incorrect usage, since the register type has changed
      // following the conversion.
      LLVM_DEBUG(dbgs() << "\n[COPY]\tAnalyzing uses of the reaching defs \
        of the copy...");
      collectQFUses(ReachDefNode, ReachingDef);
      collectConvQFInstr(ReachDefNode);
      IgnoreInsertConvList.insert(ReachingDef);
      Changed = true;
    }
  }

  // Loop through copies with qf uses
  for (auto It : QFCopys) {

    // Get details of the copy node
    NodeAddr<DefNode *> CopyNode = It.first.first;
    NodeAddr<StmtNode *> StNode = CopyNode.Addr->getOwner(*DFG);
    auto *CopyMI = StNode.Addr->getCode();
    LLVM_DEBUG(dbgs() << "\nHandling COPY: "; CopyMI->dump());
    RegType RTy = It.second;

    // Process the reached uses of the copy to find any incorrect
    // qf uses. If the copy's uses are all qf types, we need to convert
    // its result back to qf
    // FIXME: don't include the copy if its the last instruction since
    // it is *probably* not possible to insert via BuildMI at the end of BB
    RTy = HasQfUses(CopyNode, CopyMI);
    if (RTy != RegType::ieee && RTy != RegType::undefined &&
        (++CopyMI->getIterator() != CopyMI->getParent()->end())) {
      if (!ConvertToQfCopies.contains(CopyMI)) {
        ConvertToQfCopies[CopyMI] = std::make_pair(CopyNode, RTy);
        LLVM_DEBUG(dbgs() << "\n[ConvertToQfCopies]\tAdded copy: ";
                   CopyMI->dump(); std::string Type; switch (RTy) {
                     case RegType::qf32_double:
                       Type = "qf32_double";
                       break;
                     case RegType::qf32:
                       Type = "qf32";
                       break;
                     case RegType::qf16:
                       Type = "qf16";
                       break;
                     case RegType::qf16_double:
                       Type = "qf16_double";
                       break;
                     default:
                       Type = "ieee";
                   } dbgs() << "\t Type: "
                            << Type << "\n");
      }
      continue;
    }
    LLVM_DEBUG(dbgs() << "\n[COPY]\tAnalyzing uses of the copy...");
    collectQFUses(CopyNode, CopyMI);
    collectConvQFInstr(CopyNode);
  }

  Changed |= HandleReachDefOfCopies();
  Changed |= HandleMultiReachingDefs();
  Changed |= HandleConvertToQfCopies();

  return Changed;
}

// Inserts conversion instruction sf/hf = qf before spilling
// Uses the same physical register for conversion.
// Additinally checks for the uses of the register; and
// conditionally store them to handle later.
bool HexagonPostRAHandleQFP::HandleSpills() {

  LLVM_DEBUG(dbgs() << "\n[Handling Spill]\n");
  bool Changed = false;
  for (auto It : SpillMIs) {

    MachineInstr *MI = It.first;
    auto OpC = MI->getOpcode();

    auto NodeDef = It.second;
    NodeAddr<StmtNode *> Stmt = NodeDef.Addr->getOwner(*DFG);
    MachineInstr *DefMI = Stmt.Addr->getCode();
    auto RegOp = MI->getOperand(2);
    Register DefR = RegOp.getReg();

    // handles widened qf16/qf32 instructions.
    if (OpC == Hexagon::PS_vstorerw_ai) {
      if (!Hexagon::HvxWRRegClass.contains(DefR))
        assert(false && " Unhandled Vector Register class passed\n");
      // Walk through the uses of DefLo and DefHi and if there is QFP
      // instructions, the instruction needs to be updated to use sf operands
      // instead of qf operands.
      collectQFUses(NodeDef, DefMI);

      if (IgnoreInsertConvList.find(DefMI) != IgnoreInsertConvList.end())
        continue;

      // Collect the reached uses of ReachDefInstr
      // which are sf/hf = qf conversion instructions.
      collectConvQFInstr(NodeDef);
      Register DefLo = HRI->getSubReg(DefR, Hexagon::vsub_lo);
      Register DefHi = HRI->getSubReg(DefR, Hexagon::vsub_hi);

      // Create two copy instructions, one each for Hi and Lo conditionally.
      // Liveness is the same is for the store instruction for the register.
      // If both are double registers, two insertions are done.
      // If one of the subregs are reaching to the store, conversion is done
      // for that subreg.
      Register DReg = DefMI->getOperand(0).getReg();
      if (HII->isQFP16Instr(DefMI)) {
        if (DefLo == DReg || Hexagon::HvxWRRegClass.contains(DReg))
          insertInstr(DefMI, Hexagon::V6_vconv_hf_qf16, DefLo, DefLo,
                      getRegState(RegOp) | RegState::Kill);

        if (DefHi == DReg || Hexagon::HvxWRRegClass.contains(DReg))
          insertInstr(DefMI, Hexagon::V6_vconv_hf_qf16, DefHi, DefHi,
                      getRegState(RegOp) | RegState::Kill);
      } else if (HII->isQFP32Instr(DefMI)) {
        if (DefLo == DReg || Hexagon::HvxWRRegClass.contains(DReg))
          insertInstr(DefMI, Hexagon::V6_vconv_sf_qf32, DefLo, DefLo,
                      getRegState(RegOp) | RegState::Kill);

        if (DefHi == DReg || Hexagon::HvxWRRegClass.contains(DReg))
          insertInstr(DefMI, Hexagon::V6_vconv_sf_qf32, DefHi, DefHi,
                      getRegState(RegOp) | RegState::Kill);
      }
      IgnoreInsertConvList.insert(DefMI);
      Changed = true;

      // Handles instructions which output qf32 type.
    } else if (OpC == Hexagon::PS_vstorerv_ai && HII->isQFP32Instr(DefMI)) {
      collectQFUses(NodeDef, DefMI);
      if (IgnoreInsertConvList.find(DefMI) != IgnoreInsertConvList.end())
        continue;
      collectConvQFInstr(NodeDef);

      insertInstr(DefMI, Hexagon::V6_vconv_sf_qf32, DefR, DefR,
                  getRegState(RegOp) | RegState::Kill);

      IgnoreInsertConvList.insert(DefMI);
      Changed = true;

      // Handles instructions which output qf16 type.
    } else if (OpC == Hexagon::PS_vstorerv_ai && HII->isQFP16Instr(DefMI)) {
      collectQFUses(NodeDef, DefMI);
      if (IgnoreInsertConvList.find(DefMI) != IgnoreInsertConvList.end())
        continue;
      collectConvQFInstr(NodeDef);

      insertInstr(DefMI, Hexagon::V6_vconv_hf_qf16, DefR, DefR,
                  getRegState(RegOp) | RegState::Kill);

      IgnoreInsertConvList.insert(DefMI);
      Changed = true;
    } else {
      LLVM_DEBUG(MI->dump());
      llvm_unreachable("This case is not handled. Look above for MI\n");
    }
  }
  return Changed;
}

bool HexagonPostRAHandleQFP::runOnMachineFunction(MachineFunction &MF) {

  if (DisablePostRAHandleQFloat)
    return false;

  LLVM_DEBUG(
      dbgs() << "\n=== Entering Hexagon Fixup QF spills and refills pass ===\n"
             << "Mode: ";
      switch (QFloatModeValue) {
        case QFloatMode::StrictIEEE:
          dbgs() << "Strict IEEE";
          break;
        case QFloatMode::IEEE:
          dbgs() << "IEEE";
          break;
        case QFloatMode::Lossy:
          dbgs() << "Lossy";
          break;
        default:
          dbgs() << "Legacy";
          break;
      };
      dbgs() << "\n";);
  bool Changed = false;

  auto &_HST = MF.getSubtarget<HexagonSubtarget>();
  if (!_HST.useHVXOps())
    return false;

  HII = _HST.getInstrInfo();

  // If the mode is legacy, the function may not contain qf instructions
  // check if this pass is required to run for legacy mode.
  if (QFloatModeValue == QFloatMode::Legacy)
    if (!HII->hasQFPInstrs(MF))
      return false;

  HRI = _HST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  const auto &MDF = getAnalysis<MachineDominanceFrontierWrapperPass>().getMDF();
  MachineDominatorTree *MDT =
      &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  HST = &_HST;

  // We need Register Dataflow Graph(RDG) to calculate reaching definitions
  // since the Machine code is not in SSA.
  // DDG holds the graph on which we iterate for the nodes.
  DataFlowGraph G(MF, *HII, *HRI, *MDT, MDF);
  G.build();
  DFG = &G;

  Liveness L(*MRI, *DFG);
  L.computePhiInfo();
  LV = &L;

  // Find and save the list of QFP stack spills.
  // For refills store all refill instructions to process conditionally later.
  NodeAddr<FuncNode *> FA = DFG->getFunc();
  LLVM_DEBUG(dbgs() << "==== [RefMap#]=====:\n "
                    << Print<NodeAddr<FuncNode *>>(FA, *DFG) << "\n");
  for (NodeAddr<BlockNode *> BA : FA.Addr->members(*DFG)) {
    for (auto IA : BA.Addr->members(*DFG)) {

      if (!DFG->IsCode<NodeAttrs::Stmt>(IA))
        continue;

      // 'SA' holds the Statement node which contains the machine instruction.
      NodeAddr<StmtNode *> SA = IA;
      MachineInstr *I = SA.Addr->getCode();

      switch (I->getOpcode()) {
      case Hexagon::PS_vstorerw_ai:
      case Hexagon::PS_vstorerv_ai:
        collectQFPStackSpill(&SA);
        break;
      case Hexagon::PS_vloadrw_ai:
      case Hexagon::PS_vloadrv_ai:
        collectQFPStackRefill(&SA);
        break;
      case TargetOpcode::COPY:
        collectCopies(&SA);
        break;
      default:
        break;
      }
    }
  }

  // Walk through the spills and insert converts when necessary.
  // Additionally, walk though the uses of the converts and
  // store them conditionally for later processing.
  LLVM_DEBUG(dbgs() << "\nHandling spills....");
  Changed |= HandleSpills();
  SpillMIs.clear();

  // Walk through the uses of the refill instructions.
  // Process them if they are used as qf operands.
  LLVM_DEBUG(dbgs() << "\nCollecting refills....\n");
  for (NodeAddr<DefNode *> DfNode : RefillMIs) {

    NodeAddr<StmtNode *> Stmt = DfNode.Addr->getOwner(*DFG);
    MachineInstr *DefMI = Stmt.Addr->getCode();
    collectQFUses(DfNode, DefMI);
    collectConvQFInstr(DfNode);
  }
  RefillMIs.clear();

  LLVM_DEBUG(dbgs() << "\nHandling copies....");
  Changed |= HandleCopies();
  QFCopys.clear();
  PossibleMultiReachDefs.clear();
  ReachDefOfCopies.clear();
  ConvertToQfCopies.clear();

  LLVM_DEBUG(dbgs() << "\n === QF Uses map === "; for (auto It : QFUsesMap) {
    dbgs() << "\nInstruction: ";
    It.first->dump();
    dbgs() << "\t Property: " << It.second.first << " ," << It.second.second;
  });

  // Insert new opcodes as applicable for the refill uses.
  // Delete the original instructions.
  Changed |= HandleRefills();

  // Handle non-saturating instructions by inserting convert(s) from sf to qf.
  Changed |= HandleNonSatInstr();
  QFNonSatMIs.clear();
  // Cleanup
  for (auto It : QFUsesMap)
    It.first->eraseFromParent();
  QFUsesMap.clear();
  IgnoreInsertConvList.clear();

  // Option if enabled, checks for qf use-def mismatches
  if (EnablePostRAXqfCompliance) {
    dbgs() << "\nChecking for ABI compliance for XQF post register \
allocation for function: "
           << MF.getName() << "\n";
    DataFlowGraph DFG(MF, *HII, *HRI, *MDT, MDF);
    DFG.build();
    Liveness LV(*MRI, DFG);
    LV.computeLiveIns();
    XqfPostRADiagnosis VDiag(DFG, LV, HII);
    VDiag.runCompliance();
  }
  return Changed;
}

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//
INITIALIZE_PASS_BEGIN(HexagonPostRAHandleQFP, "handle-qfp-spills-refills",
                      "Hexagon Post RA Handle QFloat", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominanceFrontierWrapperPass)
INITIALIZE_PASS_END(HexagonPostRAHandleQFP, "handle-qfp-spills-refills",
                    "Hexagon PostRA Handle QFloat", false, false)

FunctionPass *llvm::createHexagonPostRAHandleQFP() {
  return new HexagonPostRAHandleQFP();
}
