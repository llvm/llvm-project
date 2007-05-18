//===-- SelectionDAGISel.cpp - Implement the SelectionDAGISel class -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the SelectionDAGISel class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "isel"
#include "llvm/ADT/BitVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Constants.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SchedulerRegistry.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Compiler.h"
#include <algorithm>
using namespace llvm;

#ifndef NDEBUG
static cl::opt<bool>
ViewISelDAGs("view-isel-dags", cl::Hidden,
          cl::desc("Pop up a window to show isel dags as they are selected"));
static cl::opt<bool>
ViewSchedDAGs("view-sched-dags", cl::Hidden,
          cl::desc("Pop up a window to show sched dags as they are processed"));
#else
static const bool ViewISelDAGs = 0, ViewSchedDAGs = 0;
#endif

//===---------------------------------------------------------------------===//
///
/// RegisterScheduler class - Track the registration of instruction schedulers.
///
//===---------------------------------------------------------------------===//
MachinePassRegistry RegisterScheduler::Registry;

//===---------------------------------------------------------------------===//
///
/// ISHeuristic command line option for instruction schedulers.
///
//===---------------------------------------------------------------------===//
namespace {
  cl::opt<RegisterScheduler::FunctionPassCtor, false,
          RegisterPassParser<RegisterScheduler> >
  ISHeuristic("sched",
              cl::init(&createDefaultScheduler),
              cl::desc("Instruction schedulers available:"));

  static RegisterScheduler
  defaultListDAGScheduler("default", "  Best scheduler for the target",
                          createDefaultScheduler);
} // namespace

namespace { struct AsmOperandInfo; }

namespace {
  /// RegsForValue - This struct represents the physical registers that a
  /// particular value is assigned and the type information about the value.
  /// This is needed because values can be promoted into larger registers and
  /// expanded into multiple smaller registers than the value.
  struct VISIBILITY_HIDDEN RegsForValue {
    /// Regs - This list hold the register (for legal and promoted values)
    /// or register set (for expanded values) that the value should be assigned
    /// to.
    std::vector<unsigned> Regs;
    
    /// RegVT - The value type of each register.
    ///
    MVT::ValueType RegVT;
    
    /// ValueVT - The value type of the LLVM value, which may be promoted from
    /// RegVT or made from merging the two expanded parts.
    MVT::ValueType ValueVT;
    
    RegsForValue() : RegVT(MVT::Other), ValueVT(MVT::Other) {}
    
    RegsForValue(unsigned Reg, MVT::ValueType regvt, MVT::ValueType valuevt)
      : RegVT(regvt), ValueVT(valuevt) {
        Regs.push_back(Reg);
    }
    RegsForValue(const std::vector<unsigned> &regs, 
                 MVT::ValueType regvt, MVT::ValueType valuevt)
      : Regs(regs), RegVT(regvt), ValueVT(valuevt) {
    }
    
    /// getCopyFromRegs - Emit a series of CopyFromReg nodes that copies from
    /// this value and returns the result as a ValueVT value.  This uses 
    /// Chain/Flag as the input and updates them for the output Chain/Flag.
    SDOperand getCopyFromRegs(SelectionDAG &DAG,
                              SDOperand &Chain, SDOperand &Flag) const;

    /// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
    /// specified value into the registers specified by this object.  This uses 
    /// Chain/Flag as the input and updates them for the output Chain/Flag.
    void getCopyToRegs(SDOperand Val, SelectionDAG &DAG,
                       SDOperand &Chain, SDOperand &Flag,
                       MVT::ValueType PtrVT) const;
    
    /// AddInlineAsmOperands - Add this value to the specified inlineasm node
    /// operand list.  This adds the code marker and includes the number of 
    /// values added into it.
    void AddInlineAsmOperands(unsigned Code, SelectionDAG &DAG,
                              std::vector<SDOperand> &Ops) const;
  };
}

namespace llvm {
  //===--------------------------------------------------------------------===//
  /// createDefaultScheduler - This creates an instruction scheduler appropriate
  /// for the target.
  ScheduleDAG* createDefaultScheduler(SelectionDAGISel *IS,
                                      SelectionDAG *DAG,
                                      MachineBasicBlock *BB) {
    TargetLowering &TLI = IS->getTargetLowering();
    
    if (TLI.getSchedulingPreference() == TargetLowering::SchedulingForLatency) {
      return createTDListDAGScheduler(IS, DAG, BB);
    } else {
      assert(TLI.getSchedulingPreference() ==
           TargetLowering::SchedulingForRegPressure && "Unknown sched type!");
      return createBURRListDAGScheduler(IS, DAG, BB);
    }
  }


  //===--------------------------------------------------------------------===//
  /// FunctionLoweringInfo - This contains information that is global to a
  /// function that is used when lowering a region of the function.
  class FunctionLoweringInfo {
  public:
    TargetLowering &TLI;
    Function &Fn;
    MachineFunction &MF;
    SSARegMap *RegMap;

    FunctionLoweringInfo(TargetLowering &TLI, Function &Fn,MachineFunction &MF);

    /// MBBMap - A mapping from LLVM basic blocks to their machine code entry.
    std::map<const BasicBlock*, MachineBasicBlock *> MBBMap;

    /// ValueMap - Since we emit code for the function a basic block at a time,
    /// we must remember which virtual registers hold the values for
    /// cross-basic-block values.
    DenseMap<const Value*, unsigned> ValueMap;

    /// StaticAllocaMap - Keep track of frame indices for fixed sized allocas in
    /// the entry block.  This allows the allocas to be efficiently referenced
    /// anywhere in the function.
    std::map<const AllocaInst*, int> StaticAllocaMap;

    unsigned MakeReg(MVT::ValueType VT) {
      return RegMap->createVirtualRegister(TLI.getRegClassFor(VT));
    }
    
    /// isExportedInst - Return true if the specified value is an instruction
    /// exported from its block.
    bool isExportedInst(const Value *V) {
      return ValueMap.count(V);
    }

    unsigned CreateRegForValue(const Value *V);
    
    unsigned InitializeRegForValue(const Value *V) {
      unsigned &R = ValueMap[V];
      assert(R == 0 && "Already initialized this value register!");
      return R = CreateRegForValue(V);
    }
  };
}

/// isUsedOutsideOfDefiningBlock - Return true if this instruction is used by
/// PHI nodes or outside of the basic block that defines it, or used by a 
/// switch instruction, which may expand to multiple basic blocks.
static bool isUsedOutsideOfDefiningBlock(Instruction *I) {
  if (isa<PHINode>(I)) return true;
  BasicBlock *BB = I->getParent();
  for (Value::use_iterator UI = I->use_begin(), E = I->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != BB || isa<PHINode>(*UI) ||
        // FIXME: Remove switchinst special case.
        isa<SwitchInst>(*UI))
      return true;
  return false;
}

/// isOnlyUsedInEntryBlock - If the specified argument is only used in the
/// entry block, return true.  This includes arguments used by switches, since
/// the switch may expand into multiple basic blocks.
static bool isOnlyUsedInEntryBlock(Argument *A) {
  BasicBlock *Entry = A->getParent()->begin();
  for (Value::use_iterator UI = A->use_begin(), E = A->use_end(); UI != E; ++UI)
    if (cast<Instruction>(*UI)->getParent() != Entry || isa<SwitchInst>(*UI))
      return false;  // Use not in entry block.
  return true;
}

FunctionLoweringInfo::FunctionLoweringInfo(TargetLowering &tli,
                                           Function &fn, MachineFunction &mf)
    : TLI(tli), Fn(fn), MF(mf), RegMap(MF.getSSARegMap()) {

  // Create a vreg for each argument register that is not dead and is used
  // outside of the entry block for the function.
  for (Function::arg_iterator AI = Fn.arg_begin(), E = Fn.arg_end();
       AI != E; ++AI)
    if (!isOnlyUsedInEntryBlock(AI))
      InitializeRegForValue(AI);

  // Initialize the mapping of values to registers.  This is only set up for
  // instruction values that are used outside of the block that defines
  // them.
  Function::iterator BB = Fn.begin(), EB = Fn.end();
  for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
    if (AllocaInst *AI = dyn_cast<AllocaInst>(I))
      if (ConstantInt *CUI = dyn_cast<ConstantInt>(AI->getArraySize())) {
        const Type *Ty = AI->getAllocatedType();
        uint64_t TySize = TLI.getTargetData()->getTypeSize(Ty);
        unsigned Align = 
          std::max((unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty),
                   AI->getAlignment());

        TySize *= CUI->getZExtValue();   // Get total allocated size.
        if (TySize == 0) TySize = 1; // Don't create zero-sized stack objects.
        StaticAllocaMap[AI] =
          MF.getFrameInfo()->CreateStackObject(TySize, Align);
      }

  for (; BB != EB; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ++I)
      if (!I->use_empty() && isUsedOutsideOfDefiningBlock(I))
        if (!isa<AllocaInst>(I) ||
            !StaticAllocaMap.count(cast<AllocaInst>(I)))
          InitializeRegForValue(I);

  // Create an initial MachineBasicBlock for each LLVM BasicBlock in F.  This
  // also creates the initial PHI MachineInstrs, though none of the input
  // operands are populated.
  for (BB = Fn.begin(), EB = Fn.end(); BB != EB; ++BB) {
    MachineBasicBlock *MBB = new MachineBasicBlock(BB);
    MBBMap[BB] = MBB;
    MF.getBasicBlockList().push_back(MBB);

    // Create Machine PHI nodes for LLVM PHI nodes, lowering them as
    // appropriate.
    PHINode *PN;
    for (BasicBlock::iterator I = BB->begin();(PN = dyn_cast<PHINode>(I)); ++I){
      if (PN->use_empty()) continue;
      
      MVT::ValueType VT = TLI.getValueType(PN->getType());
      unsigned NumElements;
      if (VT != MVT::Vector)
        NumElements = TLI.getNumElements(VT);
      else {
        MVT::ValueType VT1,VT2;
        NumElements = 
          TLI.getVectorTypeBreakdown(cast<VectorType>(PN->getType()),
                                     VT1, VT2);
      }
      unsigned PHIReg = ValueMap[PN];
      assert(PHIReg && "PHI node does not have an assigned virtual register!");
      const TargetInstrInfo *TII = TLI.getTargetMachine().getInstrInfo();
      for (unsigned i = 0; i != NumElements; ++i)
        BuildMI(MBB, TII->get(TargetInstrInfo::PHI), PHIReg+i);
    }
  }
}

/// CreateRegForValue - Allocate the appropriate number of virtual registers of
/// the correctly promoted or expanded types.  Assign these registers
/// consecutive vreg numbers and return the first assigned number.
unsigned FunctionLoweringInfo::CreateRegForValue(const Value *V) {
  MVT::ValueType VT = TLI.getValueType(V->getType());
  
  // The number of multiples of registers that we need, to, e.g., split up
  // a <2 x int64> -> 4 x i32 registers.
  unsigned NumVectorRegs = 1;
  
  // If this is a vector type, figure out what type it will decompose into
  // and how many of the elements it will use.
  if (VT == MVT::Vector) {
    const VectorType *PTy = cast<VectorType>(V->getType());
    unsigned NumElts = PTy->getNumElements();
    MVT::ValueType EltTy = TLI.getValueType(PTy->getElementType());
    MVT::ValueType VecTy = getVectorType(EltTy, NumElts);
    
    // Divide the input until we get to a supported size.  This will always
    // end with a scalar if the target doesn't support vectors.
    while (NumElts > 1 && !TLI.isTypeLegal(VecTy)) {
      NumElts >>= 1;
      NumVectorRegs <<= 1;
      VecTy = getVectorType(EltTy, NumElts);
    }

    // Check that VecTy isn't a 1-element vector.
    if (NumElts == 1 && VecTy == MVT::Other)
      VT = EltTy;
    else
      VT = VecTy;
  }

  // The common case is that we will only create one register for this
  // value.  If we have that case, create and return the virtual register.
  unsigned NV = TLI.getNumElements(VT);
  if (NV == 1) {
    // If we are promoting this value, pick the next largest supported type.
    MVT::ValueType PromotedType = TLI.getTypeToTransformTo(VT);
    unsigned Reg = MakeReg(PromotedType);
    // If this is a vector of supported or promoted types (e.g. 4 x i16),
    // create all of the registers.
    for (unsigned i = 1; i != NumVectorRegs; ++i)
      MakeReg(PromotedType);
    return Reg;
  }
  
  // If this value is represented with multiple target registers, make sure
  // to create enough consecutive registers of the right (smaller) type.
  VT = TLI.getTypeToExpandTo(VT);
  unsigned R = MakeReg(VT);
  for (unsigned i = 1; i != NV*NumVectorRegs; ++i)
    MakeReg(VT);
  return R;
}

//===----------------------------------------------------------------------===//
/// SelectionDAGLowering - This is the common target-independent lowering
/// implementation that is parameterized by a TargetLowering object.
/// Also, targets can overload any lowering method.
///
namespace llvm {
class SelectionDAGLowering {
  MachineBasicBlock *CurMBB;

  DenseMap<const Value*, SDOperand> NodeMap;

  /// PendingLoads - Loads are not emitted to the program immediately.  We bunch
  /// them up and then emit token factor nodes when possible.  This allows us to
  /// get simple disambiguation between loads without worrying about alias
  /// analysis.
  std::vector<SDOperand> PendingLoads;

  /// Case - A struct to record the Value for a switch case, and the
  /// case's target basic block.
  struct Case {
    Constant* Low;
    Constant* High;
    MachineBasicBlock* BB;

    Case() : Low(0), High(0), BB(0) { }
    Case(Constant* low, Constant* high, MachineBasicBlock* bb) :
      Low(low), High(high), BB(bb) { }
    uint64_t size() const {
      uint64_t rHigh = cast<ConstantInt>(High)->getSExtValue();
      uint64_t rLow  = cast<ConstantInt>(Low)->getSExtValue();
      return (rHigh - rLow + 1ULL);
    }
  };

  struct CaseBits {
    uint64_t Mask;
    MachineBasicBlock* BB;
    unsigned Bits;

    CaseBits(uint64_t mask, MachineBasicBlock* bb, unsigned bits):
      Mask(mask), BB(bb), Bits(bits) { }
  };

  typedef std::vector<Case>           CaseVector;
  typedef std::vector<CaseBits>       CaseBitsVector;
  typedef CaseVector::iterator        CaseItr;
  typedef std::pair<CaseItr, CaseItr> CaseRange;

  /// CaseRec - A struct with ctor used in lowering switches to a binary tree
  /// of conditional branches.
  struct CaseRec {
    CaseRec(MachineBasicBlock *bb, Constant *lt, Constant *ge, CaseRange r) :
    CaseBB(bb), LT(lt), GE(ge), Range(r) {}

    /// CaseBB - The MBB in which to emit the compare and branch
    MachineBasicBlock *CaseBB;
    /// LT, GE - If nonzero, we know the current case value must be less-than or
    /// greater-than-or-equal-to these Constants.
    Constant *LT;
    Constant *GE;
    /// Range - A pair of iterators representing the range of case values to be
    /// processed at this point in the binary search tree.
    CaseRange Range;
  };

  typedef std::vector<CaseRec> CaseRecVector;

  /// The comparison function for sorting the switch case values in the vector.
  /// WARNING: Case ranges should be disjoint!
  struct CaseCmp {
    bool operator () (const Case& C1, const Case& C2) {
      assert(isa<ConstantInt>(C1.Low) && isa<ConstantInt>(C2.High));
      const ConstantInt* CI1 = cast<const ConstantInt>(C1.Low);
      const ConstantInt* CI2 = cast<const ConstantInt>(C2.High);
      return CI1->getValue().slt(CI2->getValue());
    }
  };

  struct CaseBitsCmp {
    bool operator () (const CaseBits& C1, const CaseBits& C2) {
      return C1.Bits > C2.Bits;
    }
  };

  unsigned Clusterify(CaseVector& Cases, const SwitchInst &SI);
  
public:
  // TLI - This is information that describes the available target features we
  // need for lowering.  This indicates when operations are unavailable,
  // implemented with a libcall, etc.
  TargetLowering &TLI;
  SelectionDAG &DAG;
  const TargetData *TD;

  /// SwitchCases - Vector of CaseBlock structures used to communicate
  /// SwitchInst code generation information.
  std::vector<SelectionDAGISel::CaseBlock> SwitchCases;
  /// JTCases - Vector of JumpTable structures used to communicate
  /// SwitchInst code generation information.
  std::vector<SelectionDAGISel::JumpTableBlock> JTCases;
  std::vector<SelectionDAGISel::BitTestBlock> BitTestCases;
  
  /// FuncInfo - Information about the function as a whole.
  ///
  FunctionLoweringInfo &FuncInfo;

  SelectionDAGLowering(SelectionDAG &dag, TargetLowering &tli,
                       FunctionLoweringInfo &funcinfo)
    : TLI(tli), DAG(dag), TD(DAG.getTarget().getTargetData()),
      FuncInfo(funcinfo) {
  }

  /// getRoot - Return the current virtual root of the Selection DAG.
  ///
  SDOperand getRoot() {
    if (PendingLoads.empty())
      return DAG.getRoot();

    if (PendingLoads.size() == 1) {
      SDOperand Root = PendingLoads[0];
      DAG.setRoot(Root);
      PendingLoads.clear();
      return Root;
    }

    // Otherwise, we have to make a token factor node.
    SDOperand Root = DAG.getNode(ISD::TokenFactor, MVT::Other,
                                 &PendingLoads[0], PendingLoads.size());
    PendingLoads.clear();
    DAG.setRoot(Root);
    return Root;
  }

  SDOperand CopyValueToVirtualRegister(Value *V, unsigned Reg);

  void visit(Instruction &I) { visit(I.getOpcode(), I); }

  void visit(unsigned Opcode, User &I) {
    // Note: this doesn't use InstVisitor, because it has to work with
    // ConstantExpr's in addition to instructions.
    switch (Opcode) {
    default: assert(0 && "Unknown instruction type encountered!");
             abort();
      // Build the switch statement using the Instruction.def file.
#define HANDLE_INST(NUM, OPCODE, CLASS) \
    case Instruction::OPCODE:return visit##OPCODE((CLASS&)I);
#include "llvm/Instruction.def"
    }
  }

  void setCurrentBasicBlock(MachineBasicBlock *MBB) { CurMBB = MBB; }

  SDOperand getLoadFrom(const Type *Ty, SDOperand Ptr,
                        const Value *SV, SDOperand Root,
                        bool isVolatile, unsigned Alignment);

  SDOperand getIntPtrConstant(uint64_t Val) {
    return DAG.getConstant(Val, TLI.getPointerTy());
  }

  SDOperand getValue(const Value *V);

  void setValue(const Value *V, SDOperand NewN) {
    SDOperand &N = NodeMap[V];
    assert(N.Val == 0 && "Already set a value for this node!");
    N = NewN;
  }
  
  void GetRegistersForValue(AsmOperandInfo &OpInfo, bool HasEarlyClobber,
                            std::set<unsigned> &OutputRegs, 
                            std::set<unsigned> &InputRegs);

  void FindMergedConditions(Value *Cond, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB, MachineBasicBlock *CurBB,
                            unsigned Opc);
  bool isExportableFromCurrentBlock(Value *V, const BasicBlock *FromBB);
  void ExportFromCurrentBlock(Value *V);
  void LowerCallTo(Instruction &I,
                   const Type *CalledValueTy, unsigned CallingConv,
                   bool IsTailCall, SDOperand Callee, unsigned OpIdx);
                                         
  // Terminator instructions.
  void visitRet(ReturnInst &I);
  void visitBr(BranchInst &I);
  void visitSwitch(SwitchInst &I);
  void visitUnreachable(UnreachableInst &I) { /* noop */ }

  // Helpers for visitSwitch
  bool handleSmallSwitchRange(CaseRec& CR,
                              CaseRecVector& WorkList,
                              Value* SV,
                              MachineBasicBlock* Default);
  bool handleJTSwitchCase(CaseRec& CR,
                          CaseRecVector& WorkList,
                          Value* SV,
                          MachineBasicBlock* Default);
  bool handleBTSplitSwitchCase(CaseRec& CR,
                               CaseRecVector& WorkList,
                               Value* SV,
                               MachineBasicBlock* Default);
  bool handleBitTestsSwitchCase(CaseRec& CR,
                                CaseRecVector& WorkList,
                                Value* SV,
                                MachineBasicBlock* Default);  
  void visitSwitchCase(SelectionDAGISel::CaseBlock &CB);
  void visitBitTestHeader(SelectionDAGISel::BitTestBlock &B);
  void visitBitTestCase(MachineBasicBlock* NextMBB,
                        unsigned Reg,
                        SelectionDAGISel::BitTestCase &B);
  void visitJumpTable(SelectionDAGISel::JumpTable &JT);
  void visitJumpTableHeader(SelectionDAGISel::JumpTable &JT,
                            SelectionDAGISel::JumpTableHeader &JTH);
  
  // These all get lowered before this pass.
  void visitInvoke(InvokeInst &I);
  void visitInvoke(InvokeInst &I, bool AsTerminator);
  void visitUnwind(UnwindInst &I);

  void visitScalarBinary(User &I, unsigned OpCode);
  void visitVectorBinary(User &I, unsigned OpCode);
  void visitEitherBinary(User &I, unsigned ScalarOp, unsigned VectorOp);
  void visitShift(User &I, unsigned Opcode);
  void visitAdd(User &I) { 
    if (isa<VectorType>(I.getType()))
      visitVectorBinary(I, ISD::VADD);
    else if (I.getType()->isFloatingPoint())
      visitScalarBinary(I, ISD::FADD);
    else
      visitScalarBinary(I, ISD::ADD);
  }
  void visitSub(User &I);
  void visitMul(User &I) {
    if (isa<VectorType>(I.getType()))
      visitVectorBinary(I, ISD::VMUL);
    else if (I.getType()->isFloatingPoint())
      visitScalarBinary(I, ISD::FMUL);
    else
      visitScalarBinary(I, ISD::MUL);
  }
  void visitURem(User &I) { visitScalarBinary(I, ISD::UREM); }
  void visitSRem(User &I) { visitScalarBinary(I, ISD::SREM); }
  void visitFRem(User &I) { visitScalarBinary(I, ISD::FREM); }
  void visitUDiv(User &I) { visitEitherBinary(I, ISD::UDIV, ISD::VUDIV); }
  void visitSDiv(User &I) { visitEitherBinary(I, ISD::SDIV, ISD::VSDIV); }
  void visitFDiv(User &I) { visitEitherBinary(I, ISD::FDIV, ISD::VSDIV); }
  void visitAnd (User &I) { visitEitherBinary(I, ISD::AND,  ISD::VAND ); }
  void visitOr  (User &I) { visitEitherBinary(I, ISD::OR,   ISD::VOR  ); }
  void visitXor (User &I) { visitEitherBinary(I, ISD::XOR,  ISD::VXOR ); }
  void visitShl (User &I) { visitShift(I, ISD::SHL); }
  void visitLShr(User &I) { visitShift(I, ISD::SRL); }
  void visitAShr(User &I) { visitShift(I, ISD::SRA); }
  void visitICmp(User &I);
  void visitFCmp(User &I);
  // Visit the conversion instructions
  void visitTrunc(User &I);
  void visitZExt(User &I);
  void visitSExt(User &I);
  void visitFPTrunc(User &I);
  void visitFPExt(User &I);
  void visitFPToUI(User &I);
  void visitFPToSI(User &I);
  void visitUIToFP(User &I);
  void visitSIToFP(User &I);
  void visitPtrToInt(User &I);
  void visitIntToPtr(User &I);
  void visitBitCast(User &I);

  void visitExtractElement(User &I);
  void visitInsertElement(User &I);
  void visitShuffleVector(User &I);

  void visitGetElementPtr(User &I);
  void visitSelect(User &I);

  void visitMalloc(MallocInst &I);
  void visitFree(FreeInst &I);
  void visitAlloca(AllocaInst &I);
  void visitLoad(LoadInst &I);
  void visitStore(StoreInst &I);
  void visitPHI(PHINode &I) { } // PHI nodes are handled specially.
  void visitCall(CallInst &I);
  void visitInlineAsm(CallInst &I);
  const char *visitIntrinsicCall(CallInst &I, unsigned Intrinsic);
  void visitTargetIntrinsic(CallInst &I, unsigned Intrinsic);

  void visitVAStart(CallInst &I);
  void visitVAArg(VAArgInst &I);
  void visitVAEnd(CallInst &I);
  void visitVACopy(CallInst &I);

  void visitMemIntrinsic(CallInst &I, unsigned Op);

  void visitUserOp1(Instruction &I) {
    assert(0 && "UserOp1 should not exist at instruction selection time!");
    abort();
  }
  void visitUserOp2(Instruction &I) {
    assert(0 && "UserOp2 should not exist at instruction selection time!");
    abort();
  }
};
} // end namespace llvm

SDOperand SelectionDAGLowering::getValue(const Value *V) {
  SDOperand &N = NodeMap[V];
  if (N.Val) return N;
  
  const Type *VTy = V->getType();
  MVT::ValueType VT = TLI.getValueType(VTy);
  if (Constant *C = const_cast<Constant*>(dyn_cast<Constant>(V))) {
    if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      visit(CE->getOpcode(), *CE);
      SDOperand N1 = NodeMap[V];
      assert(N1.Val && "visit didn't populate the ValueMap!");
      return N1;
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(C)) {
      return N = DAG.getGlobalAddress(GV, VT);
    } else if (isa<ConstantPointerNull>(C)) {
      return N = DAG.getConstant(0, TLI.getPointerTy());
    } else if (isa<UndefValue>(C)) {
      if (!isa<VectorType>(VTy))
        return N = DAG.getNode(ISD::UNDEF, VT);

      // Create a VBUILD_VECTOR of undef nodes.
      const VectorType *PTy = cast<VectorType>(VTy);
      unsigned NumElements = PTy->getNumElements();
      MVT::ValueType PVT = TLI.getValueType(PTy->getElementType());

      SmallVector<SDOperand, 8> Ops;
      Ops.assign(NumElements, DAG.getNode(ISD::UNDEF, PVT));
      
      // Create a VConstant node with generic Vector type.
      Ops.push_back(DAG.getConstant(NumElements, MVT::i32));
      Ops.push_back(DAG.getValueType(PVT));
      return N = DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector,
                             &Ops[0], Ops.size());
    } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      return N = DAG.getConstantFP(CFP->getValue(), VT);
    } else if (const VectorType *PTy = dyn_cast<VectorType>(VTy)) {
      unsigned NumElements = PTy->getNumElements();
      MVT::ValueType PVT = TLI.getValueType(PTy->getElementType());
      
      // Now that we know the number and type of the elements, push a
      // Constant or ConstantFP node onto the ops list for each element of
      // the packed constant.
      SmallVector<SDOperand, 8> Ops;
      if (ConstantVector *CP = dyn_cast<ConstantVector>(C)) {
        for (unsigned i = 0; i != NumElements; ++i)
          Ops.push_back(getValue(CP->getOperand(i)));
      } else {
        assert(isa<ConstantAggregateZero>(C) && "Unknown packed constant!");
        SDOperand Op;
        if (MVT::isFloatingPoint(PVT))
          Op = DAG.getConstantFP(0, PVT);
        else
          Op = DAG.getConstant(0, PVT);
        Ops.assign(NumElements, Op);
      }
      
      // Create a VBUILD_VECTOR node with generic Vector type.
      Ops.push_back(DAG.getConstant(NumElements, MVT::i32));
      Ops.push_back(DAG.getValueType(PVT));
      return NodeMap[V] = DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector, &Ops[0],
                                      Ops.size());
    } else {
      // Canonicalize all constant ints to be unsigned.
      return N = DAG.getConstant(cast<ConstantInt>(C)->getZExtValue(),VT);
    }
  }
      
  if (const AllocaInst *AI = dyn_cast<AllocaInst>(V)) {
    std::map<const AllocaInst*, int>::iterator SI =
    FuncInfo.StaticAllocaMap.find(AI);
    if (SI != FuncInfo.StaticAllocaMap.end())
      return DAG.getFrameIndex(SI->second, TLI.getPointerTy());
  }
      
  unsigned InReg = FuncInfo.ValueMap[V];
  assert(InReg && "Value not in map!");
  
  // If this type is not legal, make it so now.
  if (VT != MVT::Vector) {
    if (TLI.getTypeAction(VT) == TargetLowering::Expand) {
      // Source must be expanded.  This input value is actually coming from the
      // register pair InReg and InReg+1.
      MVT::ValueType DestVT = TLI.getTypeToExpandTo(VT);
      unsigned NumVals = TLI.getNumElements(VT);
      N = DAG.getCopyFromReg(DAG.getEntryNode(), InReg, DestVT);
      if (NumVals == 1)
        N = DAG.getNode(ISD::BIT_CONVERT, VT, N);
      else {
        assert(NumVals == 2 && "1 to 4 (and more) expansion not implemented!");
        N = DAG.getNode(ISD::BUILD_PAIR, VT, N,
                       DAG.getCopyFromReg(DAG.getEntryNode(), InReg+1, DestVT));
      }
    } else {
      MVT::ValueType DestVT = TLI.getTypeToTransformTo(VT);
      N = DAG.getCopyFromReg(DAG.getEntryNode(), InReg, DestVT);
      if (TLI.getTypeAction(VT) == TargetLowering::Promote) // Promotion case
        N = MVT::isFloatingPoint(VT)
          ? DAG.getNode(ISD::FP_ROUND, VT, N)
          : DAG.getNode(ISD::TRUNCATE, VT, N);
    }
  } else {
    // Otherwise, if this is a vector, make it available as a generic vector
    // here.
    MVT::ValueType PTyElementVT, PTyLegalElementVT;
    const VectorType *PTy = cast<VectorType>(VTy);
    unsigned NE = TLI.getVectorTypeBreakdown(PTy, PTyElementVT,
                                             PTyLegalElementVT);

    // Build a VBUILD_VECTOR with the input registers.
    SmallVector<SDOperand, 8> Ops;
    if (PTyElementVT == PTyLegalElementVT) {
      // If the value types are legal, just VBUILD the CopyFromReg nodes.
      for (unsigned i = 0; i != NE; ++i)
        Ops.push_back(DAG.getCopyFromReg(DAG.getEntryNode(), InReg++, 
                                         PTyElementVT));
    } else if (PTyElementVT < PTyLegalElementVT) {
      // If the register was promoted, use TRUNCATE of FP_ROUND as appropriate.
      for (unsigned i = 0; i != NE; ++i) {
        SDOperand Op = DAG.getCopyFromReg(DAG.getEntryNode(), InReg++, 
                                          PTyElementVT);
        if (MVT::isFloatingPoint(PTyElementVT))
          Op = DAG.getNode(ISD::FP_ROUND, PTyElementVT, Op);
        else
          Op = DAG.getNode(ISD::TRUNCATE, PTyElementVT, Op);
        Ops.push_back(Op);
      }
    } else {
      // If the register was expanded, use BUILD_PAIR.
      assert((NE & 1) == 0 && "Must expand into a multiple of 2 elements!");
      for (unsigned i = 0; i != NE/2; ++i) {
        SDOperand Op0 = DAG.getCopyFromReg(DAG.getEntryNode(), InReg++, 
                                           PTyElementVT);
        SDOperand Op1 = DAG.getCopyFromReg(DAG.getEntryNode(), InReg++, 
                                           PTyElementVT);
        Ops.push_back(DAG.getNode(ISD::BUILD_PAIR, VT, Op0, Op1));
      }
    }
    
    Ops.push_back(DAG.getConstant(NE, MVT::i32));
    Ops.push_back(DAG.getValueType(PTyLegalElementVT));
    N = DAG.getNode(ISD::VBUILD_VECTOR, MVT::Vector, &Ops[0], Ops.size());
    
    // Finally, use a VBIT_CONVERT to make this available as the appropriate
    // vector type.
    N = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, N, 
                    DAG.getConstant(PTy->getNumElements(),
                                    MVT::i32),
                    DAG.getValueType(TLI.getValueType(PTy->getElementType())));
  }
  
  return N;
}


void SelectionDAGLowering::visitRet(ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other, getRoot()));
    return;
  }
  SmallVector<SDOperand, 8> NewValues;
  NewValues.push_back(getRoot());
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    SDOperand RetOp = getValue(I.getOperand(i));
    
    // If this is an integer return value, we need to promote it ourselves to
    // the full width of a register, since LegalizeOp will use ANY_EXTEND rather
    // than sign/zero.
    // FIXME: C calling convention requires the return type to be promoted to
    // at least 32-bit. But this is not necessary for non-C calling conventions.
    if (MVT::isInteger(RetOp.getValueType()) && 
        RetOp.getValueType() < MVT::i64) {
      MVT::ValueType TmpVT;
      if (TLI.getTypeAction(MVT::i32) == TargetLowering::Promote)
        TmpVT = TLI.getTypeToTransformTo(MVT::i32);
      else
        TmpVT = MVT::i32;
      const FunctionType *FTy = I.getParent()->getParent()->getFunctionType();
      const ParamAttrsList *Attrs = FTy->getParamAttrs();
      ISD::NodeType ExtendKind = ISD::ANY_EXTEND;
      if (Attrs && Attrs->paramHasAttr(0, ParamAttr::SExt))
        ExtendKind = ISD::SIGN_EXTEND;
      if (Attrs && Attrs->paramHasAttr(0, ParamAttr::ZExt))
        ExtendKind = ISD::ZERO_EXTEND;
      RetOp = DAG.getNode(ExtendKind, TmpVT, RetOp);
    }
    NewValues.push_back(RetOp);
    NewValues.push_back(DAG.getConstant(false, MVT::i32));
  }
  DAG.setRoot(DAG.getNode(ISD::RET, MVT::Other,
                          &NewValues[0], NewValues.size()));
}

/// ExportFromCurrentBlock - If this condition isn't known to be exported from
/// the current basic block, add it to ValueMap now so that we'll get a
/// CopyTo/FromReg.
void SelectionDAGLowering::ExportFromCurrentBlock(Value *V) {
  // No need to export constants.
  if (!isa<Instruction>(V) && !isa<Argument>(V)) return;
  
  // Already exported?
  if (FuncInfo.isExportedInst(V)) return;

  unsigned Reg = FuncInfo.InitializeRegForValue(V);
  PendingLoads.push_back(CopyValueToVirtualRegister(V, Reg));
}

bool SelectionDAGLowering::isExportableFromCurrentBlock(Value *V,
                                                    const BasicBlock *FromBB) {
  // The operands of the setcc have to be in this block.  We don't know
  // how to export them from some other block.
  if (Instruction *VI = dyn_cast<Instruction>(V)) {
    // Can export from current BB.
    if (VI->getParent() == FromBB)
      return true;
    
    // Is already exported, noop.
    return FuncInfo.isExportedInst(V);
  }
  
  // If this is an argument, we can export it if the BB is the entry block or
  // if it is already exported.
  if (isa<Argument>(V)) {
    if (FromBB == &FromBB->getParent()->getEntryBlock())
      return true;

    // Otherwise, can only export this if it is already exported.
    return FuncInfo.isExportedInst(V);
  }
  
  // Otherwise, constants can always be exported.
  return true;
}

static bool InBlock(const Value *V, const BasicBlock *BB) {
  if (const Instruction *I = dyn_cast<Instruction>(V))
    return I->getParent() == BB;
  return true;
}

/// FindMergedConditions - If Cond is an expression like 
void SelectionDAGLowering::FindMergedConditions(Value *Cond,
                                                MachineBasicBlock *TBB,
                                                MachineBasicBlock *FBB,
                                                MachineBasicBlock *CurBB,
                                                unsigned Opc) {
  // If this node is not part of the or/and tree, emit it as a branch.
  Instruction *BOp = dyn_cast<Instruction>(Cond);

  if (!BOp || !(isa<BinaryOperator>(BOp) || isa<CmpInst>(BOp)) || 
      (unsigned)BOp->getOpcode() != Opc || !BOp->hasOneUse() ||
      BOp->getParent() != CurBB->getBasicBlock() ||
      !InBlock(BOp->getOperand(0), CurBB->getBasicBlock()) ||
      !InBlock(BOp->getOperand(1), CurBB->getBasicBlock())) {
    const BasicBlock *BB = CurBB->getBasicBlock();
    
    // If the leaf of the tree is a comparison, merge the condition into 
    // the caseblock.
    if ((isa<ICmpInst>(Cond) || isa<FCmpInst>(Cond)) &&
        // The operands of the cmp have to be in this block.  We don't know
        // how to export them from some other block.  If this is the first block
        // of the sequence, no exporting is needed.
        (CurBB == CurMBB ||
         (isExportableFromCurrentBlock(BOp->getOperand(0), BB) &&
          isExportableFromCurrentBlock(BOp->getOperand(1), BB)))) {
      BOp = cast<Instruction>(Cond);
      ISD::CondCode Condition;
      if (ICmpInst *IC = dyn_cast<ICmpInst>(Cond)) {
        switch (IC->getPredicate()) {
        default: assert(0 && "Unknown icmp predicate opcode!");
        case ICmpInst::ICMP_EQ:  Condition = ISD::SETEQ;  break;
        case ICmpInst::ICMP_NE:  Condition = ISD::SETNE;  break;
        case ICmpInst::ICMP_SLE: Condition = ISD::SETLE;  break;
        case ICmpInst::ICMP_ULE: Condition = ISD::SETULE; break;
        case ICmpInst::ICMP_SGE: Condition = ISD::SETGE;  break;
        case ICmpInst::ICMP_UGE: Condition = ISD::SETUGE; break;
        case ICmpInst::ICMP_SLT: Condition = ISD::SETLT;  break;
        case ICmpInst::ICMP_ULT: Condition = ISD::SETULT; break;
        case ICmpInst::ICMP_SGT: Condition = ISD::SETGT;  break;
        case ICmpInst::ICMP_UGT: Condition = ISD::SETUGT; break;
        }
      } else if (FCmpInst *FC = dyn_cast<FCmpInst>(Cond)) {
        ISD::CondCode FPC, FOC;
        switch (FC->getPredicate()) {
        default: assert(0 && "Unknown fcmp predicate opcode!");
        case FCmpInst::FCMP_FALSE: FOC = FPC = ISD::SETFALSE; break;
        case FCmpInst::FCMP_OEQ:   FOC = ISD::SETEQ; FPC = ISD::SETOEQ; break;
        case FCmpInst::FCMP_OGT:   FOC = ISD::SETGT; FPC = ISD::SETOGT; break;
        case FCmpInst::FCMP_OGE:   FOC = ISD::SETGE; FPC = ISD::SETOGE; break;
        case FCmpInst::FCMP_OLT:   FOC = ISD::SETLT; FPC = ISD::SETOLT; break;
        case FCmpInst::FCMP_OLE:   FOC = ISD::SETLE; FPC = ISD::SETOLE; break;
        case FCmpInst::FCMP_ONE:   FOC = ISD::SETNE; FPC = ISD::SETONE; break;
        case FCmpInst::FCMP_ORD:   FOC = ISD::SETEQ; FPC = ISD::SETO;   break;
        case FCmpInst::FCMP_UNO:   FOC = ISD::SETNE; FPC = ISD::SETUO;  break;
        case FCmpInst::FCMP_UEQ:   FOC = ISD::SETEQ; FPC = ISD::SETUEQ; break;
        case FCmpInst::FCMP_UGT:   FOC = ISD::SETGT; FPC = ISD::SETUGT; break;
        case FCmpInst::FCMP_UGE:   FOC = ISD::SETGE; FPC = ISD::SETUGE; break;
        case FCmpInst::FCMP_ULT:   FOC = ISD::SETLT; FPC = ISD::SETULT; break;
        case FCmpInst::FCMP_ULE:   FOC = ISD::SETLE; FPC = ISD::SETULE; break;
        case FCmpInst::FCMP_UNE:   FOC = ISD::SETNE; FPC = ISD::SETUNE; break;
        case FCmpInst::FCMP_TRUE:  FOC = FPC = ISD::SETTRUE; break;
        }
        if (FiniteOnlyFPMath())
          Condition = FOC;
        else 
          Condition = FPC;
      } else {
        Condition = ISD::SETEQ; // silence warning.
        assert(0 && "Unknown compare instruction");
      }
      
      SelectionDAGISel::CaseBlock CB(Condition, BOp->getOperand(0), 
                                     BOp->getOperand(1), NULL, TBB, FBB, CurBB);
      SwitchCases.push_back(CB);
      return;
    }
    
    // Create a CaseBlock record representing this branch.
    SelectionDAGISel::CaseBlock CB(ISD::SETEQ, Cond, ConstantInt::getTrue(),
                                   NULL, TBB, FBB, CurBB);
    SwitchCases.push_back(CB);
    return;
  }
  
  
  //  Create TmpBB after CurBB.
  MachineFunction::iterator BBI = CurBB;
  MachineBasicBlock *TmpBB = new MachineBasicBlock(CurBB->getBasicBlock());
  CurBB->getParent()->getBasicBlockList().insert(++BBI, TmpBB);
  
  if (Opc == Instruction::Or) {
    // Codegen X | Y as:
    //   jmp_if_X TBB
    //   jmp TmpBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //
  
    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TBB, TmpBB, CurBB, Opc);
  
    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, Opc);
  } else {
    assert(Opc == Instruction::And && "Unknown merge op!");
    // Codegen X & Y as:
    //   jmp_if_X TmpBB
    //   jmp FBB
    // TmpBB:
    //   jmp_if_Y TBB
    //   jmp FBB
    //
    //  This requires creation of TmpBB after CurBB.
    
    // Emit the LHS condition.
    FindMergedConditions(BOp->getOperand(0), TmpBB, FBB, CurBB, Opc);
    
    // Emit the RHS condition into TmpBB.
    FindMergedConditions(BOp->getOperand(1), TBB, FBB, TmpBB, Opc);
  }
}

/// If the set of cases should be emitted as a series of branches, return true.
/// If we should emit this as a bunch of and/or'd together conditions, return
/// false.
static bool 
ShouldEmitAsBranches(const std::vector<SelectionDAGISel::CaseBlock> &Cases) {
  if (Cases.size() != 2) return true;
  
  // If this is two comparisons of the same values or'd or and'd together, they
  // will get folded into a single comparison, so don't emit two blocks.
  if ((Cases[0].CmpLHS == Cases[1].CmpLHS &&
       Cases[0].CmpRHS == Cases[1].CmpRHS) ||
      (Cases[0].CmpRHS == Cases[1].CmpLHS &&
       Cases[0].CmpLHS == Cases[1].CmpRHS)) {
    return false;
  }
  
  return true;
}

void SelectionDAGLowering::visitBr(BranchInst &I) {
  // Update machine-CFG edges.
  MachineBasicBlock *Succ0MBB = FuncInfo.MBBMap[I.getSuccessor(0)];

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  if (I.isUnconditional()) {
    // If this is not a fall-through branch, emit the branch.
    if (Succ0MBB != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getRoot(),
                              DAG.getBasicBlock(Succ0MBB)));

    // Update machine-CFG edges.
    CurMBB->addSuccessor(Succ0MBB);

    return;
  }

  // If this condition is one of the special cases we handle, do special stuff
  // now.
  Value *CondVal = I.getCondition();
  MachineBasicBlock *Succ1MBB = FuncInfo.MBBMap[I.getSuccessor(1)];

  // If this is a series of conditions that are or'd or and'd together, emit
  // this as a sequence of branches instead of setcc's with and/or operations.
  // For example, instead of something like:
  //     cmp A, B
  //     C = seteq 
  //     cmp D, E
  //     F = setle 
  //     or C, F
  //     jnz foo
  // Emit:
  //     cmp A, B
  //     je foo
  //     cmp D, E
  //     jle foo
  //
  if (BinaryOperator *BOp = dyn_cast<BinaryOperator>(CondVal)) {
    if (BOp->hasOneUse() && 
        (BOp->getOpcode() == Instruction::And ||
         BOp->getOpcode() == Instruction::Or)) {
      FindMergedConditions(BOp, Succ0MBB, Succ1MBB, CurMBB, BOp->getOpcode());
      // If the compares in later blocks need to use values not currently
      // exported from this block, export them now.  This block should always
      // be the first entry.
      assert(SwitchCases[0].ThisBB == CurMBB && "Unexpected lowering!");
      
      // Allow some cases to be rejected.
      if (ShouldEmitAsBranches(SwitchCases)) {
        for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i) {
          ExportFromCurrentBlock(SwitchCases[i].CmpLHS);
          ExportFromCurrentBlock(SwitchCases[i].CmpRHS);
        }
        
        // Emit the branch for this block.
        visitSwitchCase(SwitchCases[0]);
        SwitchCases.erase(SwitchCases.begin());
        return;
      }
      
      // Okay, we decided not to do this, remove any inserted MBB's and clear
      // SwitchCases.
      for (unsigned i = 1, e = SwitchCases.size(); i != e; ++i)
        CurMBB->getParent()->getBasicBlockList().erase(SwitchCases[i].ThisBB);
      
      SwitchCases.clear();
    }
  }
  
  // Create a CaseBlock record representing this branch.
  SelectionDAGISel::CaseBlock CB(ISD::SETEQ, CondVal, ConstantInt::getTrue(),
                                 NULL, Succ0MBB, Succ1MBB, CurMBB);
  // Use visitSwitchCase to actually insert the fast branch sequence for this
  // cond branch.
  visitSwitchCase(CB);
}

/// visitSwitchCase - Emits the necessary code to represent a single node in
/// the binary search tree resulting from lowering a switch instruction.
void SelectionDAGLowering::visitSwitchCase(SelectionDAGISel::CaseBlock &CB) {
  SDOperand Cond;
  SDOperand CondLHS = getValue(CB.CmpLHS);
  
  // Build the setcc now. 
  if (CB.CmpMHS == NULL) {
    // Fold "(X == true)" to X and "(X == false)" to !X to
    // handle common cases produced by branch lowering.
    if (CB.CmpRHS == ConstantInt::getTrue() && CB.CC == ISD::SETEQ)
      Cond = CondLHS;
    else if (CB.CmpRHS == ConstantInt::getFalse() && CB.CC == ISD::SETEQ) {
      SDOperand True = DAG.getConstant(1, CondLHS.getValueType());
      Cond = DAG.getNode(ISD::XOR, CondLHS.getValueType(), CondLHS, True);
    } else
      Cond = DAG.getSetCC(MVT::i1, CondLHS, getValue(CB.CmpRHS), CB.CC);
  } else {
    assert(CB.CC == ISD::SETLE && "Can handle only LE ranges now");

    uint64_t Low = cast<ConstantInt>(CB.CmpLHS)->getSExtValue();
    uint64_t High  = cast<ConstantInt>(CB.CmpRHS)->getSExtValue();

    SDOperand CmpOp = getValue(CB.CmpMHS);
    MVT::ValueType VT = CmpOp.getValueType();

    if (cast<ConstantInt>(CB.CmpLHS)->isMinValue(true)) {
      Cond = DAG.getSetCC(MVT::i1, CmpOp, DAG.getConstant(High, VT), ISD::SETLE);
    } else {
      SDOperand SUB = DAG.getNode(ISD::SUB, VT, CmpOp, DAG.getConstant(Low, VT));
      Cond = DAG.getSetCC(MVT::i1, SUB,
                          DAG.getConstant(High-Low, VT), ISD::SETULE);
    }
    
  }
  
  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;
  
  // If the lhs block is the next block, invert the condition so that we can
  // fall through to the lhs instead of the rhs block.
  if (CB.TrueBB == NextBlock) {
    std::swap(CB.TrueBB, CB.FalseBB);
    SDOperand True = DAG.getConstant(1, Cond.getValueType());
    Cond = DAG.getNode(ISD::XOR, Cond.getValueType(), Cond, True);
  }
  SDOperand BrCond = DAG.getNode(ISD::BRCOND, MVT::Other, getRoot(), Cond,
                                 DAG.getBasicBlock(CB.TrueBB));
  if (CB.FalseBB == NextBlock)
    DAG.setRoot(BrCond);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, BrCond, 
                            DAG.getBasicBlock(CB.FalseBB)));
  // Update successor info
  CurMBB->addSuccessor(CB.TrueBB);
  CurMBB->addSuccessor(CB.FalseBB);
}

/// visitJumpTable - Emit JumpTable node in the current MBB
void SelectionDAGLowering::visitJumpTable(SelectionDAGISel::JumpTable &JT) {
  // Emit the code for the jump table
  assert(JT.Reg != -1U && "Should lower JT Header first!");
  MVT::ValueType PTy = TLI.getPointerTy();
  SDOperand Index = DAG.getCopyFromReg(getRoot(), JT.Reg, PTy);
  SDOperand Table = DAG.getJumpTable(JT.JTI, PTy);
  DAG.setRoot(DAG.getNode(ISD::BR_JT, MVT::Other, Index.getValue(1),
                          Table, Index));
  return;
}

/// visitJumpTableHeader - This function emits necessary code to produce index
/// in the JumpTable from switch case.
void SelectionDAGLowering::visitJumpTableHeader(SelectionDAGISel::JumpTable &JT,
                                         SelectionDAGISel::JumpTableHeader &JTH) {
  // Subtract the lowest switch case value from the value being switched on
  // and conditional branch to default mbb if the result is greater than the
  // difference between smallest and largest cases.
  SDOperand SwitchOp = getValue(JTH.SValue);
  MVT::ValueType VT = SwitchOp.getValueType();
  SDOperand SUB = DAG.getNode(ISD::SUB, VT, SwitchOp,
                              DAG.getConstant(JTH.First, VT));
  
  // The SDNode we just created, which holds the value being switched on
  // minus the the smallest case value, needs to be copied to a virtual
  // register so it can be used as an index into the jump table in a 
  // subsequent basic block.  This value may be smaller or larger than the
  // target's pointer type, and therefore require extension or truncating.
  if (VT > TLI.getPointerTy())
    SwitchOp = DAG.getNode(ISD::TRUNCATE, TLI.getPointerTy(), SUB);
  else
    SwitchOp = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(), SUB);
  
  unsigned JumpTableReg = FuncInfo.MakeReg(TLI.getPointerTy());
  SDOperand CopyTo = DAG.getCopyToReg(getRoot(), JumpTableReg, SwitchOp);
  JT.Reg = JumpTableReg;

  // Emit the range check for the jump table, and branch to the default
  // block for the switch statement if the value being switched on exceeds
  // the largest case in the switch.
  SDOperand CMP = DAG.getSetCC(TLI.getSetCCResultTy(), SUB,
                               DAG.getConstant(JTH.Last-JTH.First,VT),
                               ISD::SETUGT);

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  SDOperand BrCond = DAG.getNode(ISD::BRCOND, MVT::Other, CopyTo, CMP,
                                 DAG.getBasicBlock(JT.Default));

  if (JT.MBB == NextBlock)
    DAG.setRoot(BrCond);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, BrCond, 
                            DAG.getBasicBlock(JT.MBB)));

  return;
}

/// visitBitTestHeader - This function emits necessary code to produce value
/// suitable for "bit tests"
void SelectionDAGLowering::visitBitTestHeader(SelectionDAGISel::BitTestBlock &B) {
  // Subtract the minimum value
  SDOperand SwitchOp = getValue(B.SValue);
  MVT::ValueType VT = SwitchOp.getValueType();
  SDOperand SUB = DAG.getNode(ISD::SUB, VT, SwitchOp,
                              DAG.getConstant(B.First, VT));

  // Check range
  SDOperand RangeCmp = DAG.getSetCC(TLI.getSetCCResultTy(), SUB,
                                    DAG.getConstant(B.Range, VT),
                                    ISD::SETUGT);

  SDOperand ShiftOp;
  if (VT > TLI.getShiftAmountTy())
    ShiftOp = DAG.getNode(ISD::TRUNCATE, TLI.getShiftAmountTy(), SUB);
  else
    ShiftOp = DAG.getNode(ISD::ZERO_EXTEND, TLI.getShiftAmountTy(), SUB);

  // Make desired shift
  SDOperand SwitchVal = DAG.getNode(ISD::SHL, TLI.getPointerTy(),
                                    DAG.getConstant(1, TLI.getPointerTy()),
                                    ShiftOp);

  unsigned SwitchReg = FuncInfo.MakeReg(TLI.getPointerTy());
  SDOperand CopyTo = DAG.getCopyToReg(getRoot(), SwitchReg, SwitchVal);
  B.Reg = SwitchReg;

  SDOperand BrRange = DAG.getNode(ISD::BRCOND, MVT::Other, CopyTo, RangeCmp,
                                  DAG.getBasicBlock(B.Default));

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  MachineBasicBlock* MBB = B.Cases[0].ThisBB;
  if (MBB == NextBlock)
    DAG.setRoot(BrRange);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, CopyTo,
                            DAG.getBasicBlock(MBB)));

  CurMBB->addSuccessor(B.Default);
  CurMBB->addSuccessor(MBB);

  return;
}

/// visitBitTestCase - this function produces one "bit test"
void SelectionDAGLowering::visitBitTestCase(MachineBasicBlock* NextMBB,
                                            unsigned Reg,
                                            SelectionDAGISel::BitTestCase &B) {
  // Emit bit tests and jumps
  SDOperand SwitchVal = DAG.getCopyFromReg(getRoot(), Reg, TLI.getPointerTy());
  
  SDOperand AndOp = DAG.getNode(ISD::AND, TLI.getPointerTy(),
                                SwitchVal,
                                DAG.getConstant(B.Mask,
                                                TLI.getPointerTy()));
  SDOperand AndCmp = DAG.getSetCC(TLI.getSetCCResultTy(), AndOp,
                                  DAG.getConstant(0, TLI.getPointerTy()),
                                  ISD::SETNE);
  SDOperand BrAnd = DAG.getNode(ISD::BRCOND, MVT::Other, getRoot(),
                                AndCmp, DAG.getBasicBlock(B.TargetBB));

  // Set NextBlock to be the MBB immediately after the current one, if any.
  // This is used to avoid emitting unnecessary branches to the next block.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;
  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  if (NextMBB == NextBlock)
    DAG.setRoot(BrAnd);
  else
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, BrAnd,
                            DAG.getBasicBlock(NextMBB)));

  CurMBB->addSuccessor(B.TargetBB);
  CurMBB->addSuccessor(NextMBB);

  return;
}

void SelectionDAGLowering::visitInvoke(InvokeInst &I) {
  assert(0 && "Should never be visited directly");
}
void SelectionDAGLowering::visitInvoke(InvokeInst &I, bool AsTerminator) {
  // Retrieve successors.
  MachineBasicBlock *Return = FuncInfo.MBBMap[I.getSuccessor(0)];
  MachineBasicBlock *LandingPad = FuncInfo.MBBMap[I.getSuccessor(1)];
  
  if (!AsTerminator) {
    // Mark landing pad so that it doesn't get deleted in branch folding.
    LandingPad->setIsLandingPad();
    
    // Insert a label before the invoke call to mark the try range.
    // This can be used to detect deletion of the invoke via the
    // MachineModuleInfo.
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    unsigned BeginLabel = MMI->NextLabelID();
    DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getRoot(),
                            DAG.getConstant(BeginLabel, MVT::i32)));

    LowerCallTo(I, I.getCalledValue()->getType(),
                   I.getCallingConv(),
                   false,
                   getValue(I.getOperand(0)),
                   3);

    // Insert a label before the invoke call to mark the try range.
    // This can be used to detect deletion of the invoke via the
    // MachineModuleInfo.
    unsigned EndLabel = MMI->NextLabelID();
    DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getRoot(),
                            DAG.getConstant(EndLabel, MVT::i32)));
                            
    // Inform MachineModuleInfo of range.    
    MMI->addInvoke(LandingPad, BeginLabel, EndLabel);
                            
    // Update successor info
    CurMBB->addSuccessor(Return);
    CurMBB->addSuccessor(LandingPad);
  } else {
    // Drop into normal successor.
    DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getRoot(), 
                            DAG.getBasicBlock(Return)));
  }
}

void SelectionDAGLowering::visitUnwind(UnwindInst &I) {
}

/// handleSmallSwitchCaseRange - Emit a series of specific tests (suitable for
/// small case ranges).
bool SelectionDAGLowering::handleSmallSwitchRange(CaseRec& CR,
                                                  CaseRecVector& WorkList,
                                                  Value* SV,
                                                  MachineBasicBlock* Default) {
  Case& BackCase  = *(CR.Range.second-1);
  
  // Size is the number of Cases represented by this range.
  unsigned Size = CR.Range.second - CR.Range.first;
  if (Size > 3)
    return false;  
  
  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();  

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  // TODO: If any two of the cases has the same destination, and if one value
  // is the same as the other, but has one bit unset that the other has set,
  // use bit manipulation to do two compares at once.  For example:
  // "if (X == 6 || X == 4)" -> "if ((X|2) == 6)"
    
  // Rearrange the case blocks so that the last one falls through if possible.
  if (NextBlock && Default != NextBlock && BackCase.BB != NextBlock) {
    // The last case block won't fall through into 'NextBlock' if we emit the
    // branches in this order.  See if rearranging a case value would help.
    for (CaseItr I = CR.Range.first, E = CR.Range.second-1; I != E; ++I) {
      if (I->BB == NextBlock) {
        std::swap(*I, BackCase);
        break;
      }
    }
  }
  
  // Create a CaseBlock record representing a conditional branch to
  // the Case's target mbb if the value being switched on SV is equal
  // to C.
  MachineBasicBlock *CurBlock = CR.CaseBB;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++I) {
    MachineBasicBlock *FallThrough;
    if (I != E-1) {
      FallThrough = new MachineBasicBlock(CurBlock->getBasicBlock());
      CurMF->getBasicBlockList().insert(BBI, FallThrough);
    } else {
      // If the last case doesn't match, go to the default block.
      FallThrough = Default;
    }

    Value *RHS, *LHS, *MHS;
    ISD::CondCode CC;
    if (I->High == I->Low) {
      // This is just small small case range :) containing exactly 1 case
      CC = ISD::SETEQ;
      LHS = SV; RHS = I->High; MHS = NULL;
    } else {
      CC = ISD::SETLE;
      LHS = I->Low; MHS = SV; RHS = I->High;
    }
    SelectionDAGISel::CaseBlock CB(CC, LHS, RHS, MHS,
                                   I->BB, FallThrough, CurBlock);
    
    // If emitting the first comparison, just call visitSwitchCase to emit the
    // code into the current block.  Otherwise, push the CaseBlock onto the
    // vector to be later processed by SDISel, and insert the node's MBB
    // before the next MBB.
    if (CurBlock == CurMBB)
      visitSwitchCase(CB);
    else
      SwitchCases.push_back(CB);
    
    CurBlock = FallThrough;
  }

  return true;
}

static inline bool areJTsAllowed(const TargetLowering &TLI) {
  return (TLI.isOperationLegal(ISD::BR_JT, MVT::Other) ||
          TLI.isOperationLegal(ISD::BRIND, MVT::Other));
}
  
/// handleJTSwitchCase - Emit jumptable for current switch case range
bool SelectionDAGLowering::handleJTSwitchCase(CaseRec& CR,
                                              CaseRecVector& WorkList,
                                              Value* SV,
                                              MachineBasicBlock* Default) {
  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);

  int64_t First = cast<ConstantInt>(FrontCase.Low)->getSExtValue();
  int64_t Last  = cast<ConstantInt>(BackCase.High)->getSExtValue();

  uint64_t TSize = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I)
    TSize += I->size();

  if (!areJTsAllowed(TLI) || TSize <= 3)
    return false;
  
  double Density = (double)TSize / (double)((Last - First) + 1ULL);  
  if (Density < 0.4)
    return false;

  DOUT << "Lowering jump table\n"
       << "First entry: " << First << ". Last entry: " << Last << "\n"
       << "Size: " << TSize << ". Density: " << Density << "\n\n";

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  // Create a new basic block to hold the code for loading the address
  // of the jump table, and jumping to it.  Update successor information;
  // we will either branch to the default case for the switch, or the jump
  // table.
  MachineBasicBlock *JumpTableBB = new MachineBasicBlock(LLVMBB);
  CurMF->getBasicBlockList().insert(BBI, JumpTableBB);
  CR.CaseBB->addSuccessor(Default);
  CR.CaseBB->addSuccessor(JumpTableBB);
                
  // Build a vector of destination BBs, corresponding to each target
  // of the jump table. If the value of the jump table slot corresponds to
  // a case statement, push the case's BB onto the vector, otherwise, push
  // the default BB.
  std::vector<MachineBasicBlock*> DestBBs;
  int64_t TEI = First;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I != E; ++TEI) {
    int64_t Low = cast<ConstantInt>(I->Low)->getSExtValue();
    int64_t High = cast<ConstantInt>(I->High)->getSExtValue();
    
    if ((Low <= TEI) && (TEI <= High)) {
      DestBBs.push_back(I->BB);
      if (TEI==High)
        ++I;
    } else {
      DestBBs.push_back(Default);
    }
  }
  
  // Update successor info. Add one edge to each unique successor.
  BitVector SuccsHandled(CR.CaseBB->getParent()->getNumBlockIDs());  
  for (std::vector<MachineBasicBlock*>::iterator I = DestBBs.begin(), 
         E = DestBBs.end(); I != E; ++I) {
    if (!SuccsHandled[(*I)->getNumber()]) {
      SuccsHandled[(*I)->getNumber()] = true;
      JumpTableBB->addSuccessor(*I);
    }
  }
      
  // Create a jump table index for this jump table, or return an existing
  // one.
  unsigned JTI = CurMF->getJumpTableInfo()->getJumpTableIndex(DestBBs);
  
  // Set the jump table information so that we can codegen it as a second
  // MachineBasicBlock
  SelectionDAGISel::JumpTable JT(-1U, JTI, JumpTableBB, Default);
  SelectionDAGISel::JumpTableHeader JTH(First, Last, SV, CR.CaseBB,
                                        (CR.CaseBB == CurMBB));
  if (CR.CaseBB == CurMBB)
    visitJumpTableHeader(JT, JTH);
        
  JTCases.push_back(SelectionDAGISel::JumpTableBlock(JTH, JT));

  return true;
}

/// handleBTSplitSwitchCase - emit comparison and split binary search tree into
/// 2 subtrees.
bool SelectionDAGLowering::handleBTSplitSwitchCase(CaseRec& CR,
                                                   CaseRecVector& WorkList,
                                                   Value* SV,
                                                   MachineBasicBlock* Default) {
  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();  

  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CR.CaseBB;

  if (++BBI != CurMBB->getParent()->end())
    NextBlock = BBI;

  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);
  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  // Size is the number of Cases represented by this range.
  unsigned Size = CR.Range.second - CR.Range.first;

  int64_t First = cast<ConstantInt>(FrontCase.Low)->getSExtValue();
  int64_t Last  = cast<ConstantInt>(BackCase.High)->getSExtValue();
  double FMetric = 0;
  CaseItr Pivot = CR.Range.first + Size/2;

  // Select optimal pivot, maximizing sum density of LHS and RHS. This will
  // (heuristically) allow us to emit JumpTable's later.
  uint64_t TSize = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I)
    TSize += I->size();

  uint64_t LSize = FrontCase.size();
  uint64_t RSize = TSize-LSize;
  DOUT << "Selecting best pivot: \n"
       << "First: " << First << ", Last: " << Last <<"\n"
       << "LSize: " << LSize << ", RSize: " << RSize << "\n";
  for (CaseItr I = CR.Range.first, J=I+1, E = CR.Range.second;
       J!=E; ++I, ++J) {
    int64_t LEnd = cast<ConstantInt>(I->High)->getSExtValue();
    int64_t RBegin = cast<ConstantInt>(J->Low)->getSExtValue();
    assert((RBegin-LEnd>=1) && "Invalid case distance");
    double LDensity = (double)LSize / (double)((LEnd - First) + 1ULL);
    double RDensity = (double)RSize / (double)((Last - RBegin) + 1ULL);
    double Metric = Log2_64(RBegin-LEnd)*(LDensity+RDensity);
    // Should always split in some non-trivial place
    DOUT <<"=>Step\n"
         << "LEnd: " << LEnd << ", RBegin: " << RBegin << "\n"
         << "LDensity: " << LDensity << ", RDensity: " << RDensity << "\n"
         << "Metric: " << Metric << "\n"; 
    if (FMetric < Metric) {
      Pivot = J;
      FMetric = Metric;
      DOUT << "Current metric set to: " << FMetric << "\n";
    }

    LSize += J->size();
    RSize -= J->size();
  }
  if (areJTsAllowed(TLI)) {
    // If our case is dense we *really* should handle it earlier!
    assert((FMetric > 0) && "Should handle dense range earlier!");
  } else {
    Pivot = CR.Range.first + Size/2;
  }
  
  CaseRange LHSR(CR.Range.first, Pivot);
  CaseRange RHSR(Pivot, CR.Range.second);
  Constant *C = Pivot->Low;
  MachineBasicBlock *FalseBB = 0, *TrueBB = 0;
      
  // We know that we branch to the LHS if the Value being switched on is
  // less than the Pivot value, C.  We use this to optimize our binary 
  // tree a bit, by recognizing that if SV is greater than or equal to the
  // LHS's Case Value, and that Case Value is exactly one less than the 
  // Pivot's Value, then we can branch directly to the LHS's Target,
  // rather than creating a leaf node for it.
  if ((LHSR.second - LHSR.first) == 1 &&
      LHSR.first->High == CR.GE &&
      cast<ConstantInt>(C)->getSExtValue() ==
      (cast<ConstantInt>(CR.GE)->getSExtValue() + 1LL)) {
    TrueBB = LHSR.first->BB;
  } else {
    TrueBB = new MachineBasicBlock(LLVMBB);
    CurMF->getBasicBlockList().insert(BBI, TrueBB);
    WorkList.push_back(CaseRec(TrueBB, C, CR.GE, LHSR));
  }
  
  // Similar to the optimization above, if the Value being switched on is
  // known to be less than the Constant CR.LT, and the current Case Value
  // is CR.LT - 1, then we can branch directly to the target block for
  // the current Case Value, rather than emitting a RHS leaf node for it.
  if ((RHSR.second - RHSR.first) == 1 && CR.LT &&
      cast<ConstantInt>(RHSR.first->Low)->getSExtValue() ==
      (cast<ConstantInt>(CR.LT)->getSExtValue() - 1LL)) {
    FalseBB = RHSR.first->BB;
  } else {
    FalseBB = new MachineBasicBlock(LLVMBB);
    CurMF->getBasicBlockList().insert(BBI, FalseBB);
    WorkList.push_back(CaseRec(FalseBB,CR.LT,C,RHSR));
  }

  // Create a CaseBlock record representing a conditional branch to
  // the LHS node if the value being switched on SV is less than C. 
  // Otherwise, branch to LHS.
  SelectionDAGISel::CaseBlock CB(ISD::SETLT, SV, C, NULL,
                                 TrueBB, FalseBB, CR.CaseBB);

  if (CR.CaseBB == CurMBB)
    visitSwitchCase(CB);
  else
    SwitchCases.push_back(CB);

  return true;
}

/// handleBitTestsSwitchCase - if current case range has few destination and
/// range span less, than machine word bitwidth, encode case range into series
/// of masks and emit bit tests with these masks.
bool SelectionDAGLowering::handleBitTestsSwitchCase(CaseRec& CR,
                                                    CaseRecVector& WorkList,
                                                    Value* SV,
                                                    MachineBasicBlock* Default){
  unsigned IntPtrBits = getSizeInBits(TLI.getPointerTy());

  Case& FrontCase = *CR.Range.first;
  Case& BackCase  = *(CR.Range.second-1);

  // Get the MachineFunction which holds the current MBB.  This is used when
  // inserting any additional MBBs necessary to represent the switch.
  MachineFunction *CurMF = CurMBB->getParent();  

  unsigned numCmps = 0;
  for (CaseItr I = CR.Range.first, E = CR.Range.second;
       I!=E; ++I) {
    // Single case counts one, case range - two.
    if (I->Low == I->High)
      numCmps +=1;
    else
      numCmps +=2;
  }
    
  // Count unique destinations
  SmallSet<MachineBasicBlock*, 4> Dests;
  for (CaseItr I = CR.Range.first, E = CR.Range.second; I!=E; ++I) {
    Dests.insert(I->BB);
    if (Dests.size() > 3)
      // Don't bother the code below, if there are too much unique destinations
      return false;
  }
  DOUT << "Total number of unique destinations: " << Dests.size() << "\n"
       << "Total number of comparisons: " << numCmps << "\n";
  
  // Compute span of values.
  Constant* minValue = FrontCase.Low;
  Constant* maxValue = BackCase.High;
  uint64_t range = cast<ConstantInt>(maxValue)->getSExtValue() -
                   cast<ConstantInt>(minValue)->getSExtValue();
  DOUT << "Compare range: " << range << "\n"
       << "Low bound: " << cast<ConstantInt>(minValue)->getSExtValue() << "\n"
       << "High bound: " << cast<ConstantInt>(maxValue)->getSExtValue() << "\n";
  
  if (range>=IntPtrBits ||
      (!(Dests.size() == 1 && numCmps >= 3) &&
       !(Dests.size() == 2 && numCmps >= 5) &&
       !(Dests.size() >= 3 && numCmps >= 6)))
    return false;
  
  DOUT << "Emitting bit tests\n";
  int64_t lowBound = 0;
    
  // Optimize the case where all the case values fit in a
  // word without having to subtract minValue. In this case,
  // we can optimize away the subtraction.
  if (cast<ConstantInt>(minValue)->getSExtValue() >= 0 &&
      cast<ConstantInt>(maxValue)->getSExtValue() <  IntPtrBits) {
    range = cast<ConstantInt>(maxValue)->getSExtValue();
  } else {
    lowBound = cast<ConstantInt>(minValue)->getSExtValue();
  }
    
  CaseBitsVector CasesBits;
  unsigned i, count = 0;

  for (CaseItr I = CR.Range.first, E = CR.Range.second; I!=E; ++I) {
    MachineBasicBlock* Dest = I->BB;
    for (i = 0; i < count; ++i)
      if (Dest == CasesBits[i].BB)
        break;
    
    if (i == count) {
      assert((count < 3) && "Too much destinations to test!");
      CasesBits.push_back(CaseBits(0, Dest, 0));
      count++;
    }
    
    uint64_t lo = cast<ConstantInt>(I->Low)->getSExtValue() - lowBound;
    uint64_t hi = cast<ConstantInt>(I->High)->getSExtValue() - lowBound;
    
    for (uint64_t j = lo; j <= hi; j++) {
      CasesBits[i].Mask |=  1ULL << j;
      CasesBits[i].Bits++;
    }
      
  }
  std::sort(CasesBits.begin(), CasesBits.end(), CaseBitsCmp());
  
  SelectionDAGISel::BitTestInfo BTC;

  // Figure out which block is immediately after the current one.
  MachineFunction::iterator BBI = CR.CaseBB;
  ++BBI;

  const BasicBlock *LLVMBB = CR.CaseBB->getBasicBlock();

  DOUT << "Cases:\n";
  for (unsigned i = 0, e = CasesBits.size(); i!=e; ++i) {
    DOUT << "Mask: " << CasesBits[i].Mask << ", Bits: " << CasesBits[i].Bits
         << ", BB: " << CasesBits[i].BB << "\n";

    MachineBasicBlock *CaseBB = new MachineBasicBlock(LLVMBB);
    CurMF->getBasicBlockList().insert(BBI, CaseBB);
    BTC.push_back(SelectionDAGISel::BitTestCase(CasesBits[i].Mask,
                                                CaseBB,
                                                CasesBits[i].BB));
  }
  
  SelectionDAGISel::BitTestBlock BTB(lowBound, range, SV,
                                     -1U, (CR.CaseBB == CurMBB),
                                     CR.CaseBB, Default, BTC);

  if (CR.CaseBB == CurMBB)
    visitBitTestHeader(BTB);
  
  BitTestCases.push_back(BTB);

  return true;
}


// Clusterify - Transform simple list of Cases into list of CaseRange's
unsigned SelectionDAGLowering::Clusterify(CaseVector& Cases,
                                          const SwitchInst& SI) {
  unsigned numCmps = 0;

  // Start with "simple" cases
  for (unsigned i = 1; i < SI.getNumSuccessors(); ++i) {
    MachineBasicBlock *SMBB = FuncInfo.MBBMap[SI.getSuccessor(i)];
    Cases.push_back(Case(SI.getSuccessorValue(i),
                         SI.getSuccessorValue(i),
                         SMBB));
  }
  sort(Cases.begin(), Cases.end(), CaseCmp());

  // Merge case into clusters
  if (Cases.size()>=2)
    for (CaseItr I=Cases.begin(), J=++(Cases.begin()), E=Cases.end(); J!=E; ) {
      int64_t nextValue = cast<ConstantInt>(J->Low)->getSExtValue();
      int64_t currentValue = cast<ConstantInt>(I->High)->getSExtValue();
      MachineBasicBlock* nextBB = J->BB;
      MachineBasicBlock* currentBB = I->BB;

      // If the two neighboring cases go to the same destination, merge them
      // into a single case.
      if ((nextValue-currentValue==1) && (currentBB == nextBB)) {
        I->High = J->High;
        J = Cases.erase(J);
      } else {
        I = J++;
      }
    }

  for (CaseItr I=Cases.begin(), E=Cases.end(); I!=E; ++I, ++numCmps) {
    if (I->Low != I->High)
      // A range counts double, since it requires two compares.
      ++numCmps;
  }

  return numCmps;
}

void SelectionDAGLowering::visitSwitch(SwitchInst &SI) {  
  // Figure out which block is immediately after the current one.
  MachineBasicBlock *NextBlock = 0;
  MachineFunction::iterator BBI = CurMBB;

  MachineBasicBlock *Default = FuncInfo.MBBMap[SI.getDefaultDest()];

  // If there is only the default destination, branch to it if it is not the
  // next basic block.  Otherwise, just fall through.
  if (SI.getNumOperands() == 2) {
    // Update machine-CFG edges.

    // If this is not a fall-through branch, emit the branch.
    if (Default != NextBlock)
      DAG.setRoot(DAG.getNode(ISD::BR, MVT::Other, getRoot(),
                              DAG.getBasicBlock(Default)));

    CurMBB->addSuccessor(Default);
    return;
  }
  
  // If there are any non-default case statements, create a vector of Cases
  // representing each one, and sort the vector so that we can efficiently
  // create a binary search tree from them.
  CaseVector Cases;
  unsigned numCmps = Clusterify(Cases, SI);
  DOUT << "Clusterify finished. Total clusters: " << Cases.size()
       << ". Total compares: " << numCmps << "\n";

  // Get the Value to be switched on and default basic blocks, which will be
  // inserted into CaseBlock records, representing basic blocks in the binary
  // search tree.
  Value *SV = SI.getOperand(0);

  // Push the initial CaseRec onto the worklist
  CaseRecVector WorkList;
  WorkList.push_back(CaseRec(CurMBB,0,0,CaseRange(Cases.begin(),Cases.end())));

  while (!WorkList.empty()) {
    // Grab a record representing a case range to process off the worklist
    CaseRec CR = WorkList.back();
    WorkList.pop_back();

    if (handleBitTestsSwitchCase(CR, WorkList, SV, Default))
      continue;
    
    // If the range has few cases (two or less) emit a series of specific
    // tests.
    if (handleSmallSwitchRange(CR, WorkList, SV, Default))
      continue;
    
    // If the switch has more than 5 blocks, and at least 40% dense, and the 
    // target supports indirect branches, then emit a jump table rather than 
    // lowering the switch to a binary tree of conditional branches.
    if (handleJTSwitchCase(CR, WorkList, SV, Default))
      continue;
          
    // Emit binary tree. We need to pick a pivot, and push left and right ranges
    // onto the worklist. Leafs are handled via handleSmallSwitchRange() call.
    handleBTSplitSwitchCase(CR, WorkList, SV, Default);
  }
}


void SelectionDAGLowering::visitSub(User &I) {
  // -0.0 - X --> fneg
  const Type *Ty = I.getType();
  if (isa<VectorType>(Ty)) {
    visitVectorBinary(I, ISD::VSUB);
  } else if (Ty->isFloatingPoint()) {
    if (ConstantFP *CFP = dyn_cast<ConstantFP>(I.getOperand(0)))
      if (CFP->isExactlyValue(-0.0)) {
        SDOperand Op2 = getValue(I.getOperand(1));
        setValue(&I, DAG.getNode(ISD::FNEG, Op2.getValueType(), Op2));
        return;
      }
    visitScalarBinary(I, ISD::FSUB);
  } else 
    visitScalarBinary(I, ISD::SUB);
}

void SelectionDAGLowering::visitScalarBinary(User &I, unsigned OpCode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  
  setValue(&I, DAG.getNode(OpCode, Op1.getValueType(), Op1, Op2));
}

void
SelectionDAGLowering::visitVectorBinary(User &I, unsigned OpCode) {
  assert(isa<VectorType>(I.getType()));
  const VectorType *Ty = cast<VectorType>(I.getType());
  SDOperand Typ = DAG.getValueType(TLI.getValueType(Ty->getElementType()));

  setValue(&I, DAG.getNode(OpCode, MVT::Vector,
                           getValue(I.getOperand(0)),
                           getValue(I.getOperand(1)),
                           DAG.getConstant(Ty->getNumElements(), MVT::i32),
                           Typ));
}

void SelectionDAGLowering::visitEitherBinary(User &I, unsigned ScalarOp,
                                             unsigned VectorOp) {
  if (isa<VectorType>(I.getType()))
    visitVectorBinary(I, VectorOp);
  else
    visitScalarBinary(I, ScalarOp);
}

void SelectionDAGLowering::visitShift(User &I, unsigned Opcode) {
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  
  if (TLI.getShiftAmountTy() < Op2.getValueType())
    Op2 = DAG.getNode(ISD::TRUNCATE, TLI.getShiftAmountTy(), Op2);
  else if (TLI.getShiftAmountTy() > Op2.getValueType())
    Op2 = DAG.getNode(ISD::ANY_EXTEND, TLI.getShiftAmountTy(), Op2);
  
  setValue(&I, DAG.getNode(Opcode, Op1.getValueType(), Op1, Op2));
}

void SelectionDAGLowering::visitICmp(User &I) {
  ICmpInst::Predicate predicate = ICmpInst::BAD_ICMP_PREDICATE;
  if (ICmpInst *IC = dyn_cast<ICmpInst>(&I))
    predicate = IC->getPredicate();
  else if (ConstantExpr *IC = dyn_cast<ConstantExpr>(&I))
    predicate = ICmpInst::Predicate(IC->getPredicate());
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Opcode;
  switch (predicate) {
    case ICmpInst::ICMP_EQ  : Opcode = ISD::SETEQ; break;
    case ICmpInst::ICMP_NE  : Opcode = ISD::SETNE; break;
    case ICmpInst::ICMP_UGT : Opcode = ISD::SETUGT; break;
    case ICmpInst::ICMP_UGE : Opcode = ISD::SETUGE; break;
    case ICmpInst::ICMP_ULT : Opcode = ISD::SETULT; break;
    case ICmpInst::ICMP_ULE : Opcode = ISD::SETULE; break;
    case ICmpInst::ICMP_SGT : Opcode = ISD::SETGT; break;
    case ICmpInst::ICMP_SGE : Opcode = ISD::SETGE; break;
    case ICmpInst::ICMP_SLT : Opcode = ISD::SETLT; break;
    case ICmpInst::ICMP_SLE : Opcode = ISD::SETLE; break;
    default:
      assert(!"Invalid ICmp predicate value");
      Opcode = ISD::SETEQ;
      break;
  }
  setValue(&I, DAG.getSetCC(MVT::i1, Op1, Op2, Opcode));
}

void SelectionDAGLowering::visitFCmp(User &I) {
  FCmpInst::Predicate predicate = FCmpInst::BAD_FCMP_PREDICATE;
  if (FCmpInst *FC = dyn_cast<FCmpInst>(&I))
    predicate = FC->getPredicate();
  else if (ConstantExpr *FC = dyn_cast<ConstantExpr>(&I))
    predicate = FCmpInst::Predicate(FC->getPredicate());
  SDOperand Op1 = getValue(I.getOperand(0));
  SDOperand Op2 = getValue(I.getOperand(1));
  ISD::CondCode Condition, FOC, FPC;
  switch (predicate) {
    case FCmpInst::FCMP_FALSE: FOC = FPC = ISD::SETFALSE; break;
    case FCmpInst::FCMP_OEQ:   FOC = ISD::SETEQ; FPC = ISD::SETOEQ; break;
    case FCmpInst::FCMP_OGT:   FOC = ISD::SETGT; FPC = ISD::SETOGT; break;
    case FCmpInst::FCMP_OGE:   FOC = ISD::SETGE; FPC = ISD::SETOGE; break;
    case FCmpInst::FCMP_OLT:   FOC = ISD::SETLT; FPC = ISD::SETOLT; break;
    case FCmpInst::FCMP_OLE:   FOC = ISD::SETLE; FPC = ISD::SETOLE; break;
    case FCmpInst::FCMP_ONE:   FOC = ISD::SETNE; FPC = ISD::SETONE; break;
    case FCmpInst::FCMP_ORD:   FOC = ISD::SETEQ; FPC = ISD::SETO;   break;
    case FCmpInst::FCMP_UNO:   FOC = ISD::SETNE; FPC = ISD::SETUO;  break;
    case FCmpInst::FCMP_UEQ:   FOC = ISD::SETEQ; FPC = ISD::SETUEQ; break;
    case FCmpInst::FCMP_UGT:   FOC = ISD::SETGT; FPC = ISD::SETUGT; break;
    case FCmpInst::FCMP_UGE:   FOC = ISD::SETGE; FPC = ISD::SETUGE; break;
    case FCmpInst::FCMP_ULT:   FOC = ISD::SETLT; FPC = ISD::SETULT; break;
    case FCmpInst::FCMP_ULE:   FOC = ISD::SETLE; FPC = ISD::SETULE; break;
    case FCmpInst::FCMP_UNE:   FOC = ISD::SETNE; FPC = ISD::SETUNE; break;
    case FCmpInst::FCMP_TRUE:  FOC = FPC = ISD::SETTRUE; break;
    default:
      assert(!"Invalid FCmp predicate value");
      FOC = FPC = ISD::SETFALSE;
      break;
  }
  if (FiniteOnlyFPMath())
    Condition = FOC;
  else 
    Condition = FPC;
  setValue(&I, DAG.getSetCC(MVT::i1, Op1, Op2, Condition));
}

void SelectionDAGLowering::visitSelect(User &I) {
  SDOperand Cond     = getValue(I.getOperand(0));
  SDOperand TrueVal  = getValue(I.getOperand(1));
  SDOperand FalseVal = getValue(I.getOperand(2));
  if (!isa<VectorType>(I.getType())) {
    setValue(&I, DAG.getNode(ISD::SELECT, TrueVal.getValueType(), Cond,
                             TrueVal, FalseVal));
  } else {
    setValue(&I, DAG.getNode(ISD::VSELECT, MVT::Vector, Cond, TrueVal, FalseVal,
                             *(TrueVal.Val->op_end()-2),
                             *(TrueVal.Val->op_end()-1)));
  }
}


void SelectionDAGLowering::visitTrunc(User &I) {
  // TruncInst cannot be a no-op cast because sizeof(src) > sizeof(dest).
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::TRUNCATE, DestVT, N));
}

void SelectionDAGLowering::visitZExt(User &I) {
  // ZExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // ZExt also can't be a cast to bool for same reason. So, nothing much to do
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitSExt(User &I) {
  // SExt cannot be a no-op cast because sizeof(src) < sizeof(dest).
  // SExt also can't be a cast to bool for same reason. So, nothing much to do
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::SIGN_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitFPTrunc(User &I) {
  // FPTrunc is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_ROUND, DestVT, N));
}

void SelectionDAGLowering::visitFPExt(User &I){ 
  // FPTrunc is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitFPToUI(User &I) { 
  // FPToUI is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_UINT, DestVT, N));
}

void SelectionDAGLowering::visitFPToSI(User &I) {
  // FPToSI is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::FP_TO_SINT, DestVT, N));
}

void SelectionDAGLowering::visitUIToFP(User &I) { 
  // UIToFP is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::UINT_TO_FP, DestVT, N));
}

void SelectionDAGLowering::visitSIToFP(User &I){ 
  // UIToFP is never a no-op cast, no need to check
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  setValue(&I, DAG.getNode(ISD::SINT_TO_FP, DestVT, N));
}

void SelectionDAGLowering::visitPtrToInt(User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType SrcVT = N.getValueType();
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  SDOperand Result;
  if (MVT::getSizeInBits(DestVT) < MVT::getSizeInBits(SrcVT))
    Result = DAG.getNode(ISD::TRUNCATE, DestVT, N);
  else 
    // Note: ZERO_EXTEND can handle cases where the sizes are equal too
    Result = DAG.getNode(ISD::ZERO_EXTEND, DestVT, N);
  setValue(&I, Result);
}

void SelectionDAGLowering::visitIntToPtr(User &I) {
  // What to do depends on the size of the integer and the size of the pointer.
  // We can either truncate, zero extend, or no-op, accordingly.
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType SrcVT = N.getValueType();
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  if (MVT::getSizeInBits(DestVT) < MVT::getSizeInBits(SrcVT))
    setValue(&I, DAG.getNode(ISD::TRUNCATE, DestVT, N));
  else 
    // Note: ZERO_EXTEND can handle cases where the sizes are equal too
    setValue(&I, DAG.getNode(ISD::ZERO_EXTEND, DestVT, N));
}

void SelectionDAGLowering::visitBitCast(User &I) { 
  SDOperand N = getValue(I.getOperand(0));
  MVT::ValueType DestVT = TLI.getValueType(I.getType());
  if (DestVT == MVT::Vector) {
    // This is a cast to a vector from something else.  
    // Get information about the output vector.
    const VectorType *DestTy = cast<VectorType>(I.getType());
    MVT::ValueType EltVT = TLI.getValueType(DestTy->getElementType());
    setValue(&I, DAG.getNode(ISD::VBIT_CONVERT, DestVT, N, 
                             DAG.getConstant(DestTy->getNumElements(),MVT::i32),
                             DAG.getValueType(EltVT)));
    return;
  } 
  MVT::ValueType SrcVT = N.getValueType();
  if (SrcVT == MVT::Vector) {
    // This is a cast from a vctor to something else. 
    // Get information about the input vector.
    setValue(&I, DAG.getNode(ISD::VBIT_CONVERT, DestVT, N));
    return;
  }

  // BitCast assures us that source and destination are the same size so this 
  // is either a BIT_CONVERT or a no-op.
  if (DestVT != N.getValueType())
    setValue(&I, DAG.getNode(ISD::BIT_CONVERT, DestVT, N)); // convert types
  else
    setValue(&I, N); // noop cast.
}

void SelectionDAGLowering::visitInsertElement(User &I) {
  SDOperand InVec = getValue(I.getOperand(0));
  SDOperand InVal = getValue(I.getOperand(1));
  SDOperand InIdx = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(),
                                getValue(I.getOperand(2)));

  SDOperand Num = *(InVec.Val->op_end()-2);
  SDOperand Typ = *(InVec.Val->op_end()-1);
  setValue(&I, DAG.getNode(ISD::VINSERT_VECTOR_ELT, MVT::Vector,
                           InVec, InVal, InIdx, Num, Typ));
}

void SelectionDAGLowering::visitExtractElement(User &I) {
  SDOperand InVec = getValue(I.getOperand(0));
  SDOperand InIdx = DAG.getNode(ISD::ZERO_EXTEND, TLI.getPointerTy(),
                                getValue(I.getOperand(1)));
  SDOperand Typ = *(InVec.Val->op_end()-1);
  setValue(&I, DAG.getNode(ISD::VEXTRACT_VECTOR_ELT,
                           TLI.getValueType(I.getType()), InVec, InIdx));
}

void SelectionDAGLowering::visitShuffleVector(User &I) {
  SDOperand V1   = getValue(I.getOperand(0));
  SDOperand V2   = getValue(I.getOperand(1));
  SDOperand Mask = getValue(I.getOperand(2));

  SDOperand Num = *(V1.Val->op_end()-2);
  SDOperand Typ = *(V2.Val->op_end()-1);
  setValue(&I, DAG.getNode(ISD::VVECTOR_SHUFFLE, MVT::Vector,
                           V1, V2, Mask, Num, Typ));
}


void SelectionDAGLowering::visitGetElementPtr(User &I) {
  SDOperand N = getValue(I.getOperand(0));
  const Type *Ty = I.getOperand(0)->getType();

  for (GetElementPtrInst::op_iterator OI = I.op_begin()+1, E = I.op_end();
       OI != E; ++OI) {
    Value *Idx = *OI;
    if (const StructType *StTy = dyn_cast<StructType>(Ty)) {
      unsigned Field = cast<ConstantInt>(Idx)->getZExtValue();
      if (Field) {
        // N = N + Offset
        uint64_t Offset = TD->getStructLayout(StTy)->getElementOffset(Field);
        N = DAG.getNode(ISD::ADD, N.getValueType(), N,
                        getIntPtrConstant(Offset));
      }
      Ty = StTy->getElementType(Field);
    } else {
      Ty = cast<SequentialType>(Ty)->getElementType();

      // If this is a constant subscript, handle it quickly.
      if (ConstantInt *CI = dyn_cast<ConstantInt>(Idx)) {
        if (CI->getZExtValue() == 0) continue;
        uint64_t Offs = 
            TD->getTypeSize(Ty)*cast<ConstantInt>(CI)->getSExtValue();
        N = DAG.getNode(ISD::ADD, N.getValueType(), N, getIntPtrConstant(Offs));
        continue;
      }
      
      // N = N + Idx * ElementSize;
      uint64_t ElementSize = TD->getTypeSize(Ty);
      SDOperand IdxN = getValue(Idx);

      // If the index is smaller or larger than intptr_t, truncate or extend
      // it.
      if (IdxN.getValueType() < N.getValueType()) {
        IdxN = DAG.getNode(ISD::SIGN_EXTEND, N.getValueType(), IdxN);
      } else if (IdxN.getValueType() > N.getValueType())
        IdxN = DAG.getNode(ISD::TRUNCATE, N.getValueType(), IdxN);

      // If this is a multiply by a power of two, turn it into a shl
      // immediately.  This is a very common case.
      if (isPowerOf2_64(ElementSize)) {
        unsigned Amt = Log2_64(ElementSize);
        IdxN = DAG.getNode(ISD::SHL, N.getValueType(), IdxN,
                           DAG.getConstant(Amt, TLI.getShiftAmountTy()));
        N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
        continue;
      }
      
      SDOperand Scale = getIntPtrConstant(ElementSize);
      IdxN = DAG.getNode(ISD::MUL, N.getValueType(), IdxN, Scale);
      N = DAG.getNode(ISD::ADD, N.getValueType(), N, IdxN);
    }
  }
  setValue(&I, N);
}

void SelectionDAGLowering::visitAlloca(AllocaInst &I) {
  // If this is a fixed sized alloca in the entry block of the function,
  // allocate it statically on the stack.
  if (FuncInfo.StaticAllocaMap.count(&I))
    return;   // getValue will auto-populate this.

  const Type *Ty = I.getAllocatedType();
  uint64_t TySize = TLI.getTargetData()->getTypeSize(Ty);
  unsigned Align =
    std::max((unsigned)TLI.getTargetData()->getPrefTypeAlignment(Ty),
             I.getAlignment());

  SDOperand AllocSize = getValue(I.getArraySize());
  MVT::ValueType IntPtr = TLI.getPointerTy();
  if (IntPtr < AllocSize.getValueType())
    AllocSize = DAG.getNode(ISD::TRUNCATE, IntPtr, AllocSize);
  else if (IntPtr > AllocSize.getValueType())
    AllocSize = DAG.getNode(ISD::ZERO_EXTEND, IntPtr, AllocSize);

  AllocSize = DAG.getNode(ISD::MUL, IntPtr, AllocSize,
                          getIntPtrConstant(TySize));

  // Handle alignment.  If the requested alignment is less than or equal to the
  // stack alignment, ignore it and round the size of the allocation up to the
  // stack alignment size.  If the size is greater than the stack alignment, we
  // note this in the DYNAMIC_STACKALLOC node.
  unsigned StackAlign =
    TLI.getTargetMachine().getFrameInfo()->getStackAlignment();
  if (Align <= StackAlign) {
    Align = 0;
    // Add SA-1 to the size.
    AllocSize = DAG.getNode(ISD::ADD, AllocSize.getValueType(), AllocSize,
                            getIntPtrConstant(StackAlign-1));
    // Mask out the low bits for alignment purposes.
    AllocSize = DAG.getNode(ISD::AND, AllocSize.getValueType(), AllocSize,
                            getIntPtrConstant(~(uint64_t)(StackAlign-1)));
  }

  SDOperand Ops[] = { getRoot(), AllocSize, getIntPtrConstant(Align) };
  const MVT::ValueType *VTs = DAG.getNodeValueTypes(AllocSize.getValueType(),
                                                    MVT::Other);
  SDOperand DSA = DAG.getNode(ISD::DYNAMIC_STACKALLOC, VTs, 2, Ops, 3);
  setValue(&I, DSA);
  DAG.setRoot(DSA.getValue(1));

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  CurMBB->getParent()->getFrameInfo()->CreateVariableSizedObject();
}

void SelectionDAGLowering::visitLoad(LoadInst &I) {
  SDOperand Ptr = getValue(I.getOperand(0));

  SDOperand Root;
  if (I.isVolatile())
    Root = getRoot();
  else {
    // Do not serialize non-volatile loads against each other.
    Root = DAG.getRoot();
  }

  setValue(&I, getLoadFrom(I.getType(), Ptr, I.getOperand(0),
                           Root, I.isVolatile(), I.getAlignment()));
}

SDOperand SelectionDAGLowering::getLoadFrom(const Type *Ty, SDOperand Ptr,
                                            const Value *SV, SDOperand Root,
                                            bool isVolatile, 
                                            unsigned Alignment) {
  SDOperand L;
  if (const VectorType *PTy = dyn_cast<VectorType>(Ty)) {
    MVT::ValueType PVT = TLI.getValueType(PTy->getElementType());
    L = DAG.getVecLoad(PTy->getNumElements(), PVT, Root, Ptr,
                       DAG.getSrcValue(SV));
  } else {
    L = DAG.getLoad(TLI.getValueType(Ty), Root, Ptr, SV, 0, 
                    isVolatile, Alignment);
  }

  if (isVolatile)
    DAG.setRoot(L.getValue(1));
  else
    PendingLoads.push_back(L.getValue(1));
  
  return L;
}


void SelectionDAGLowering::visitStore(StoreInst &I) {
  Value *SrcV = I.getOperand(0);
  SDOperand Src = getValue(SrcV);
  SDOperand Ptr = getValue(I.getOperand(1));
  DAG.setRoot(DAG.getStore(getRoot(), Src, Ptr, I.getOperand(1), 0,
                           I.isVolatile(), I.getAlignment()));
}

/// IntrinsicCannotAccessMemory - Return true if the specified intrinsic cannot
/// access memory and has no other side effects at all.
static bool IntrinsicCannotAccessMemory(unsigned IntrinsicID) {
#define GET_NO_MEMORY_INTRINSICS
#include "llvm/Intrinsics.gen"
#undef GET_NO_MEMORY_INTRINSICS
  return false;
}

// IntrinsicOnlyReadsMemory - Return true if the specified intrinsic doesn't
// have any side-effects or if it only reads memory.
static bool IntrinsicOnlyReadsMemory(unsigned IntrinsicID) {
#define GET_SIDE_EFFECT_INFO
#include "llvm/Intrinsics.gen"
#undef GET_SIDE_EFFECT_INFO
  return false;
}

/// visitTargetIntrinsic - Lower a call of a target intrinsic to an INTRINSIC
/// node.
void SelectionDAGLowering::visitTargetIntrinsic(CallInst &I, 
                                                unsigned Intrinsic) {
  bool HasChain = !IntrinsicCannotAccessMemory(Intrinsic);
  bool OnlyLoad = HasChain && IntrinsicOnlyReadsMemory(Intrinsic);
  
  // Build the operand list.
  SmallVector<SDOperand, 8> Ops;
  if (HasChain) {  // If this intrinsic has side-effects, chainify it.
    if (OnlyLoad) {
      // We don't need to serialize loads against other loads.
      Ops.push_back(DAG.getRoot());
    } else { 
      Ops.push_back(getRoot());
    }
  }
  
  // Add the intrinsic ID as an integer operand.
  Ops.push_back(DAG.getConstant(Intrinsic, TLI.getPointerTy()));

  // Add all operands of the call to the operand list.
  for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i) {
    SDOperand Op = getValue(I.getOperand(i));
    
    // If this is a vector type, force it to the right vector type.
    if (Op.getValueType() == MVT::Vector) {
      const VectorType *OpTy = cast<VectorType>(I.getOperand(i)->getType());
      MVT::ValueType EltVT = TLI.getValueType(OpTy->getElementType());
      
      MVT::ValueType VVT = MVT::getVectorType(EltVT, OpTy->getNumElements());
      assert(VVT != MVT::Other && "Intrinsic uses a non-legal type?");
      Op = DAG.getNode(ISD::VBIT_CONVERT, VVT, Op);
    }
    
    assert(TLI.isTypeLegal(Op.getValueType()) &&
           "Intrinsic uses a non-legal type?");
    Ops.push_back(Op);
  }

  std::vector<MVT::ValueType> VTs;
  if (I.getType() != Type::VoidTy) {
    MVT::ValueType VT = TLI.getValueType(I.getType());
    if (VT == MVT::Vector) {
      const VectorType *DestTy = cast<VectorType>(I.getType());
      MVT::ValueType EltVT = TLI.getValueType(DestTy->getElementType());
      
      VT = MVT::getVectorType(EltVT, DestTy->getNumElements());
      assert(VT != MVT::Other && "Intrinsic uses a non-legal type?");
    }
    
    assert(TLI.isTypeLegal(VT) && "Intrinsic uses a non-legal type?");
    VTs.push_back(VT);
  }
  if (HasChain)
    VTs.push_back(MVT::Other);

  const MVT::ValueType *VTList = DAG.getNodeValueTypes(VTs);

  // Create the node.
  SDOperand Result;
  if (!HasChain)
    Result = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, VTList, VTs.size(),
                         &Ops[0], Ops.size());
  else if (I.getType() != Type::VoidTy)
    Result = DAG.getNode(ISD::INTRINSIC_W_CHAIN, VTList, VTs.size(),
                         &Ops[0], Ops.size());
  else
    Result = DAG.getNode(ISD::INTRINSIC_VOID, VTList, VTs.size(),
                         &Ops[0], Ops.size());

  if (HasChain) {
    SDOperand Chain = Result.getValue(Result.Val->getNumValues()-1);
    if (OnlyLoad)
      PendingLoads.push_back(Chain);
    else
      DAG.setRoot(Chain);
  }
  if (I.getType() != Type::VoidTy) {
    if (const VectorType *PTy = dyn_cast<VectorType>(I.getType())) {
      MVT::ValueType EVT = TLI.getValueType(PTy->getElementType());
      Result = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, Result,
                           DAG.getConstant(PTy->getNumElements(), MVT::i32),
                           DAG.getValueType(EVT));
    } 
    setValue(&I, Result);
  }
}

/// ExtractGlobalVariable - If C is a global variable, or a bitcast of one
/// (possibly constant folded), return it.  Otherwise return NULL.
static GlobalVariable *ExtractGlobalVariable (Constant *C) {
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(C))
    return GV;
  else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    if (CE->getOpcode() == Instruction::BitCast)
      return dyn_cast<GlobalVariable>(CE->getOperand(0));
    else if (CE->getOpcode() == Instruction::GetElementPtr) {
      for (unsigned i = 1, e = CE->getNumOperands(); i != e; ++i)
        if (!CE->getOperand(i)->isNullValue())
          return NULL;
      return dyn_cast<GlobalVariable>(CE->getOperand(0));
    }
  }
  return NULL;
}

/// visitIntrinsicCall - Lower the call to the specified intrinsic function.  If
/// we want to emit this as a call to a named external function, return the name
/// otherwise lower it and return null.
const char *
SelectionDAGLowering::visitIntrinsicCall(CallInst &I, unsigned Intrinsic) {
  switch (Intrinsic) {
  default:
    // By default, turn this into a target intrinsic node.
    visitTargetIntrinsic(I, Intrinsic);
    return 0;
  case Intrinsic::vastart:  visitVAStart(I); return 0;
  case Intrinsic::vaend:    visitVAEnd(I); return 0;
  case Intrinsic::vacopy:   visitVACopy(I); return 0;
  case Intrinsic::returnaddress:
    setValue(&I, DAG.getNode(ISD::RETURNADDR, TLI.getPointerTy(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::frameaddress:
    setValue(&I, DAG.getNode(ISD::FRAMEADDR, TLI.getPointerTy(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::setjmp:
    return "_setjmp"+!TLI.usesUnderscoreSetJmp();
    break;
  case Intrinsic::longjmp:
    return "_longjmp"+!TLI.usesUnderscoreLongJmp();
    break;
  case Intrinsic::memcpy_i32:
  case Intrinsic::memcpy_i64:
    visitMemIntrinsic(I, ISD::MEMCPY);
    return 0;
  case Intrinsic::memset_i32:
  case Intrinsic::memset_i64:
    visitMemIntrinsic(I, ISD::MEMSET);
    return 0;
  case Intrinsic::memmove_i32:
  case Intrinsic::memmove_i64:
    visitMemIntrinsic(I, ISD::MEMMOVE);
    return 0;
    
  case Intrinsic::dbg_stoppoint: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgStopPointInst &SPI = cast<DbgStopPointInst>(I);
    if (MMI && SPI.getContext() && MMI->Verify(SPI.getContext())) {
      SDOperand Ops[5];

      Ops[0] = getRoot();
      Ops[1] = getValue(SPI.getLineValue());
      Ops[2] = getValue(SPI.getColumnValue());

      DebugInfoDesc *DD = MMI->getDescFor(SPI.getContext());
      assert(DD && "Not a debug information descriptor");
      CompileUnitDesc *CompileUnit = cast<CompileUnitDesc>(DD);
      
      Ops[3] = DAG.getString(CompileUnit->getFileName());
      Ops[4] = DAG.getString(CompileUnit->getDirectory());
      
      DAG.setRoot(DAG.getNode(ISD::LOCATION, MVT::Other, Ops, 5));
    }

    return 0;
  }
  case Intrinsic::dbg_region_start: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgRegionStartInst &RSI = cast<DbgRegionStartInst>(I);
    if (MMI && RSI.getContext() && MMI->Verify(RSI.getContext())) {
      unsigned LabelID = MMI->RecordRegionStart(RSI.getContext());
      DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, getRoot(),
                              DAG.getConstant(LabelID, MVT::i32)));
    }

    return 0;
  }
  case Intrinsic::dbg_region_end: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgRegionEndInst &REI = cast<DbgRegionEndInst>(I);
    if (MMI && REI.getContext() && MMI->Verify(REI.getContext())) {
      unsigned LabelID = MMI->RecordRegionEnd(REI.getContext());
      DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other,
                              getRoot(), DAG.getConstant(LabelID, MVT::i32)));
    }

    return 0;
  }
  case Intrinsic::dbg_func_start: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgFuncStartInst &FSI = cast<DbgFuncStartInst>(I);
    if (MMI && FSI.getSubprogram() &&
        MMI->Verify(FSI.getSubprogram())) {
      unsigned LabelID = MMI->RecordRegionStart(FSI.getSubprogram());
      DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other,
                  getRoot(), DAG.getConstant(LabelID, MVT::i32)));
    }

    return 0;
  }
  case Intrinsic::dbg_declare: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    DbgDeclareInst &DI = cast<DbgDeclareInst>(I);
    if (MMI && DI.getVariable() && MMI->Verify(DI.getVariable())) {
      SDOperand AddressOp  = getValue(DI.getAddress());
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(AddressOp))
        MMI->RecordVariable(DI.getVariable(), FI->getIndex());
    }

    return 0;
  }
    
  case Intrinsic::eh_exception: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    
    if (MMI) {
      // Add a label to mark the beginning of the landing pad.  Deletion of the
      // landing pad can thus be detected via the MachineModuleInfo.
      unsigned LabelID = MMI->addLandingPad(CurMBB);
      DAG.setRoot(DAG.getNode(ISD::LABEL, MVT::Other, DAG.getEntryNode(),
                              DAG.getConstant(LabelID, MVT::i32)));
      
      // Mark exception register as live in.
      unsigned Reg = TLI.getExceptionAddressRegister();
      if (Reg) CurMBB->addLiveIn(Reg);
      
      // Insert the EXCEPTIONADDR instruction.
      SDVTList VTs = DAG.getVTList(TLI.getPointerTy(), MVT::Other);
      SDOperand Ops[1];
      Ops[0] = DAG.getRoot();
      SDOperand Op = DAG.getNode(ISD::EXCEPTIONADDR, VTs, Ops, 1);
      setValue(&I, Op);
      DAG.setRoot(Op.getValue(1));
    } else {
      setValue(&I, DAG.getConstant(0, TLI.getPointerTy()));
    }
    return 0;
  }

  case Intrinsic::eh_selector:
  case Intrinsic::eh_filter:{
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    
    if (MMI) {
      // Inform the MachineModuleInfo of the personality for this landing pad.
      ConstantExpr *CE = dyn_cast<ConstantExpr>(I.getOperand(2));
      assert(CE && CE->getOpcode() == Instruction::BitCast &&
             isa<Function>(CE->getOperand(0)) &&
             "Personality should be a function");
      MMI->addPersonality(CurMBB, cast<Function>(CE->getOperand(0)));
      if (Intrinsic == Intrinsic::eh_filter)
        MMI->setIsFilterLandingPad(CurMBB);

      // Gather all the type infos for this landing pad and pass them along to
      // MachineModuleInfo.
      std::vector<GlobalVariable *> TyInfo;
      for (unsigned i = 3, N = I.getNumOperands(); i < N; ++i) {
        Constant *C = cast<Constant>(I.getOperand(i));
        GlobalVariable *GV = ExtractGlobalVariable(C);
        assert (GV || (isa<ConstantInt>(C) &&
                       cast<ConstantInt>(C)->isNullValue()) &&
                "TypeInfo must be a global variable or NULL");
        TyInfo.push_back(GV);
      }
      MMI->addCatchTypeInfo(CurMBB, TyInfo);
      
      // Mark exception selector register as live in.
      unsigned Reg = TLI.getExceptionSelectorRegister();
      if (Reg) CurMBB->addLiveIn(Reg);

      // Insert the EHSELECTION instruction.
      SDVTList VTs = DAG.getVTList(TLI.getPointerTy(), MVT::Other);
      SDOperand Ops[2];
      Ops[0] = getValue(I.getOperand(1));
      Ops[1] = getRoot();
      SDOperand Op = DAG.getNode(ISD::EHSELECTION, VTs, Ops, 2);
      setValue(&I, Op);
      DAG.setRoot(Op.getValue(1));
    } else {
      setValue(&I, DAG.getConstant(0, TLI.getPointerTy()));
    }
    
    return 0;
  }
  
  case Intrinsic::eh_typeid_for: {
    MachineModuleInfo *MMI = DAG.getMachineModuleInfo();
    
    if (MMI) {
      // Find the type id for the given typeinfo.
      Constant *C = cast<Constant>(I.getOperand(1));
      GlobalVariable *GV = ExtractGlobalVariable(C);
      assert (GV || (isa<ConstantInt>(C) &&
                     cast<ConstantInt>(C)->isNullValue()) &&
              "TypeInfo must be a global variable or NULL");

      unsigned TypeID = MMI->getTypeIDFor(GV);
      setValue(&I, DAG.getConstant(TypeID, MVT::i32));
    } else {
      setValue(&I, DAG.getConstant(0, MVT::i32));
    }

    return 0;
  }

  case Intrinsic::sqrt_f32:
  case Intrinsic::sqrt_f64:
    setValue(&I, DAG.getNode(ISD::FSQRT,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::powi_f32:
  case Intrinsic::powi_f64:
    setValue(&I, DAG.getNode(ISD::FPOWI,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1)),
                             getValue(I.getOperand(2))));
    return 0;
  case Intrinsic::pcmarker: {
    SDOperand Tmp = getValue(I.getOperand(1));
    DAG.setRoot(DAG.getNode(ISD::PCMARKER, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::readcyclecounter: {
    SDOperand Op = getRoot();
    SDOperand Tmp = DAG.getNode(ISD::READCYCLECOUNTER,
                                DAG.getNodeValueTypes(MVT::i64, MVT::Other), 2,
                                &Op, 1);
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::part_select: {
    // Currently not implemented: just abort
    assert(0 && "part_select intrinsic not implemented");
    abort();
  }
  case Intrinsic::part_set: {
    // Currently not implemented: just abort
    assert(0 && "part_set intrinsic not implemented");
    abort();
  }
  case Intrinsic::bswap:
    setValue(&I, DAG.getNode(ISD::BSWAP,
                             getValue(I.getOperand(1)).getValueType(),
                             getValue(I.getOperand(1))));
    return 0;
  case Intrinsic::cttz: {
    SDOperand Arg = getValue(I.getOperand(1));
    MVT::ValueType Ty = Arg.getValueType();
    SDOperand result = DAG.getNode(ISD::CTTZ, Ty, Arg);
    if (Ty < MVT::i32)
      result = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, result);
    else if (Ty > MVT::i32)
      result = DAG.getNode(ISD::TRUNCATE, MVT::i32, result);
    setValue(&I, result);
    return 0;
  }
  case Intrinsic::ctlz: {
    SDOperand Arg = getValue(I.getOperand(1));
    MVT::ValueType Ty = Arg.getValueType();
    SDOperand result = DAG.getNode(ISD::CTLZ, Ty, Arg);
    if (Ty < MVT::i32)
      result = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, result);
    else if (Ty > MVT::i32)
      result = DAG.getNode(ISD::TRUNCATE, MVT::i32, result);
    setValue(&I, result);
    return 0;
  }
  case Intrinsic::ctpop: {
    SDOperand Arg = getValue(I.getOperand(1));
    MVT::ValueType Ty = Arg.getValueType();
    SDOperand result = DAG.getNode(ISD::CTPOP, Ty, Arg);
    if (Ty < MVT::i32)
      result = DAG.getNode(ISD::ZERO_EXTEND, MVT::i32, result);
    else if (Ty > MVT::i32)
      result = DAG.getNode(ISD::TRUNCATE, MVT::i32, result);
    setValue(&I, result);
    return 0;
  }
  case Intrinsic::stacksave: {
    SDOperand Op = getRoot();
    SDOperand Tmp = DAG.getNode(ISD::STACKSAVE,
              DAG.getNodeValueTypes(TLI.getPointerTy(), MVT::Other), 2, &Op, 1);
    setValue(&I, Tmp);
    DAG.setRoot(Tmp.getValue(1));
    return 0;
  }
  case Intrinsic::stackrestore: {
    SDOperand Tmp = getValue(I.getOperand(1));
    DAG.setRoot(DAG.getNode(ISD::STACKRESTORE, MVT::Other, getRoot(), Tmp));
    return 0;
  }
  case Intrinsic::prefetch:
    // FIXME: Currently discarding prefetches.
    return 0;
  }
}


void SelectionDAGLowering::LowerCallTo(Instruction &I,
                                       const Type *CalledValueTy,
                                       unsigned CallingConv,
                                       bool IsTailCall,
                                       SDOperand Callee, unsigned OpIdx) {
  const PointerType *PT = cast<PointerType>(CalledValueTy);
  const FunctionType *FTy = cast<FunctionType>(PT->getElementType());
  const ParamAttrsList *Attrs = FTy->getParamAttrs();

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Args.reserve(I.getNumOperands());
  for (unsigned i = OpIdx, e = I.getNumOperands(); i != e; ++i) {
    Value *Arg = I.getOperand(i);
    SDOperand ArgNode = getValue(Arg);
    Entry.Node = ArgNode; Entry.Ty = Arg->getType();

    unsigned attrInd = i - OpIdx + 1;
    Entry.isSExt  = Attrs && Attrs->paramHasAttr(attrInd, ParamAttr::SExt);
    Entry.isZExt  = Attrs && Attrs->paramHasAttr(attrInd, ParamAttr::ZExt);
    Entry.isInReg = Attrs && Attrs->paramHasAttr(attrInd, ParamAttr::InReg);
    Entry.isSRet  = Attrs && Attrs->paramHasAttr(attrInd, ParamAttr::StructRet);
    Args.push_back(Entry);
  }

  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), I.getType(), 
                    Attrs && Attrs->paramHasAttr(0, ParamAttr::SExt),
                    FTy->isVarArg(), CallingConv, IsTailCall, 
                    Callee, Args, DAG);
  if (I.getType() != Type::VoidTy)
    setValue(&I, Result.first);
  DAG.setRoot(Result.second);
}


void SelectionDAGLowering::visitCall(CallInst &I) {
  const char *RenameFn = 0;
  if (Function *F = I.getCalledFunction()) {
    if (F->isDeclaration())
      if (unsigned IID = F->getIntrinsicID()) {
        RenameFn = visitIntrinsicCall(I, IID);
        if (!RenameFn)
          return;
      } else {    // Not an LLVM intrinsic.
        const std::string &Name = F->getName();
        if (Name[0] == 'c' && (Name == "copysign" || Name == "copysignf")) {
          if (I.getNumOperands() == 3 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType() &&
              I.getType() == I.getOperand(2)->getType()) {
            SDOperand LHS = getValue(I.getOperand(1));
            SDOperand RHS = getValue(I.getOperand(2));
            setValue(&I, DAG.getNode(ISD::FCOPYSIGN, LHS.getValueType(),
                                     LHS, RHS));
            return;
          }
        } else if (Name[0] == 'f' && (Name == "fabs" || Name == "fabsf")) {
          if (I.getNumOperands() == 2 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType()) {
            SDOperand Tmp = getValue(I.getOperand(1));
            setValue(&I, DAG.getNode(ISD::FABS, Tmp.getValueType(), Tmp));
            return;
          }
        } else if (Name[0] == 's' && (Name == "sin" || Name == "sinf")) {
          if (I.getNumOperands() == 2 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType()) {
            SDOperand Tmp = getValue(I.getOperand(1));
            setValue(&I, DAG.getNode(ISD::FSIN, Tmp.getValueType(), Tmp));
            return;
          }
        } else if (Name[0] == 'c' && (Name == "cos" || Name == "cosf")) {
          if (I.getNumOperands() == 2 &&   // Basic sanity checks.
              I.getOperand(1)->getType()->isFloatingPoint() &&
              I.getType() == I.getOperand(1)->getType()) {
            SDOperand Tmp = getValue(I.getOperand(1));
            setValue(&I, DAG.getNode(ISD::FCOS, Tmp.getValueType(), Tmp));
            return;
          }
        }
      }
  } else if (isa<InlineAsm>(I.getOperand(0))) {
    visitInlineAsm(I);
    return;
  }

  SDOperand Callee;
  if (!RenameFn)
    Callee = getValue(I.getOperand(0));
  else
    Callee = DAG.getExternalSymbol(RenameFn, TLI.getPointerTy());
    
  LowerCallTo(I, I.getCalledValue()->getType(),
                 I.getCallingConv(),
                 I.isTailCall(),
                 Callee,
                 1);
}


SDOperand RegsForValue::getCopyFromRegs(SelectionDAG &DAG,
                                        SDOperand &Chain, SDOperand &Flag)const{
  SDOperand Val = DAG.getCopyFromReg(Chain, Regs[0], RegVT, Flag);
  Chain = Val.getValue(1);
  Flag  = Val.getValue(2);
  
  // If the result was expanded, copy from the top part.
  if (Regs.size() > 1) {
    assert(Regs.size() == 2 &&
           "Cannot expand to more than 2 elts yet!");
    SDOperand Hi = DAG.getCopyFromReg(Chain, Regs[1], RegVT, Flag);
    Chain = Hi.getValue(1);
    Flag  = Hi.getValue(2);
    if (DAG.getTargetLoweringInfo().isLittleEndian())
      return DAG.getNode(ISD::BUILD_PAIR, ValueVT, Val, Hi);
    else
      return DAG.getNode(ISD::BUILD_PAIR, ValueVT, Hi, Val);
  }

  // Otherwise, if the return value was promoted or extended, truncate it to the
  // appropriate type.
  if (RegVT == ValueVT)
    return Val;
  
  if (MVT::isVector(RegVT)) {
    assert(ValueVT == MVT::Vector && "Unknown vector conversion!");
    return DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, Val, 
                       DAG.getConstant(MVT::getVectorNumElements(RegVT),
                                       MVT::i32),
                       DAG.getValueType(MVT::getVectorBaseType(RegVT)));
  }
  
  if (MVT::isInteger(RegVT)) {
    if (ValueVT < RegVT)
      return DAG.getNode(ISD::TRUNCATE, ValueVT, Val);
    else
      return DAG.getNode(ISD::ANY_EXTEND, ValueVT, Val);
  }
  
  assert(MVT::isFloatingPoint(RegVT) && MVT::isFloatingPoint(ValueVT));
  return DAG.getNode(ISD::FP_ROUND, ValueVT, Val);
}

/// getCopyToRegs - Emit a series of CopyToReg nodes that copies the
/// specified value into the registers specified by this object.  This uses 
/// Chain/Flag as the input and updates them for the output Chain/Flag.
void RegsForValue::getCopyToRegs(SDOperand Val, SelectionDAG &DAG,
                                 SDOperand &Chain, SDOperand &Flag,
                                 MVT::ValueType PtrVT) const {
  if (Regs.size() == 1) {
    // If there is a single register and the types differ, this must be
    // a promotion.
    if (RegVT != ValueVT) {
      if (MVT::isVector(RegVT)) {
        assert(Val.getValueType() == MVT::Vector &&"Not a vector-vector cast?");
        Val = DAG.getNode(ISD::VBIT_CONVERT, RegVT, Val);
      } else if (MVT::isInteger(RegVT) && MVT::isInteger(Val.getValueType())) {
        if (RegVT < ValueVT)
          Val = DAG.getNode(ISD::TRUNCATE, RegVT, Val);
        else
          Val = DAG.getNode(ISD::ANY_EXTEND, RegVT, Val);
      } else if (MVT::isFloatingPoint(RegVT) &&
                 MVT::isFloatingPoint(Val.getValueType())) {
        Val = DAG.getNode(ISD::FP_EXTEND, RegVT, Val);
      } else if (MVT::getSizeInBits(RegVT) == 
                 MVT::getSizeInBits(Val.getValueType())) {
        Val = DAG.getNode(ISD::BIT_CONVERT, RegVT, Val);
      } else {
        assert(0 && "Unknown mismatch!");
      }
    }
    Chain = DAG.getCopyToReg(Chain, Regs[0], Val, Flag);
    Flag = Chain.getValue(1);
  } else {
    std::vector<unsigned> R(Regs);
    if (!DAG.getTargetLoweringInfo().isLittleEndian())
      std::reverse(R.begin(), R.end());
    
    for (unsigned i = 0, e = R.size(); i != e; ++i) {
      SDOperand Part = DAG.getNode(ISD::EXTRACT_ELEMENT, RegVT, Val, 
                                   DAG.getConstant(i, PtrVT));
      Chain = DAG.getCopyToReg(Chain, R[i], Part, Flag);
      Flag = Chain.getValue(1);
    }
  }
}

/// AddInlineAsmOperands - Add this value to the specified inlineasm node
/// operand list.  This adds the code marker and includes the number of 
/// values added into it.
void RegsForValue::AddInlineAsmOperands(unsigned Code, SelectionDAG &DAG,
                                        std::vector<SDOperand> &Ops) const {
  MVT::ValueType IntPtrTy = DAG.getTargetLoweringInfo().getPointerTy();
  Ops.push_back(DAG.getTargetConstant(Code | (Regs.size() << 3), IntPtrTy));
  for (unsigned i = 0, e = Regs.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(Regs[i], RegVT));
}

/// isAllocatableRegister - If the specified register is safe to allocate, 
/// i.e. it isn't a stack pointer or some other special register, return the
/// register class for the register.  Otherwise, return null.
static const TargetRegisterClass *
isAllocatableRegister(unsigned Reg, MachineFunction &MF,
                      const TargetLowering &TLI, const MRegisterInfo *MRI) {
  MVT::ValueType FoundVT = MVT::Other;
  const TargetRegisterClass *FoundRC = 0;
  for (MRegisterInfo::regclass_iterator RCI = MRI->regclass_begin(),
       E = MRI->regclass_end(); RCI != E; ++RCI) {
    MVT::ValueType ThisVT = MVT::Other;

    const TargetRegisterClass *RC = *RCI;
    // If none of the the value types for this register class are valid, we 
    // can't use it.  For example, 64-bit reg classes on 32-bit targets.
    for (TargetRegisterClass::vt_iterator I = RC->vt_begin(), E = RC->vt_end();
         I != E; ++I) {
      if (TLI.isTypeLegal(*I)) {
        // If we have already found this register in a different register class,
        // choose the one with the largest VT specified.  For example, on
        // PowerPC, we favor f64 register classes over f32.
        if (FoundVT == MVT::Other || 
            MVT::getSizeInBits(FoundVT) < MVT::getSizeInBits(*I)) {
          ThisVT = *I;
          break;
        }
      }
    }
    
    if (ThisVT == MVT::Other) continue;
    
    // NOTE: This isn't ideal.  In particular, this might allocate the
    // frame pointer in functions that need it (due to them not being taken
    // out of allocation, because a variable sized allocation hasn't been seen
    // yet).  This is a slight code pessimization, but should still work.
    for (TargetRegisterClass::iterator I = RC->allocation_order_begin(MF),
         E = RC->allocation_order_end(MF); I != E; ++I)
      if (*I == Reg) {
        // We found a matching register class.  Keep looking at others in case
        // we find one with larger registers that this physreg is also in.
        FoundRC = RC;
        FoundVT = ThisVT;
        break;
      }
  }
  return FoundRC;
}    


namespace {
/// AsmOperandInfo - This contains information for each constraint that we are
/// lowering.
struct AsmOperandInfo : public InlineAsm::ConstraintInfo {
  /// ConstraintCode - This contains the actual string for the code, like "m".
  std::string ConstraintCode;

  /// ConstraintType - Information about the constraint code, e.g. Register,
  /// RegisterClass, Memory, Other, Unknown.
  TargetLowering::ConstraintType ConstraintType;
  
  /// CallOperand/CallOperandval - If this is the result output operand or a
  /// clobber, this is null, otherwise it is the incoming operand to the
  /// CallInst.  This gets modified as the asm is processed.
  SDOperand CallOperand;
  Value *CallOperandVal;
  
  /// ConstraintVT - The ValueType for the operand value.
  MVT::ValueType ConstraintVT;
  
  /// AssignedRegs - If this is a register or register class operand, this
  /// contains the set of register corresponding to the operand.
  RegsForValue AssignedRegs;
  
  AsmOperandInfo(const InlineAsm::ConstraintInfo &info)
    : InlineAsm::ConstraintInfo(info), 
      ConstraintType(TargetLowering::C_Unknown),
      CallOperand(0,0), CallOperandVal(0), ConstraintVT(MVT::Other) {
  }
  
  void ComputeConstraintToUse(const TargetLowering &TLI);
  
  /// MarkAllocatedRegs - Once AssignedRegs is set, mark the assigned registers
  /// busy in OutputRegs/InputRegs.
  void MarkAllocatedRegs(bool isOutReg, bool isInReg,
                         std::set<unsigned> &OutputRegs, 
                         std::set<unsigned> &InputRegs) const {
     if (isOutReg)
       OutputRegs.insert(AssignedRegs.Regs.begin(), AssignedRegs.Regs.end());
     if (isInReg)
       InputRegs.insert(AssignedRegs.Regs.begin(), AssignedRegs.Regs.end());
   }
};
} // end anon namespace.

/// getConstraintGenerality - Return an integer indicating how general CT is.
static unsigned getConstraintGenerality(TargetLowering::ConstraintType CT) {
  switch (CT) {
    default: assert(0 && "Unknown constraint type!");
    case TargetLowering::C_Other:
    case TargetLowering::C_Unknown:
      return 0;
    case TargetLowering::C_Register:
      return 1;
    case TargetLowering::C_RegisterClass:
      return 2;
    case TargetLowering::C_Memory:
      return 3;
  }
}

void AsmOperandInfo::ComputeConstraintToUse(const TargetLowering &TLI) {
  assert(!Codes.empty() && "Must have at least one constraint");
  
  std::string *Current = &Codes[0];
  TargetLowering::ConstraintType CurType = TLI.getConstraintType(*Current);
  if (Codes.size() == 1) {   // Single-letter constraints ('r') are very common.
    ConstraintCode = *Current;
    ConstraintType = CurType;
    return;
  }
  
  unsigned CurGenerality = getConstraintGenerality(CurType);
  
  // If we have multiple constraints, try to pick the most general one ahead
  // of time.  This isn't a wonderful solution, but handles common cases.
  for (unsigned j = 1, e = Codes.size(); j != e; ++j) {
    TargetLowering::ConstraintType ThisType = TLI.getConstraintType(Codes[j]);
    unsigned ThisGenerality = getConstraintGenerality(ThisType);
    if (ThisGenerality > CurGenerality) {
      // This constraint letter is more general than the previous one,
      // use it.
      CurType = ThisType;
      Current = &Codes[j];
      CurGenerality = ThisGenerality;
    }
  }
  
  ConstraintCode = *Current;
  ConstraintType = CurType;
}


void SelectionDAGLowering::
GetRegistersForValue(AsmOperandInfo &OpInfo, bool HasEarlyClobber,
                     std::set<unsigned> &OutputRegs, 
                     std::set<unsigned> &InputRegs) {
  // Compute whether this value requires an input register, an output register,
  // or both.
  bool isOutReg = false;
  bool isInReg = false;
  switch (OpInfo.Type) {
  case InlineAsm::isOutput:
    isOutReg = true;
    
    // If this is an early-clobber output, or if there is an input
    // constraint that matches this, we need to reserve the input register
    // so no other inputs allocate to it.
    isInReg = OpInfo.isEarlyClobber || OpInfo.hasMatchingInput;
    break;
  case InlineAsm::isInput:
    isInReg = true;
    isOutReg = false;
    break;
  case InlineAsm::isClobber:
    isOutReg = true;
    isInReg = true;
    break;
  }
  
  
  MachineFunction &MF = DAG.getMachineFunction();
  std::vector<unsigned> Regs;
  
  // If this is a constraint for a single physreg, or a constraint for a
  // register class, find it.
  std::pair<unsigned, const TargetRegisterClass*> PhysReg = 
    TLI.getRegForInlineAsmConstraint(OpInfo.ConstraintCode,
                                     OpInfo.ConstraintVT);

  unsigned NumRegs = 1;
  if (OpInfo.ConstraintVT != MVT::Other)
    NumRegs = TLI.getNumElements(OpInfo.ConstraintVT);
  MVT::ValueType RegVT;
  MVT::ValueType ValueVT = OpInfo.ConstraintVT;
  

  // If this is a constraint for a specific physical register, like {r17},
  // assign it now.
  if (PhysReg.first) {
    if (OpInfo.ConstraintVT == MVT::Other)
      ValueVT = *PhysReg.second->vt_begin();
    
    // Get the actual register value type.  This is important, because the user
    // may have asked for (e.g.) the AX register in i32 type.  We need to
    // remember that AX is actually i16 to get the right extension.
    RegVT = *PhysReg.second->vt_begin();
    
    // This is a explicit reference to a physical register.
    Regs.push_back(PhysReg.first);

    // If this is an expanded reference, add the rest of the regs to Regs.
    if (NumRegs != 1) {
      TargetRegisterClass::iterator I = PhysReg.second->begin();
      TargetRegisterClass::iterator E = PhysReg.second->end();
      for (; *I != PhysReg.first; ++I)
        assert(I != E && "Didn't find reg!"); 
      
      // Already added the first reg.
      --NumRegs; ++I;
      for (; NumRegs; --NumRegs, ++I) {
        assert(I != E && "Ran out of registers to allocate!");
        Regs.push_back(*I);
      }
    }
    OpInfo.AssignedRegs = RegsForValue(Regs, RegVT, ValueVT);
    OpInfo.MarkAllocatedRegs(isOutReg, isInReg, OutputRegs, InputRegs);
    return;
  }
  
  // Otherwise, if this was a reference to an LLVM register class, create vregs
  // for this reference.
  std::vector<unsigned> RegClassRegs;
  if (PhysReg.second) {
    // If this is an early clobber or tied register, our regalloc doesn't know
    // how to maintain the constraint.  If it isn't, go ahead and create vreg
    // and let the regalloc do the right thing.
    if (!OpInfo.hasMatchingInput && !OpInfo.isEarlyClobber &&
        // If there is some other early clobber and this is an input register,
        // then we are forced to pre-allocate the input reg so it doesn't
        // conflict with the earlyclobber.
        !(OpInfo.Type == InlineAsm::isInput && HasEarlyClobber)) {
      RegVT = *PhysReg.second->vt_begin();
      
      if (OpInfo.ConstraintVT == MVT::Other)
        ValueVT = RegVT;

      // Create the appropriate number of virtual registers.
      SSARegMap *RegMap = MF.getSSARegMap();
      for (; NumRegs; --NumRegs)
        Regs.push_back(RegMap->createVirtualRegister(PhysReg.second));
      
      OpInfo.AssignedRegs = RegsForValue(Regs, RegVT, ValueVT);
      OpInfo.MarkAllocatedRegs(isOutReg, isInReg, OutputRegs, InputRegs);
      return;
    }
    
    // Otherwise, we can't allocate it.  Let the code below figure out how to
    // maintain these constraints.
    RegClassRegs.assign(PhysReg.second->begin(), PhysReg.second->end());
    
  } else {
    // This is a reference to a register class that doesn't directly correspond
    // to an LLVM register class.  Allocate NumRegs consecutive, available,
    // registers from the class.
    RegClassRegs = TLI.getRegClassForInlineAsmConstraint(OpInfo.ConstraintCode,
                                                         OpInfo.ConstraintVT);
  }
  
  const MRegisterInfo *MRI = DAG.getTarget().getRegisterInfo();
  unsigned NumAllocated = 0;
  for (unsigned i = 0, e = RegClassRegs.size(); i != e; ++i) {
    unsigned Reg = RegClassRegs[i];
    // See if this register is available.
    if ((isOutReg && OutputRegs.count(Reg)) ||   // Already used.
        (isInReg  && InputRegs.count(Reg))) {    // Already used.
      // Make sure we find consecutive registers.
      NumAllocated = 0;
      continue;
    }
    
    // Check to see if this register is allocatable (i.e. don't give out the
    // stack pointer).
    const TargetRegisterClass *RC = isAllocatableRegister(Reg, MF, TLI, MRI);
    if (!RC) {
      // Make sure we find consecutive registers.
      NumAllocated = 0;
      continue;
    }
    
    // Okay, this register is good, we can use it.
    ++NumAllocated;

    // If we allocated enough consecutive registers, succeed.
    if (NumAllocated == NumRegs) {
      unsigned RegStart = (i-NumAllocated)+1;
      unsigned RegEnd   = i+1;
      // Mark all of the allocated registers used.
      for (unsigned i = RegStart; i != RegEnd; ++i)
        Regs.push_back(RegClassRegs[i]);
      
      OpInfo.AssignedRegs = RegsForValue(Regs, *RC->vt_begin(), 
                                         OpInfo.ConstraintVT);
      OpInfo.MarkAllocatedRegs(isOutReg, isInReg, OutputRegs, InputRegs);
      return;
    }
  }
  
  // Otherwise, we couldn't allocate enough registers for this.
  return;
}


/// visitInlineAsm - Handle a call to an InlineAsm object.
///
void SelectionDAGLowering::visitInlineAsm(CallInst &I) {
  InlineAsm *IA = cast<InlineAsm>(I.getOperand(0));

  /// ConstraintOperands - Information about all of the constraints.
  std::vector<AsmOperandInfo> ConstraintOperands;
  
  SDOperand Chain = getRoot();
  SDOperand Flag;
  
  std::set<unsigned> OutputRegs, InputRegs;

  // Do a prepass over the constraints, canonicalizing them, and building up the
  // ConstraintOperands list.
  std::vector<InlineAsm::ConstraintInfo>
    ConstraintInfos = IA->ParseConstraints();

  // SawEarlyClobber - Keep track of whether we saw an earlyclobber output
  // constraint.  If so, we can't let the register allocator allocate any input
  // registers, because it will not know to avoid the earlyclobbered output reg.
  bool SawEarlyClobber = false;
  
  unsigned OpNo = 1;   // OpNo - The operand of the CallInst.
  for (unsigned i = 0, e = ConstraintInfos.size(); i != e; ++i) {
    ConstraintOperands.push_back(AsmOperandInfo(ConstraintInfos[i]));
    AsmOperandInfo &OpInfo = ConstraintOperands.back();
    
    MVT::ValueType OpVT = MVT::Other;

    // Compute the value type for each operand.
    switch (OpInfo.Type) {
    case InlineAsm::isOutput:
      if (!OpInfo.isIndirect) {
        // The return value of the call is this value.  As such, there is no
        // corresponding argument.
        assert(I.getType() != Type::VoidTy && "Bad inline asm!");
        OpVT = TLI.getValueType(I.getType());
      } else {
        OpInfo.CallOperandVal = I.getOperand(OpNo++);
      }
      break;
    case InlineAsm::isInput:
      OpInfo.CallOperandVal = I.getOperand(OpNo++);
      break;
    case InlineAsm::isClobber:
      // Nothing to do.
      break;
    }

    // If this is an input or an indirect output, process the call argument.
    if (OpInfo.CallOperandVal) {
      OpInfo.CallOperand = getValue(OpInfo.CallOperandVal);
      const Type *OpTy = OpInfo.CallOperandVal->getType();
      // If this is an indirect operand, the operand is a pointer to the
      // accessed type.
      if (OpInfo.isIndirect)
        OpTy = cast<PointerType>(OpTy)->getElementType();
      
      // If OpTy is not a first-class value, it may be a struct/union that we
      // can tile with integers.
      if (!OpTy->isFirstClassType() && OpTy->isSized()) {
        unsigned BitSize = TD->getTypeSizeInBits(OpTy);
        switch (BitSize) {
        default: break;
        case 1:
        case 8:
        case 16:
        case 32:
        case 64:
          OpTy = IntegerType::get(BitSize);
          break;
        }
      }
      
      OpVT = TLI.getValueType(OpTy, true);
    }
    
    OpInfo.ConstraintVT = OpVT;
    
    // Compute the constraint code and ConstraintType to use.
    OpInfo.ComputeConstraintToUse(TLI);

    // Keep track of whether we see an earlyclobber.
    SawEarlyClobber |= OpInfo.isEarlyClobber;
    
    // If this is a memory input, and if the operand is not indirect, do what we
    // need to to provide an address for the memory input.
    if (OpInfo.ConstraintType == TargetLowering::C_Memory &&
        !OpInfo.isIndirect) {
      assert(OpInfo.Type == InlineAsm::isInput &&
             "Can only indirectify direct input operands!");
      
      // Memory operands really want the address of the value.  If we don't have
      // an indirect input, put it in the constpool if we can, otherwise spill
      // it to a stack slot.
      
      // If the operand is a float, integer, or vector constant, spill to a
      // constant pool entry to get its address.
      Value *OpVal = OpInfo.CallOperandVal;
      if (isa<ConstantFP>(OpVal) || isa<ConstantInt>(OpVal) ||
          isa<ConstantVector>(OpVal)) {
        OpInfo.CallOperand = DAG.getConstantPool(cast<Constant>(OpVal),
                                                 TLI.getPointerTy());
      } else {
        // Otherwise, create a stack slot and emit a store to it before the
        // asm.
        const Type *Ty = OpVal->getType();
        uint64_t TySize = TLI.getTargetData()->getTypeSize(Ty);
        unsigned Align  = TLI.getTargetData()->getPrefTypeAlignment(Ty);
        MachineFunction &MF = DAG.getMachineFunction();
        int SSFI = MF.getFrameInfo()->CreateStackObject(TySize, Align);
        SDOperand StackSlot = DAG.getFrameIndex(SSFI, TLI.getPointerTy());
        Chain = DAG.getStore(Chain, OpInfo.CallOperand, StackSlot, NULL, 0);
        OpInfo.CallOperand = StackSlot;
      }
     
      // There is no longer a Value* corresponding to this operand.
      OpInfo.CallOperandVal = 0;
      // It is now an indirect operand.
      OpInfo.isIndirect = true;
    }
    
    // If this constraint is for a specific register, allocate it before
    // anything else.
    if (OpInfo.ConstraintType == TargetLowering::C_Register)
      GetRegistersForValue(OpInfo, SawEarlyClobber, OutputRegs, InputRegs);
  }
  ConstraintInfos.clear();
  
  
  // Second pass - Loop over all of the operands, assigning virtual or physregs
  // to registerclass operands.
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    AsmOperandInfo &OpInfo = ConstraintOperands[i];
    
    // C_Register operands have already been allocated, Other/Memory don't need
    // to be.
    if (OpInfo.ConstraintType == TargetLowering::C_RegisterClass)
      GetRegistersForValue(OpInfo, SawEarlyClobber, OutputRegs, InputRegs);
  }    
  
  // AsmNodeOperands - The operands for the ISD::INLINEASM node.
  std::vector<SDOperand> AsmNodeOperands;
  AsmNodeOperands.push_back(SDOperand());  // reserve space for input chain
  AsmNodeOperands.push_back(
          DAG.getTargetExternalSymbol(IA->getAsmString().c_str(), MVT::Other));
  
  
  // Loop over all of the inputs, copying the operand values into the
  // appropriate registers and processing the output regs.
  RegsForValue RetValRegs;
  
  // IndirectStoresToEmit - The set of stores to emit after the inline asm node.
  std::vector<std::pair<RegsForValue, Value*> > IndirectStoresToEmit;
  
  for (unsigned i = 0, e = ConstraintOperands.size(); i != e; ++i) {
    AsmOperandInfo &OpInfo = ConstraintOperands[i];

    switch (OpInfo.Type) {
    case InlineAsm::isOutput: {
      if (OpInfo.ConstraintType != TargetLowering::C_RegisterClass &&
          OpInfo.ConstraintType != TargetLowering::C_Register) {
        // Memory output, or 'other' output (e.g. 'X' constraint).
        assert(OpInfo.isIndirect && "Memory output must be indirect operand");

        // Add information to the INLINEASM node to know about this output.
        unsigned ResOpType = 4/*MEM*/ | (1 << 3);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType, 
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(OpInfo.CallOperand);
        break;
      }

      // Otherwise, this is a register or register class output.

      // Copy the output from the appropriate register.  Find a register that
      // we can use.
      if (OpInfo.AssignedRegs.Regs.empty()) {
        cerr << "Couldn't allocate output reg for contraint '"
             << OpInfo.ConstraintCode << "'!\n";
        exit(1);
      }

      if (!OpInfo.isIndirect) {
        // This is the result value of the call.
        assert(RetValRegs.Regs.empty() &&
               "Cannot have multiple output constraints yet!");
        assert(I.getType() != Type::VoidTy && "Bad inline asm!");
        RetValRegs = OpInfo.AssignedRegs;
      } else {
        IndirectStoresToEmit.push_back(std::make_pair(OpInfo.AssignedRegs,
                                                      OpInfo.CallOperandVal));
      }
      
      // Add information to the INLINEASM node to know that this register is
      // set.
      OpInfo.AssignedRegs.AddInlineAsmOperands(2 /*REGDEF*/, DAG,
                                               AsmNodeOperands);
      break;
    }
    case InlineAsm::isInput: {
      SDOperand InOperandVal = OpInfo.CallOperand;
      
      if (isdigit(OpInfo.ConstraintCode[0])) {    // Matching constraint?
        // If this is required to match an output register we have already set,
        // just use its register.
        unsigned OperandNo = atoi(OpInfo.ConstraintCode.c_str());
        
        // Scan until we find the definition we already emitted of this operand.
        // When we find it, create a RegsForValue operand.
        unsigned CurOp = 2;  // The first operand.
        for (; OperandNo; --OperandNo) {
          // Advance to the next operand.
          unsigned NumOps = 
            cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getValue();
          assert(((NumOps & 7) == 2 /*REGDEF*/ ||
                  (NumOps & 7) == 4 /*MEM*/) &&
                 "Skipped past definitions?");
          CurOp += (NumOps>>3)+1;
        }

        unsigned NumOps = 
          cast<ConstantSDNode>(AsmNodeOperands[CurOp])->getValue();
        if ((NumOps & 7) == 2 /*REGDEF*/) {
          // Add NumOps>>3 registers to MatchedRegs.
          RegsForValue MatchedRegs;
          MatchedRegs.ValueVT = InOperandVal.getValueType();
          MatchedRegs.RegVT   = AsmNodeOperands[CurOp+1].getValueType();
          for (unsigned i = 0, e = NumOps>>3; i != e; ++i) {
            unsigned Reg =
              cast<RegisterSDNode>(AsmNodeOperands[++CurOp])->getReg();
            MatchedRegs.Regs.push_back(Reg);
          }
        
          // Use the produced MatchedRegs object to 
          MatchedRegs.getCopyToRegs(InOperandVal, DAG, Chain, Flag,
                                    TLI.getPointerTy());
          MatchedRegs.AddInlineAsmOperands(1 /*REGUSE*/, DAG, AsmNodeOperands);
          break;
        } else {
          assert((NumOps & 7) == 4/*MEM*/ && "Unknown matching constraint!");
          assert(0 && "matching constraints for memory operands unimp");
        }
      }
      
      if (OpInfo.ConstraintType == TargetLowering::C_Other) {
        assert(!OpInfo.isIndirect && 
               "Don't know how to handle indirect other inputs yet!");
        
        InOperandVal = TLI.isOperandValidForConstraint(InOperandVal,
                                                       OpInfo.ConstraintCode[0],
                                                       DAG);
        if (!InOperandVal.Val) {
          cerr << "Invalid operand for inline asm constraint '"
               << OpInfo.ConstraintCode << "'!\n";
          exit(1);
        }
        
        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType = 3 /*IMM*/ | (1 << 3);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType, 
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(InOperandVal);
        break;
      } else if (OpInfo.ConstraintType == TargetLowering::C_Memory) {
        assert(OpInfo.isIndirect && "Operand must be indirect to be a mem!");
        assert(InOperandVal.getValueType() == TLI.getPointerTy() &&
               "Memory operands expect pointer values");
               
        // Add information to the INLINEASM node to know about this input.
        unsigned ResOpType = 4/*MEM*/ | (1 << 3);
        AsmNodeOperands.push_back(DAG.getTargetConstant(ResOpType,
                                                        TLI.getPointerTy()));
        AsmNodeOperands.push_back(InOperandVal);
        break;
      }
        
      assert((OpInfo.ConstraintType == TargetLowering::C_RegisterClass ||
              OpInfo.ConstraintType == TargetLowering::C_Register) &&
             "Unknown constraint type!");
      assert(!OpInfo.isIndirect && 
             "Don't know how to handle indirect register inputs yet!");

      // Copy the input into the appropriate registers.
      assert(!OpInfo.AssignedRegs.Regs.empty() &&
             "Couldn't allocate input reg!");

      OpInfo.AssignedRegs.getCopyToRegs(InOperandVal, DAG, Chain, Flag, 
                                        TLI.getPointerTy());
      
      OpInfo.AssignedRegs.AddInlineAsmOperands(1/*REGUSE*/, DAG,
                                               AsmNodeOperands);
      break;
    }
    case InlineAsm::isClobber: {
      // Add the clobbered value to the operand list, so that the register
      // allocator is aware that the physreg got clobbered.
      if (!OpInfo.AssignedRegs.Regs.empty())
        OpInfo.AssignedRegs.AddInlineAsmOperands(2/*REGDEF*/, DAG,
                                                 AsmNodeOperands);
      break;
    }
    }
  }
  
  // Finish up input operands.
  AsmNodeOperands[0] = Chain;
  if (Flag.Val) AsmNodeOperands.push_back(Flag);
  
  Chain = DAG.getNode(ISD::INLINEASM, 
                      DAG.getNodeValueTypes(MVT::Other, MVT::Flag), 2,
                      &AsmNodeOperands[0], AsmNodeOperands.size());
  Flag = Chain.getValue(1);

  // If this asm returns a register value, copy the result from that register
  // and set it as the value of the call.
  if (!RetValRegs.Regs.empty()) {
    SDOperand Val = RetValRegs.getCopyFromRegs(DAG, Chain, Flag);
    
    // If the result of the inline asm is a vector, it may have the wrong
    // width/num elts.  Make sure to convert it to the right type with
    // vbit_convert.
    if (Val.getValueType() == MVT::Vector) {
      const VectorType *VTy = cast<VectorType>(I.getType());
      unsigned DesiredNumElts = VTy->getNumElements();
      MVT::ValueType DesiredEltVT = TLI.getValueType(VTy->getElementType());
      
      Val = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, Val, 
                        DAG.getConstant(DesiredNumElts, MVT::i32),
                        DAG.getValueType(DesiredEltVT));
    }
    
    setValue(&I, Val);
  }
  
  std::vector<std::pair<SDOperand, Value*> > StoresToEmit;
  
  // Process indirect outputs, first output all of the flagged copies out of
  // physregs.
  for (unsigned i = 0, e = IndirectStoresToEmit.size(); i != e; ++i) {
    RegsForValue &OutRegs = IndirectStoresToEmit[i].first;
    Value *Ptr = IndirectStoresToEmit[i].second;
    SDOperand OutVal = OutRegs.getCopyFromRegs(DAG, Chain, Flag);
    StoresToEmit.push_back(std::make_pair(OutVal, Ptr));
  }
  
  // Emit the non-flagged stores from the physregs.
  SmallVector<SDOperand, 8> OutChains;
  for (unsigned i = 0, e = StoresToEmit.size(); i != e; ++i)
    OutChains.push_back(DAG.getStore(Chain, StoresToEmit[i].first,
                                    getValue(StoresToEmit[i].second),
                                    StoresToEmit[i].second, 0));
  if (!OutChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &OutChains[0], OutChains.size());
  DAG.setRoot(Chain);
}


void SelectionDAGLowering::visitMalloc(MallocInst &I) {
  SDOperand Src = getValue(I.getOperand(0));

  MVT::ValueType IntPtr = TLI.getPointerTy();

  if (IntPtr < Src.getValueType())
    Src = DAG.getNode(ISD::TRUNCATE, IntPtr, Src);
  else if (IntPtr > Src.getValueType())
    Src = DAG.getNode(ISD::ZERO_EXTEND, IntPtr, Src);

  // Scale the source by the type size.
  uint64_t ElementSize = TD->getTypeSize(I.getType()->getElementType());
  Src = DAG.getNode(ISD::MUL, Src.getValueType(),
                    Src, getIntPtrConstant(ElementSize));

  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = Src;
  Entry.Ty = TLI.getTargetData()->getIntPtrType();
  Args.push_back(Entry);

  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), I.getType(), false, false, CallingConv::C, true,
                    DAG.getExternalSymbol("malloc", IntPtr),
                    Args, DAG);
  setValue(&I, Result.first);  // Pointers always fit in registers
  DAG.setRoot(Result.second);
}

void SelectionDAGLowering::visitFree(FreeInst &I) {
  TargetLowering::ArgListTy Args;
  TargetLowering::ArgListEntry Entry;
  Entry.Node = getValue(I.getOperand(0));
  Entry.Ty = TLI.getTargetData()->getIntPtrType();
  Args.push_back(Entry);
  MVT::ValueType IntPtr = TLI.getPointerTy();
  std::pair<SDOperand,SDOperand> Result =
    TLI.LowerCallTo(getRoot(), Type::VoidTy, false, false, CallingConv::C, true,
                    DAG.getExternalSymbol("free", IntPtr), Args, DAG);
  DAG.setRoot(Result.second);
}

// InsertAtEndOfBasicBlock - This method should be implemented by targets that
// mark instructions with the 'usesCustomDAGSchedInserter' flag.  These
// instructions are special in various ways, which require special support to
// insert.  The specified MachineInstr is created but not inserted into any
// basic blocks, and the scheduler passes ownership of it to this method.
MachineBasicBlock *TargetLowering::InsertAtEndOfBasicBlock(MachineInstr *MI,
                                                       MachineBasicBlock *MBB) {
  cerr << "If a target marks an instruction with "
       << "'usesCustomDAGSchedInserter', it must implement "
       << "TargetLowering::InsertAtEndOfBasicBlock!\n";
  abort();
  return 0;  
}

void SelectionDAGLowering::visitVAStart(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VASTART, MVT::Other, getRoot(), 
                          getValue(I.getOperand(1)), 
                          DAG.getSrcValue(I.getOperand(1))));
}

void SelectionDAGLowering::visitVAArg(VAArgInst &I) {
  SDOperand V = DAG.getVAArg(TLI.getValueType(I.getType()), getRoot(),
                             getValue(I.getOperand(0)),
                             DAG.getSrcValue(I.getOperand(0)));
  setValue(&I, V);
  DAG.setRoot(V.getValue(1));
}

void SelectionDAGLowering::visitVAEnd(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VAEND, MVT::Other, getRoot(),
                          getValue(I.getOperand(1)), 
                          DAG.getSrcValue(I.getOperand(1))));
}

void SelectionDAGLowering::visitVACopy(CallInst &I) {
  DAG.setRoot(DAG.getNode(ISD::VACOPY, MVT::Other, getRoot(), 
                          getValue(I.getOperand(1)), 
                          getValue(I.getOperand(2)),
                          DAG.getSrcValue(I.getOperand(1)),
                          DAG.getSrcValue(I.getOperand(2))));
}

/// ExpandScalarFormalArgs - Recursively expand the formal_argument node, either
/// bit_convert it or join a pair of them with a BUILD_PAIR when appropriate.
static SDOperand ExpandScalarFormalArgs(MVT::ValueType VT, SDNode *Arg,
                                        unsigned &i, SelectionDAG &DAG,
                                        TargetLowering &TLI) {
  if (TLI.getTypeAction(VT) != TargetLowering::Expand)
    return SDOperand(Arg, i++);

  MVT::ValueType EVT = TLI.getTypeToTransformTo(VT);
  unsigned NumVals = MVT::getSizeInBits(VT) / MVT::getSizeInBits(EVT);
  if (NumVals == 1) {
    return DAG.getNode(ISD::BIT_CONVERT, VT,
                       ExpandScalarFormalArgs(EVT, Arg, i, DAG, TLI));
  } else if (NumVals == 2) {
    SDOperand Lo = ExpandScalarFormalArgs(EVT, Arg, i, DAG, TLI);
    SDOperand Hi = ExpandScalarFormalArgs(EVT, Arg, i, DAG, TLI);
    if (!TLI.isLittleEndian())
      std::swap(Lo, Hi);
    return DAG.getNode(ISD::BUILD_PAIR, VT, Lo, Hi);
  } else {
    // Value scalarized into many values.  Unimp for now.
    assert(0 && "Cannot expand i64 -> i16 yet!");
  }
  return SDOperand();
}

/// TargetLowering::LowerArguments - This is the default LowerArguments
/// implementation, which just inserts a FORMAL_ARGUMENTS node.  FIXME: When all
/// targets are migrated to using FORMAL_ARGUMENTS, this hook should be 
/// integrated into SDISel.
std::vector<SDOperand> 
TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  const FunctionType *FTy = F.getFunctionType();
  const ParamAttrsList *Attrs = FTy->getParamAttrs();
  // Add CC# and isVararg as operands to the FORMAL_ARGUMENTS node.
  std::vector<SDOperand> Ops;
  Ops.push_back(DAG.getRoot());
  Ops.push_back(DAG.getConstant(F.getCallingConv(), getPointerTy()));
  Ops.push_back(DAG.getConstant(F.isVarArg(), getPointerTy()));

  // Add one result value for each formal argument.
  std::vector<MVT::ValueType> RetVals;
  unsigned j = 1;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I, ++j) {
    MVT::ValueType VT = getValueType(I->getType());
    unsigned Flags = ISD::ParamFlags::NoFlagSet;
    unsigned OriginalAlignment =
      getTargetData()->getABITypeAlignment(I->getType());

    // FIXME: Distinguish between a formal with no [sz]ext attribute from one
    // that is zero extended!
    if (Attrs && Attrs->paramHasAttr(j, ParamAttr::ZExt))
      Flags &= ~(ISD::ParamFlags::SExt);
    if (Attrs && Attrs->paramHasAttr(j, ParamAttr::SExt))
      Flags |= ISD::ParamFlags::SExt;
    if (Attrs && Attrs->paramHasAttr(j, ParamAttr::InReg))
      Flags |= ISD::ParamFlags::InReg;
    if (Attrs && Attrs->paramHasAttr(j, ParamAttr::StructRet))
      Flags |= ISD::ParamFlags::StructReturn;
    Flags |= (OriginalAlignment << ISD::ParamFlags::OrigAlignmentOffs);
    
    switch (getTypeAction(VT)) {
    default: assert(0 && "Unknown type action!");
    case Legal: 
      RetVals.push_back(VT);
      Ops.push_back(DAG.getConstant(Flags, MVT::i32));
      break;
    case Promote:
      RetVals.push_back(getTypeToTransformTo(VT));
      Ops.push_back(DAG.getConstant(Flags, MVT::i32));
      break;
    case Expand:
      if (VT != MVT::Vector) {
        // If this is a large integer, it needs to be broken up into small
        // integers.  Figure out what the destination type is and how many small
        // integers it turns into.
        MVT::ValueType NVT = getTypeToExpandTo(VT);
        unsigned NumVals = getNumElements(VT);
        for (unsigned i = 0; i != NumVals; ++i) {
          RetVals.push_back(NVT);
          // if it isn't first piece, alignment must be 1
          if (i > 0)
            Flags = (Flags & (~ISD::ParamFlags::OrigAlignment)) |
              (1 << ISD::ParamFlags::OrigAlignmentOffs);
          Ops.push_back(DAG.getConstant(Flags, MVT::i32));
        }
      } else {
        // Otherwise, this is a vector type.  We only support legal vectors
        // right now.
        unsigned NumElems = cast<VectorType>(I->getType())->getNumElements();
        const Type *EltTy = cast<VectorType>(I->getType())->getElementType();

        // Figure out if there is a Packed type corresponding to this Vector
        // type.  If so, convert to the vector type.
        MVT::ValueType TVT = MVT::getVectorType(getValueType(EltTy), NumElems);
        if (TVT != MVT::Other && isTypeLegal(TVT)) {
          RetVals.push_back(TVT);
          Ops.push_back(DAG.getConstant(Flags, MVT::i32));
        } else {
          assert(0 && "Don't support illegal by-val vector arguments yet!");
        }
      }
      break;
    }
  }

  RetVals.push_back(MVT::Other);
  
  // Create the node.
  SDNode *Result = DAG.getNode(ISD::FORMAL_ARGUMENTS,
                               DAG.getNodeValueTypes(RetVals), RetVals.size(),
                               &Ops[0], Ops.size()).Val;
  
  DAG.setRoot(SDOperand(Result, Result->getNumValues()-1));

  // Set up the return result vector.
  Ops.clear();
  unsigned i = 0;
  unsigned Idx = 1;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; 
      ++I, ++Idx) {
    MVT::ValueType VT = getValueType(I->getType());
    
    switch (getTypeAction(VT)) {
    default: assert(0 && "Unknown type action!");
    case Legal: 
      Ops.push_back(SDOperand(Result, i++));
      break;
    case Promote: {
      SDOperand Op(Result, i++);
      if (MVT::isInteger(VT)) {
        if (Attrs && Attrs->paramHasAttr(Idx, ParamAttr::SExt))
          Op = DAG.getNode(ISD::AssertSext, Op.getValueType(), Op,
                           DAG.getValueType(VT));
        else if (Attrs && Attrs->paramHasAttr(Idx, ParamAttr::ZExt))
          Op = DAG.getNode(ISD::AssertZext, Op.getValueType(), Op,
                           DAG.getValueType(VT));
        Op = DAG.getNode(ISD::TRUNCATE, VT, Op);
      } else {
        assert(MVT::isFloatingPoint(VT) && "Not int or FP?");
        Op = DAG.getNode(ISD::FP_ROUND, VT, Op);
      }
      Ops.push_back(Op);
      break;
    }
    case Expand:
      if (VT != MVT::Vector) {
        // If this is a large integer or a floating point node that needs to be
        // expanded, it needs to be reassembled from small integers.  Figure out
        // what the source elt type is and how many small integers it is.
        Ops.push_back(ExpandScalarFormalArgs(VT, Result, i, DAG, *this));
      } else {
        // Otherwise, this is a vector type.  We only support legal vectors
        // right now.
        const VectorType *PTy = cast<VectorType>(I->getType());
        unsigned NumElems = PTy->getNumElements();
        const Type *EltTy = PTy->getElementType();

        // Figure out if there is a Packed type corresponding to this Vector
        // type.  If so, convert to the vector type.
        MVT::ValueType TVT = MVT::getVectorType(getValueType(EltTy), NumElems);
        if (TVT != MVT::Other && isTypeLegal(TVT)) {
          SDOperand N = SDOperand(Result, i++);
          // Handle copies from generic vectors to registers.
          N = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, N,
                          DAG.getConstant(NumElems, MVT::i32), 
                          DAG.getValueType(getValueType(EltTy)));
          Ops.push_back(N);
        } else {
          assert(0 && "Don't support illegal by-val vector arguments yet!");
          abort();
        }
      }
      break;
    }
  }
  return Ops;
}


/// ExpandScalarCallArgs - Recursively expand call argument node by
/// bit_converting it or extract a pair of elements from the larger  node.
static void ExpandScalarCallArgs(MVT::ValueType VT, SDOperand Arg,
                                 unsigned Flags,
                                 SmallVector<SDOperand, 32> &Ops,
                                 SelectionDAG &DAG,
                                 TargetLowering &TLI,
                                 bool isFirst = true) {

  if (TLI.getTypeAction(VT) != TargetLowering::Expand) {
    // if it isn't first piece, alignment must be 1
    if (!isFirst)
      Flags = (Flags & (~ISD::ParamFlags::OrigAlignment)) |
        (1 << ISD::ParamFlags::OrigAlignmentOffs);
    Ops.push_back(Arg);
    Ops.push_back(DAG.getConstant(Flags, MVT::i32));
    return;
  }

  MVT::ValueType EVT = TLI.getTypeToTransformTo(VT);
  unsigned NumVals = MVT::getSizeInBits(VT) / MVT::getSizeInBits(EVT);
  if (NumVals == 1) {
    Arg = DAG.getNode(ISD::BIT_CONVERT, EVT, Arg);
    ExpandScalarCallArgs(EVT, Arg, Flags, Ops, DAG, TLI, isFirst);
  } else if (NumVals == 2) {
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, EVT, Arg,
                               DAG.getConstant(0, TLI.getPointerTy()));
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, EVT, Arg,
                               DAG.getConstant(1, TLI.getPointerTy()));
    if (!TLI.isLittleEndian())
      std::swap(Lo, Hi);
    ExpandScalarCallArgs(EVT, Lo, Flags, Ops, DAG, TLI, isFirst);
    ExpandScalarCallArgs(EVT, Hi, Flags, Ops, DAG, TLI, false);
  } else {
    // Value scalarized into many values.  Unimp for now.
    assert(0 && "Cannot expand i64 -> i16 yet!");
  }
}

/// TargetLowering::LowerCallTo - This is the default LowerCallTo
/// implementation, which just inserts an ISD::CALL node, which is later custom
/// lowered by the target to something concrete.  FIXME: When all targets are
/// migrated to using ISD::CALL, this hook should be integrated into SDISel.
std::pair<SDOperand, SDOperand>
TargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy, 
                            bool RetTyIsSigned, bool isVarArg,
                            unsigned CallingConv, bool isTailCall, 
                            SDOperand Callee,
                            ArgListTy &Args, SelectionDAG &DAG) {
  SmallVector<SDOperand, 32> Ops;
  Ops.push_back(Chain);   // Op#0 - Chain
  Ops.push_back(DAG.getConstant(CallingConv, getPointerTy())); // Op#1 - CC
  Ops.push_back(DAG.getConstant(isVarArg, getPointerTy()));    // Op#2 - VarArg
  Ops.push_back(DAG.getConstant(isTailCall, getPointerTy()));  // Op#3 - Tail
  Ops.push_back(Callee);
  
  // Handle all of the outgoing arguments.
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    MVT::ValueType VT = getValueType(Args[i].Ty);
    SDOperand Op = Args[i].Node;
    unsigned Flags = ISD::ParamFlags::NoFlagSet;
    unsigned OriginalAlignment =
      getTargetData()->getABITypeAlignment(Args[i].Ty);
    
    if (Args[i].isSExt)
      Flags |= ISD::ParamFlags::SExt;
    if (Args[i].isZExt)
      Flags |= ISD::ParamFlags::ZExt;
    if (Args[i].isInReg)
      Flags |= ISD::ParamFlags::InReg;
    if (Args[i].isSRet)
      Flags |= ISD::ParamFlags::StructReturn;
    Flags |= OriginalAlignment << ISD::ParamFlags::OrigAlignmentOffs;
    
    switch (getTypeAction(VT)) {
    default: assert(0 && "Unknown type action!");
    case Legal:
      Ops.push_back(Op);
      Ops.push_back(DAG.getConstant(Flags, MVT::i32));
      break;
    case Promote:
      if (MVT::isInteger(VT)) {
        unsigned ExtOp;
        if (Args[i].isSExt)
          ExtOp = ISD::SIGN_EXTEND;
        else if (Args[i].isZExt)
          ExtOp = ISD::ZERO_EXTEND;
        else
          ExtOp = ISD::ANY_EXTEND;
        Op = DAG.getNode(ExtOp, getTypeToTransformTo(VT), Op);
      } else {
        assert(MVT::isFloatingPoint(VT) && "Not int or FP?");
        Op = DAG.getNode(ISD::FP_EXTEND, getTypeToTransformTo(VT), Op);
      }
      Ops.push_back(Op);
      Ops.push_back(DAG.getConstant(Flags, MVT::i32));
      break;
    case Expand:
      if (VT != MVT::Vector) {
        // If this is a large integer, it needs to be broken down into small
        // integers.  Figure out what the source elt type is and how many small
        // integers it is.
        ExpandScalarCallArgs(VT, Op, Flags, Ops, DAG, *this);
      } else {
        // Otherwise, this is a vector type.  We only support legal vectors
        // right now.
        const VectorType *PTy = cast<VectorType>(Args[i].Ty);
        unsigned NumElems = PTy->getNumElements();
        const Type *EltTy = PTy->getElementType();
        
        // Figure out if there is a Packed type corresponding to this Vector
        // type.  If so, convert to the vector type.
        MVT::ValueType TVT = MVT::getVectorType(getValueType(EltTy), NumElems);
        if (TVT != MVT::Other && isTypeLegal(TVT)) {
          // Insert a VBIT_CONVERT of the MVT::Vector type to the vector type.
          Op = DAG.getNode(ISD::VBIT_CONVERT, TVT, Op);
          Ops.push_back(Op);
          Ops.push_back(DAG.getConstant(Flags, MVT::i32));
        } else {
          assert(0 && "Don't support illegal by-val vector call args yet!");
          abort();
        }
      }
      break;
    }
  }
  
  // Figure out the result value types.
  SmallVector<MVT::ValueType, 4> RetTys;

  if (RetTy != Type::VoidTy) {
    MVT::ValueType VT = getValueType(RetTy);
    switch (getTypeAction(VT)) {
    default: assert(0 && "Unknown type action!");
    case Legal:
      RetTys.push_back(VT);
      break;
    case Promote:
      RetTys.push_back(getTypeToTransformTo(VT));
      break;
    case Expand:
      if (VT != MVT::Vector) {
        // If this is a large integer, it needs to be reassembled from small
        // integers.  Figure out what the source elt type is and how many small
        // integers it is.
        MVT::ValueType NVT = getTypeToExpandTo(VT);
        unsigned NumVals = getNumElements(VT);
        for (unsigned i = 0; i != NumVals; ++i)
          RetTys.push_back(NVT);
      } else {
        // Otherwise, this is a vector type.  We only support legal vectors
        // right now.
        const VectorType *PTy = cast<VectorType>(RetTy);
        unsigned NumElems = PTy->getNumElements();
        const Type *EltTy = PTy->getElementType();
        
        // Figure out if there is a Packed type corresponding to this Vector
        // type.  If so, convert to the vector type.
        MVT::ValueType TVT = MVT::getVectorType(getValueType(EltTy), NumElems);
        if (TVT != MVT::Other && isTypeLegal(TVT)) {
          RetTys.push_back(TVT);
        } else {
          assert(0 && "Don't support illegal by-val vector call results yet!");
          abort();
        }
      }
    }    
  }
  
  RetTys.push_back(MVT::Other);  // Always has a chain.
  
  // Finally, create the CALL node.
  SDOperand Res = DAG.getNode(ISD::CALL,
                              DAG.getVTList(&RetTys[0], RetTys.size()),
                              &Ops[0], Ops.size());
  
  // This returns a pair of operands.  The first element is the
  // return value for the function (if RetTy is not VoidTy).  The second
  // element is the outgoing token chain.
  SDOperand ResVal;
  if (RetTys.size() != 1) {
    MVT::ValueType VT = getValueType(RetTy);
    if (RetTys.size() == 2) {
      ResVal = Res;
      
      // If this value was promoted, truncate it down.
      if (ResVal.getValueType() != VT) {
        if (VT == MVT::Vector) {
          // Insert a VBIT_CONVERT to convert from the packed result type to the
          // MVT::Vector type.
          unsigned NumElems = cast<VectorType>(RetTy)->getNumElements();
          const Type *EltTy = cast<VectorType>(RetTy)->getElementType();
          
          // Figure out if there is a Packed type corresponding to this Vector
          // type.  If so, convert to the vector type.
          MVT::ValueType TVT = MVT::getVectorType(getValueType(EltTy),NumElems);
          if (TVT != MVT::Other && isTypeLegal(TVT)) {
            // Insert a VBIT_CONVERT of the FORMAL_ARGUMENTS to a
            // "N x PTyElementVT" MVT::Vector type.
            ResVal = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, ResVal,
                                 DAG.getConstant(NumElems, MVT::i32), 
                                 DAG.getValueType(getValueType(EltTy)));
          } else {
            abort();
          }
        } else if (MVT::isInteger(VT)) {
          unsigned AssertOp = ISD::AssertSext;
          if (!RetTyIsSigned)
            AssertOp = ISD::AssertZext;
          ResVal = DAG.getNode(AssertOp, ResVal.getValueType(), ResVal, 
                               DAG.getValueType(VT));
          ResVal = DAG.getNode(ISD::TRUNCATE, VT, ResVal);
        } else {
          assert(MVT::isFloatingPoint(VT));
          if (getTypeAction(VT) == Expand)
            ResVal = DAG.getNode(ISD::BIT_CONVERT, VT, ResVal);
          else
            ResVal = DAG.getNode(ISD::FP_ROUND, VT, ResVal);
        }
      }
    } else if (RetTys.size() == 3) {
      ResVal = DAG.getNode(ISD::BUILD_PAIR, VT, 
                           Res.getValue(0), Res.getValue(1));
      
    } else {
      assert(0 && "Case not handled yet!");
    }
  }
  
  return std::make_pair(ResVal, Res.getValue(Res.Val->getNumValues()-1));
}

SDOperand TargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  assert(0 && "LowerOperation not implemented for this target!");
  abort();
  return SDOperand();
}

SDOperand TargetLowering::CustomPromoteOperation(SDOperand Op,
                                                 SelectionDAG &DAG) {
  assert(0 && "CustomPromoteOperation not implemented for this target!");
  abort();
  return SDOperand();
}

/// getMemsetValue - Vectorized representation of the memset value
/// operand.
static SDOperand getMemsetValue(SDOperand Value, MVT::ValueType VT,
                                SelectionDAG &DAG) {
  MVT::ValueType CurVT = VT;
  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(Value)) {
    uint64_t Val   = C->getValue() & 255;
    unsigned Shift = 8;
    while (CurVT != MVT::i8) {
      Val = (Val << Shift) | Val;
      Shift <<= 1;
      CurVT = (MVT::ValueType)((unsigned)CurVT - 1);
    }
    return DAG.getConstant(Val, VT);
  } else {
    Value = DAG.getNode(ISD::ZERO_EXTEND, VT, Value);
    unsigned Shift = 8;
    while (CurVT != MVT::i8) {
      Value =
        DAG.getNode(ISD::OR, VT,
                    DAG.getNode(ISD::SHL, VT, Value,
                                DAG.getConstant(Shift, MVT::i8)), Value);
      Shift <<= 1;
      CurVT = (MVT::ValueType)((unsigned)CurVT - 1);
    }

    return Value;
  }
}

/// getMemsetStringVal - Similar to getMemsetValue. Except this is only
/// used when a memcpy is turned into a memset when the source is a constant
/// string ptr.
static SDOperand getMemsetStringVal(MVT::ValueType VT,
                                    SelectionDAG &DAG, TargetLowering &TLI,
                                    std::string &Str, unsigned Offset) {
  uint64_t Val = 0;
  unsigned MSB = getSizeInBits(VT) / 8;
  if (TLI.isLittleEndian())
    Offset = Offset + MSB - 1;
  for (unsigned i = 0; i != MSB; ++i) {
    Val = (Val << 8) | (unsigned char)Str[Offset];
    Offset += TLI.isLittleEndian() ? -1 : 1;
  }
  return DAG.getConstant(Val, VT);
}

/// getMemBasePlusOffset - Returns base and offset node for the 
static SDOperand getMemBasePlusOffset(SDOperand Base, unsigned Offset,
                                      SelectionDAG &DAG, TargetLowering &TLI) {
  MVT::ValueType VT = Base.getValueType();
  return DAG.getNode(ISD::ADD, VT, Base, DAG.getConstant(Offset, VT));
}

/// MeetsMaxMemopRequirement - Determines if the number of memory ops required
/// to replace the memset / memcpy is below the threshold. It also returns the
/// types of the sequence of  memory ops to perform memset / memcpy.
static bool MeetsMaxMemopRequirement(std::vector<MVT::ValueType> &MemOps,
                                     unsigned Limit, uint64_t Size,
                                     unsigned Align, TargetLowering &TLI) {
  MVT::ValueType VT;

  if (TLI.allowsUnalignedMemoryAccesses()) {
    VT = MVT::i64;
  } else {
    switch (Align & 7) {
    case 0:
      VT = MVT::i64;
      break;
    case 4:
      VT = MVT::i32;
      break;
    case 2:
      VT = MVT::i16;
      break;
    default:
      VT = MVT::i8;
      break;
    }
  }

  MVT::ValueType LVT = MVT::i64;
  while (!TLI.isTypeLegal(LVT))
    LVT = (MVT::ValueType)((unsigned)LVT - 1);
  assert(MVT::isInteger(LVT));

  if (VT > LVT)
    VT = LVT;

  unsigned NumMemOps = 0;
  while (Size != 0) {
    unsigned VTSize = getSizeInBits(VT) / 8;
    while (VTSize > Size) {
      VT = (MVT::ValueType)((unsigned)VT - 1);
      VTSize >>= 1;
    }
    assert(MVT::isInteger(VT));

    if (++NumMemOps > Limit)
      return false;
    MemOps.push_back(VT);
    Size -= VTSize;
  }

  return true;
}

void SelectionDAGLowering::visitMemIntrinsic(CallInst &I, unsigned Op) {
  SDOperand Op1 = getValue(I.getOperand(1));
  SDOperand Op2 = getValue(I.getOperand(2));
  SDOperand Op3 = getValue(I.getOperand(3));
  SDOperand Op4 = getValue(I.getOperand(4));
  unsigned Align = (unsigned)cast<ConstantSDNode>(Op4)->getValue();
  if (Align == 0) Align = 1;

  if (ConstantSDNode *Size = dyn_cast<ConstantSDNode>(Op3)) {
    std::vector<MVT::ValueType> MemOps;

    // Expand memset / memcpy to a series of load / store ops
    // if the size operand falls below a certain threshold.
    SmallVector<SDOperand, 8> OutChains;
    switch (Op) {
    default: break;  // Do nothing for now.
    case ISD::MEMSET: {
      if (MeetsMaxMemopRequirement(MemOps, TLI.getMaxStoresPerMemset(),
                                   Size->getValue(), Align, TLI)) {
        unsigned NumMemOps = MemOps.size();
        unsigned Offset = 0;
        for (unsigned i = 0; i < NumMemOps; i++) {
          MVT::ValueType VT = MemOps[i];
          unsigned VTSize = getSizeInBits(VT) / 8;
          SDOperand Value = getMemsetValue(Op2, VT, DAG);
          SDOperand Store = DAG.getStore(getRoot(), Value,
                                    getMemBasePlusOffset(Op1, Offset, DAG, TLI),
                                         I.getOperand(1), Offset);
          OutChains.push_back(Store);
          Offset += VTSize;
        }
      }
      break;
    }
    case ISD::MEMCPY: {
      if (MeetsMaxMemopRequirement(MemOps, TLI.getMaxStoresPerMemcpy(),
                                   Size->getValue(), Align, TLI)) {
        unsigned NumMemOps = MemOps.size();
        unsigned SrcOff = 0, DstOff = 0, SrcDelta = 0;
        GlobalAddressSDNode *G = NULL;
        std::string Str;
        bool CopyFromStr = false;

        if (Op2.getOpcode() == ISD::GlobalAddress)
          G = cast<GlobalAddressSDNode>(Op2);
        else if (Op2.getOpcode() == ISD::ADD &&
                 Op2.getOperand(0).getOpcode() == ISD::GlobalAddress &&
                 Op2.getOperand(1).getOpcode() == ISD::Constant) {
          G = cast<GlobalAddressSDNode>(Op2.getOperand(0));
          SrcDelta = cast<ConstantSDNode>(Op2.getOperand(1))->getValue();
        }
        if (G) {
          GlobalVariable *GV = dyn_cast<GlobalVariable>(G->getGlobal());
          if (GV && GV->isConstant()) {
            Str = GV->getStringValue(false);
            if (!Str.empty()) {
              CopyFromStr = true;
              SrcOff += SrcDelta;
            }
          }
        }

        for (unsigned i = 0; i < NumMemOps; i++) {
          MVT::ValueType VT = MemOps[i];
          unsigned VTSize = getSizeInBits(VT) / 8;
          SDOperand Value, Chain, Store;

          if (CopyFromStr) {
            Value = getMemsetStringVal(VT, DAG, TLI, Str, SrcOff);
            Chain = getRoot();
            Store =
              DAG.getStore(Chain, Value,
                           getMemBasePlusOffset(Op1, DstOff, DAG, TLI),
                           I.getOperand(1), DstOff);
          } else {
            Value = DAG.getLoad(VT, getRoot(),
                        getMemBasePlusOffset(Op2, SrcOff, DAG, TLI),
                        I.getOperand(2), SrcOff);
            Chain = Value.getValue(1);
            Store =
              DAG.getStore(Chain, Value,
                           getMemBasePlusOffset(Op1, DstOff, DAG, TLI),
                           I.getOperand(1), DstOff);
          }
          OutChains.push_back(Store);
          SrcOff += VTSize;
          DstOff += VTSize;
        }
      }
      break;
    }
    }

    if (!OutChains.empty()) {
      DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other,
                  &OutChains[0], OutChains.size()));
      return;
    }
  }

  DAG.setRoot(DAG.getNode(Op, MVT::Other, getRoot(), Op1, Op2, Op3, Op4));
}

//===----------------------------------------------------------------------===//
// SelectionDAGISel code
//===----------------------------------------------------------------------===//

unsigned SelectionDAGISel::MakeReg(MVT::ValueType VT) {
  return RegMap->createVirtualRegister(TLI.getRegClassFor(VT));
}

void SelectionDAGISel::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<AliasAnalysis>();
  AU.setPreservesAll();
}



bool SelectionDAGISel::runOnFunction(Function &Fn) {
  MachineFunction &MF = MachineFunction::construct(&Fn, TLI.getTargetMachine());
  RegMap = MF.getSSARegMap();
  DOUT << "\n\n\n=== " << Fn.getName() << "\n";

  FunctionLoweringInfo FuncInfo(TLI, Fn, MF);

  for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
    SelectBasicBlock(I, MF, FuncInfo);

  // Add function live-ins to entry block live-in set.
  BasicBlock *EntryBB = &Fn.getEntryBlock();
  BB = FuncInfo.MBBMap[EntryBB];
  if (!MF.livein_empty())
    for (MachineFunction::livein_iterator I = MF.livein_begin(),
           E = MF.livein_end(); I != E; ++I)
      BB->addLiveIn(I->first);

  return true;
}

SDOperand SelectionDAGLowering::CopyValueToVirtualRegister(Value *V, 
                                                           unsigned Reg) {
  SDOperand Op = getValue(V);
  assert((Op.getOpcode() != ISD::CopyFromReg ||
          cast<RegisterSDNode>(Op.getOperand(1))->getReg() != Reg) &&
         "Copy from a reg to the same reg!");
  
  // If this type is not legal, we must make sure to not create an invalid
  // register use.
  MVT::ValueType SrcVT = Op.getValueType();
  MVT::ValueType DestVT = TLI.getTypeToTransformTo(SrcVT);
  if (SrcVT == DestVT) {
    return DAG.getCopyToReg(getRoot(), Reg, Op);
  } else if (SrcVT == MVT::Vector) {
    // Handle copies from generic vectors to registers.
    MVT::ValueType PTyElementVT, PTyLegalElementVT;
    unsigned NE = TLI.getVectorTypeBreakdown(cast<VectorType>(V->getType()),
                                             PTyElementVT, PTyLegalElementVT);
    
    // Insert a VBIT_CONVERT of the input vector to a "N x PTyElementVT" 
    // MVT::Vector type.
    Op = DAG.getNode(ISD::VBIT_CONVERT, MVT::Vector, Op,
                     DAG.getConstant(NE, MVT::i32), 
                     DAG.getValueType(PTyElementVT));

    // Loop over all of the elements of the resultant vector,
    // VEXTRACT_VECTOR_ELT'ing them, converting them to PTyLegalElementVT, then
    // copying them into output registers.
    SmallVector<SDOperand, 8> OutChains;
    SDOperand Root = getRoot();
    for (unsigned i = 0; i != NE; ++i) {
      SDOperand Elt = DAG.getNode(ISD::VEXTRACT_VECTOR_ELT, PTyElementVT,
                                  Op, DAG.getConstant(i, TLI.getPointerTy()));
      if (PTyElementVT == PTyLegalElementVT) {
        // Elements are legal.
        OutChains.push_back(DAG.getCopyToReg(Root, Reg++, Elt));
      } else if (PTyLegalElementVT > PTyElementVT) {
        // Elements are promoted.
        if (MVT::isFloatingPoint(PTyLegalElementVT))
          Elt = DAG.getNode(ISD::FP_EXTEND, PTyLegalElementVT, Elt);
        else
          Elt = DAG.getNode(ISD::ANY_EXTEND, PTyLegalElementVT, Elt);
        OutChains.push_back(DAG.getCopyToReg(Root, Reg++, Elt));
      } else {
        // Elements are expanded.
        // The src value is expanded into multiple registers.
        SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, PTyLegalElementVT,
                                   Elt, DAG.getConstant(0, TLI.getPointerTy()));
        SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, PTyLegalElementVT,
                                   Elt, DAG.getConstant(1, TLI.getPointerTy()));
        OutChains.push_back(DAG.getCopyToReg(Root, Reg++, Lo));
        OutChains.push_back(DAG.getCopyToReg(Root, Reg++, Hi));
      }
    }
    return DAG.getNode(ISD::TokenFactor, MVT::Other,
                       &OutChains[0], OutChains.size());
  } else if (TLI.getTypeAction(SrcVT) == TargetLowering::Promote) {
    // The src value is promoted to the register.
    if (MVT::isFloatingPoint(SrcVT))
      Op = DAG.getNode(ISD::FP_EXTEND, DestVT, Op);
    else
      Op = DAG.getNode(ISD::ANY_EXTEND, DestVT, Op);
    return DAG.getCopyToReg(getRoot(), Reg, Op);
  } else  {
    DestVT = TLI.getTypeToExpandTo(SrcVT);
    unsigned NumVals = TLI.getNumElements(SrcVT);
    if (NumVals == 1)
      return DAG.getCopyToReg(getRoot(), Reg,
                              DAG.getNode(ISD::BIT_CONVERT, DestVT, Op));
    assert(NumVals == 2 && "1 to 4 (and more) expansion not implemented!");
    // The src value is expanded into multiple registers.
    SDOperand Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DestVT,
                               Op, DAG.getConstant(0, TLI.getPointerTy()));
    SDOperand Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DestVT,
                               Op, DAG.getConstant(1, TLI.getPointerTy()));
    Op = DAG.getCopyToReg(getRoot(), Reg, Lo);
    return DAG.getCopyToReg(Op, Reg+1, Hi);
  }
}

void SelectionDAGISel::
LowerArguments(BasicBlock *LLVMBB, SelectionDAGLowering &SDL,
               std::vector<SDOperand> &UnorderedChains) {
  // If this is the entry block, emit arguments.
  Function &F = *LLVMBB->getParent();
  FunctionLoweringInfo &FuncInfo = SDL.FuncInfo;
  SDOperand OldRoot = SDL.DAG.getRoot();
  std::vector<SDOperand> Args = TLI.LowerArguments(F, SDL.DAG);

  unsigned a = 0;
  for (Function::arg_iterator AI = F.arg_begin(), E = F.arg_end();
       AI != E; ++AI, ++a)
    if (!AI->use_empty()) {
      SDL.setValue(AI, Args[a]);

      // If this argument is live outside of the entry block, insert a copy from
      // whereever we got it to the vreg that other BB's will reference it as.
      DenseMap<const Value*, unsigned>::iterator VMI=FuncInfo.ValueMap.find(AI);
      if (VMI != FuncInfo.ValueMap.end()) {
        SDOperand Copy = SDL.CopyValueToVirtualRegister(AI, VMI->second);
        UnorderedChains.push_back(Copy);
      }
    }

  // Finally, if the target has anything special to do, allow it to do so.
  // FIXME: this should insert code into the DAG!
  EmitFunctionEntryCode(F, SDL.DAG.getMachineFunction());
}

void SelectionDAGISel::BuildSelectionDAG(SelectionDAG &DAG, BasicBlock *LLVMBB,
       std::vector<std::pair<MachineInstr*, unsigned> > &PHINodesToUpdate,
                                         FunctionLoweringInfo &FuncInfo) {
  SelectionDAGLowering SDL(DAG, TLI, FuncInfo);

  std::vector<SDOperand> UnorderedChains;

  // Lower any arguments needed in this block if this is the entry block.
  if (LLVMBB == &LLVMBB->getParent()->getEntryBlock())
    LowerArguments(LLVMBB, SDL, UnorderedChains);

  BB = FuncInfo.MBBMap[LLVMBB];
  SDL.setCurrentBasicBlock(BB);

  // Lower all of the non-terminator instructions.
  for (BasicBlock::iterator I = LLVMBB->begin(), E = --LLVMBB->end();
       I != E; ++I)
    SDL.visit(*I);
    
  // Lower call part of invoke.
  InvokeInst *Invoke = dyn_cast<InvokeInst>(LLVMBB->getTerminator());
  if (Invoke) SDL.visitInvoke(*Invoke, false);
  
  // Ensure that all instructions which are used outside of their defining
  // blocks are available as virtual registers.
  for (BasicBlock::iterator I = LLVMBB->begin(), E = LLVMBB->end(); I != E;++I)
    if (!I->use_empty() && !isa<PHINode>(I)) {
      DenseMap<const Value*, unsigned>::iterator VMI =FuncInfo.ValueMap.find(I);
      if (VMI != FuncInfo.ValueMap.end())
        UnorderedChains.push_back(
                                SDL.CopyValueToVirtualRegister(I, VMI->second));
    }

  // Handle PHI nodes in successor blocks.  Emit code into the SelectionDAG to
  // ensure constants are generated when needed.  Remember the virtual registers
  // that need to be added to the Machine PHI nodes as input.  We cannot just
  // directly add them, because expansion might result in multiple MBB's for one
  // BB.  As such, the start of the BB might correspond to a different MBB than
  // the end.
  //
  TerminatorInst *TI = LLVMBB->getTerminator();

  // Emit constants only once even if used by multiple PHI nodes.
  std::map<Constant*, unsigned> ConstantsOut;
  
  // Vector bool would be better, but vector<bool> is really slow.
  std::vector<unsigned char> SuccsHandled;
  if (TI->getNumSuccessors())
    SuccsHandled.resize(BB->getParent()->getNumBlockIDs());
    
  // Check successor nodes PHI nodes that expect a constant to be available from
  // this block.
  for (unsigned succ = 0, e = TI->getNumSuccessors(); succ != e; ++succ) {
    BasicBlock *SuccBB = TI->getSuccessor(succ);
    if (!isa<PHINode>(SuccBB->begin())) continue;
    MachineBasicBlock *SuccMBB = FuncInfo.MBBMap[SuccBB];
    
    // If this terminator has multiple identical successors (common for
    // switches), only handle each succ once.
    unsigned SuccMBBNo = SuccMBB->getNumber();
    if (SuccsHandled[SuccMBBNo]) continue;
    SuccsHandled[SuccMBBNo] = true;
    
    MachineBasicBlock::iterator MBBI = SuccMBB->begin();
    PHINode *PN;

    // At this point we know that there is a 1-1 correspondence between LLVM PHI
    // nodes and Machine PHI nodes, but the incoming operands have not been
    // emitted yet.
    for (BasicBlock::iterator I = SuccBB->begin();
         (PN = dyn_cast<PHINode>(I)); ++I) {
      // Ignore dead phi's.
      if (PN->use_empty()) continue;
      
      unsigned Reg;
      Value *PHIOp = PN->getIncomingValueForBlock(LLVMBB);
      
      if (Constant *C = dyn_cast<Constant>(PHIOp)) {
        unsigned &RegOut = ConstantsOut[C];
        if (RegOut == 0) {
          RegOut = FuncInfo.CreateRegForValue(C);
          UnorderedChains.push_back(
                           SDL.CopyValueToVirtualRegister(C, RegOut));
        }
        Reg = RegOut;
      } else {
        Reg = FuncInfo.ValueMap[PHIOp];
        if (Reg == 0) {
          assert(isa<AllocaInst>(PHIOp) &&
                 FuncInfo.StaticAllocaMap.count(cast<AllocaInst>(PHIOp)) &&
                 "Didn't codegen value into a register!??");
          Reg = FuncInfo.CreateRegForValue(PHIOp);
          UnorderedChains.push_back(
                           SDL.CopyValueToVirtualRegister(PHIOp, Reg));
        }
      }

      // Remember that this register needs to added to the machine PHI node as
      // the input for this MBB.
      MVT::ValueType VT = TLI.getValueType(PN->getType());
      unsigned NumElements;
      if (VT != MVT::Vector)
        NumElements = TLI.getNumElements(VT);
      else {
        MVT::ValueType VT1,VT2;
        NumElements = 
          TLI.getVectorTypeBreakdown(cast<VectorType>(PN->getType()),
                                     VT1, VT2);
      }
      for (unsigned i = 0, e = NumElements; i != e; ++i)
        PHINodesToUpdate.push_back(std::make_pair(MBBI++, Reg+i));
    }
  }
  ConstantsOut.clear();

  // Turn all of the unordered chains into one factored node.
  if (!UnorderedChains.empty()) {
    SDOperand Root = SDL.getRoot();
    if (Root.getOpcode() != ISD::EntryToken) {
      unsigned i = 0, e = UnorderedChains.size();
      for (; i != e; ++i) {
        assert(UnorderedChains[i].Val->getNumOperands() > 1);
        if (UnorderedChains[i].Val->getOperand(0) == Root)
          break;  // Don't add the root if we already indirectly depend on it.
      }
        
      if (i == e)
        UnorderedChains.push_back(Root);
    }
    DAG.setRoot(DAG.getNode(ISD::TokenFactor, MVT::Other,
                            &UnorderedChains[0], UnorderedChains.size()));
  }

  // Lower the terminator after the copies are emitted.
  if (Invoke) {
    // Just the branch part of invoke.
    SDL.visitInvoke(*Invoke, true);
  } else {
    SDL.visit(*LLVMBB->getTerminator());
  }

  // Copy over any CaseBlock records that may now exist due to SwitchInst
  // lowering, as well as any jump table information.
  SwitchCases.clear();
  SwitchCases = SDL.SwitchCases;
  JTCases.clear();
  JTCases = SDL.JTCases;
  BitTestCases.clear();
  BitTestCases = SDL.BitTestCases;
    
  // Make sure the root of the DAG is up-to-date.
  DAG.setRoot(SDL.getRoot());
}

void SelectionDAGISel::CodeGenAndEmitDAG(SelectionDAG &DAG) {
  // Get alias analysis for load/store combining.
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  // Run the DAG combiner in pre-legalize mode.
  DAG.Combine(false, AA);
  
  DOUT << "Lowered selection DAG:\n";
  DEBUG(DAG.dump());
  
  // Second step, hack on the DAG until it only uses operations and types that
  // the target supports.
  DAG.Legalize();
  
  DOUT << "Legalized selection DAG:\n";
  DEBUG(DAG.dump());
  
  // Run the DAG combiner in post-legalize mode.
  DAG.Combine(true, AA);
  
  if (ViewISelDAGs) DAG.viewGraph();

  // Third, instruction select all of the operations to machine code, adding the
  // code to the MachineBasicBlock.
  InstructionSelectBasicBlock(DAG);
  
  DOUT << "Selected machine code:\n";
  DEBUG(BB->dump());
}  

void SelectionDAGISel::SelectBasicBlock(BasicBlock *LLVMBB, MachineFunction &MF,
                                        FunctionLoweringInfo &FuncInfo) {
  std::vector<std::pair<MachineInstr*, unsigned> > PHINodesToUpdate;
  {
    SelectionDAG DAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
    CurDAG = &DAG;
  
    // First step, lower LLVM code to some DAG.  This DAG may use operations and
    // types that are not supported by the target.
    BuildSelectionDAG(DAG, LLVMBB, PHINodesToUpdate, FuncInfo);

    // Second step, emit the lowered DAG as machine code.
    CodeGenAndEmitDAG(DAG);
  }

  DOUT << "Total amount of phi nodes to update: "
       << PHINodesToUpdate.size() << "\n";
  DEBUG(for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i)
          DOUT << "Node " << i << " : (" << PHINodesToUpdate[i].first
               << ", " << PHINodesToUpdate[i].second << ")\n";);
  
  // Next, now that we know what the last MBB the LLVM BB expanded is, update
  // PHI nodes in successors.
  if (SwitchCases.empty() && JTCases.empty() && BitTestCases.empty()) {
    for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i) {
      MachineInstr *PHI = PHINodesToUpdate[i].first;
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      PHI->addRegOperand(PHINodesToUpdate[i].second, false);
      PHI->addMachineBasicBlockOperand(BB);
    }
    return;
  }

  for (unsigned i = 0, e = BitTestCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!BitTestCases[i].Emitted) {
      SelectionDAG HSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
      CurDAG = &HSDAG;
      SelectionDAGLowering HSDL(HSDAG, TLI, FuncInfo);    
      // Set the current basic block to the mbb we wish to insert the code into
      BB = BitTestCases[i].Parent;
      HSDL.setCurrentBasicBlock(BB);
      // Emit the code
      HSDL.visitBitTestHeader(BitTestCases[i]);
      HSDAG.setRoot(HSDL.getRoot());
      CodeGenAndEmitDAG(HSDAG);
    }    

    for (unsigned j = 0, ej = BitTestCases[i].Cases.size(); j != ej; ++j) {
      SelectionDAG BSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
      CurDAG = &BSDAG;
      SelectionDAGLowering BSDL(BSDAG, TLI, FuncInfo);
      // Set the current basic block to the mbb we wish to insert the code into
      BB = BitTestCases[i].Cases[j].ThisBB;
      BSDL.setCurrentBasicBlock(BB);
      // Emit the code
      if (j+1 != ej)
        BSDL.visitBitTestCase(BitTestCases[i].Cases[j+1].ThisBB,
                              BitTestCases[i].Reg,
                              BitTestCases[i].Cases[j]);
      else
        BSDL.visitBitTestCase(BitTestCases[i].Default,
                              BitTestCases[i].Reg,
                              BitTestCases[i].Cases[j]);
        
        
      BSDAG.setRoot(BSDL.getRoot());
      CodeGenAndEmitDAG(BSDAG);
    }

    // Update PHI Nodes
    for (unsigned pi = 0, pe = PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // This is "default" BB. We have two jumps to it. From "header" BB and
      // from last "case" BB.
      if (PHIBB == BitTestCases[i].Default) {
        PHI->addRegOperand(PHINodesToUpdate[pi].second, false);
        PHI->addMachineBasicBlockOperand(BitTestCases[i].Parent);
        PHI->addRegOperand(PHINodesToUpdate[pi].second, false);
        PHI->addMachineBasicBlockOperand(BitTestCases[i].Cases.back().ThisBB);
      }
      // One of "cases" BB.
      for (unsigned j = 0, ej = BitTestCases[i].Cases.size(); j != ej; ++j) {
        MachineBasicBlock* cBB = BitTestCases[i].Cases[j].ThisBB;
        if (cBB->succ_end() !=
            std::find(cBB->succ_begin(),cBB->succ_end(), PHIBB)) {
          PHI->addRegOperand(PHINodesToUpdate[pi].second, false);
          PHI->addMachineBasicBlockOperand(cBB);
        }
      }
    }
  }

  // If the JumpTable record is filled in, then we need to emit a jump table.
  // Updating the PHI nodes is tricky in this case, since we need to determine
  // whether the PHI is a successor of the range check MBB or the jump table MBB
  for (unsigned i = 0, e = JTCases.size(); i != e; ++i) {
    // Lower header first, if it wasn't already lowered
    if (!JTCases[i].first.Emitted) {
      SelectionDAG HSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
      CurDAG = &HSDAG;
      SelectionDAGLowering HSDL(HSDAG, TLI, FuncInfo);    
      // Set the current basic block to the mbb we wish to insert the code into
      BB = JTCases[i].first.HeaderBB;
      HSDL.setCurrentBasicBlock(BB);
      // Emit the code
      HSDL.visitJumpTableHeader(JTCases[i].second, JTCases[i].first);
      HSDAG.setRoot(HSDL.getRoot());
      CodeGenAndEmitDAG(HSDAG);
    }
    
    SelectionDAG JSDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
    CurDAG = &JSDAG;
    SelectionDAGLowering JSDL(JSDAG, TLI, FuncInfo);
    // Set the current basic block to the mbb we wish to insert the code into
    BB = JTCases[i].second.MBB;
    JSDL.setCurrentBasicBlock(BB);
    // Emit the code
    JSDL.visitJumpTable(JTCases[i].second);
    JSDAG.setRoot(JSDL.getRoot());
    CodeGenAndEmitDAG(JSDAG);
    
    // Update PHI Nodes
    for (unsigned pi = 0, pe = PHINodesToUpdate.size(); pi != pe; ++pi) {
      MachineInstr *PHI = PHINodesToUpdate[pi].first;
      MachineBasicBlock *PHIBB = PHI->getParent();
      assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
             "This is not a machine PHI node that we are updating!");
      // "default" BB. We can go there only from header BB.
      if (PHIBB == JTCases[i].second.Default) {
        PHI->addRegOperand(PHINodesToUpdate[pi].second, false);
        PHI->addMachineBasicBlockOperand(JTCases[i].first.HeaderBB);
      }
      // JT BB. Just iterate over successors here
      if (BB->succ_end() != std::find(BB->succ_begin(),BB->succ_end(), PHIBB)) {
        PHI->addRegOperand(PHINodesToUpdate[pi].second, false);
        PHI->addMachineBasicBlockOperand(BB);
      }
    }
  }
  
  // If the switch block involved a branch to one of the actual successors, we
  // need to update PHI nodes in that block.
  for (unsigned i = 0, e = PHINodesToUpdate.size(); i != e; ++i) {
    MachineInstr *PHI = PHINodesToUpdate[i].first;
    assert(PHI->getOpcode() == TargetInstrInfo::PHI &&
           "This is not a machine PHI node that we are updating!");
    if (BB->isSuccessor(PHI->getParent())) {
      PHI->addRegOperand(PHINodesToUpdate[i].second, false);
      PHI->addMachineBasicBlockOperand(BB);
    }
  }
  
  // If we generated any switch lowering information, build and codegen any
  // additional DAGs necessary.
  for (unsigned i = 0, e = SwitchCases.size(); i != e; ++i) {
    SelectionDAG SDAG(TLI, MF, getAnalysisToUpdate<MachineModuleInfo>());
    CurDAG = &SDAG;
    SelectionDAGLowering SDL(SDAG, TLI, FuncInfo);
    
    // Set the current basic block to the mbb we wish to insert the code into
    BB = SwitchCases[i].ThisBB;
    SDL.setCurrentBasicBlock(BB);
    
    // Emit the code
    SDL.visitSwitchCase(SwitchCases[i]);
    SDAG.setRoot(SDL.getRoot());
    CodeGenAndEmitDAG(SDAG);
    
    // Handle any PHI nodes in successors of this chunk, as if we were coming
    // from the original BB before switch expansion.  Note that PHI nodes can
    // occur multiple times in PHINodesToUpdate.  We have to be very careful to
    // handle them the right number of times.
    while ((BB = SwitchCases[i].TrueBB)) {  // Handle LHS and RHS.
      for (MachineBasicBlock::iterator Phi = BB->begin();
           Phi != BB->end() && Phi->getOpcode() == TargetInstrInfo::PHI; ++Phi){
        // This value for this PHI node is recorded in PHINodesToUpdate, get it.
        for (unsigned pn = 0; ; ++pn) {
          assert(pn != PHINodesToUpdate.size() && "Didn't find PHI entry!");
          if (PHINodesToUpdate[pn].first == Phi) {
            Phi->addRegOperand(PHINodesToUpdate[pn].second, false);
            Phi->addMachineBasicBlockOperand(SwitchCases[i].ThisBB);
            break;
          }
        }
      }
      
      // Don't process RHS if same block as LHS.
      if (BB == SwitchCases[i].FalseBB)
        SwitchCases[i].FalseBB = 0;
      
      // If we haven't handled the RHS, do so now.  Otherwise, we're done.
      SwitchCases[i].TrueBB = SwitchCases[i].FalseBB;
      SwitchCases[i].FalseBB = 0;
    }
    assert(SwitchCases[i].TrueBB == 0 && SwitchCases[i].FalseBB == 0);
  }
}


//===----------------------------------------------------------------------===//
/// ScheduleAndEmitDAG - Pick a safe ordering and emit instructions for each
/// target node in the graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &DAG) {
  if (ViewSchedDAGs) DAG.viewGraph();

  RegisterScheduler::FunctionPassCtor Ctor = RegisterScheduler::getDefault();
  
  if (!Ctor) {
    Ctor = ISHeuristic;
    RegisterScheduler::setDefault(Ctor);
  }
  
  ScheduleDAG *SL = Ctor(this, &DAG, BB);
  BB = SL->Run();
  delete SL;
}


HazardRecognizer *SelectionDAGISel::CreateTargetHazardRecognizer() {
  return new HazardRecognizer();
}

//===----------------------------------------------------------------------===//
// Helper functions used by the generated instruction selector.
//===----------------------------------------------------------------------===//
// Calls to these methods are generated by tblgen.

/// CheckAndMask - The isel is trying to match something like (and X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an AND, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckAndMask(SDOperand LHS, ConstantSDNode *RHS, 
                                    int64_t DesiredMaskS) {
  uint64_t ActualMask = RHS->getValue();
  uint64_t DesiredMask =DesiredMaskS & MVT::getIntVTBitMask(LHS.getValueType());
  
  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;
  
  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask & ~DesiredMask)
    return false;
  
  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  uint64_t NeededMask = DesiredMask & ~ActualMask;
  if (getTargetLowering().MaskedValueIsZero(LHS, NeededMask))
    return true;
  
  // TODO: check to see if missing bits are just not demanded.

  // Otherwise, this pattern doesn't match.
  return false;
}

/// CheckOrMask - The isel is trying to match something like (or X, 255).  If
/// the dag combiner simplified the 255, we still want to match.  RHS is the
/// actual value in the DAG on the RHS of an OR, and DesiredMaskS is the value
/// specified in the .td file (e.g. 255).
bool SelectionDAGISel::CheckOrMask(SDOperand LHS, ConstantSDNode *RHS, 
                                    int64_t DesiredMaskS) {
  uint64_t ActualMask = RHS->getValue();
  uint64_t DesiredMask =DesiredMaskS & MVT::getIntVTBitMask(LHS.getValueType());
  
  // If the actual mask exactly matches, success!
  if (ActualMask == DesiredMask)
    return true;
  
  // If the actual AND mask is allowing unallowed bits, this doesn't match.
  if (ActualMask & ~DesiredMask)
    return false;
  
  // Otherwise, the DAG Combiner may have proven that the value coming in is
  // either already zero or is not demanded.  Check for known zero input bits.
  uint64_t NeededMask = DesiredMask & ~ActualMask;
  
  uint64_t KnownZero, KnownOne;
  getTargetLowering().ComputeMaskedBits(LHS, NeededMask, KnownZero, KnownOne);
  
  // If all the missing bits in the or are already known to be set, match!
  if ((NeededMask & KnownOne) == NeededMask)
    return true;
  
  // TODO: check to see if missing bits are just not demanded.
  
  // Otherwise, this pattern doesn't match.
  return false;
}


/// SelectInlineAsmMemoryOperands - Calls to this are automatically generated
/// by tblgen.  Others should not call it.
void SelectionDAGISel::
SelectInlineAsmMemoryOperands(std::vector<SDOperand> &Ops, SelectionDAG &DAG) {
  std::vector<SDOperand> InOps;
  std::swap(InOps, Ops);

  Ops.push_back(InOps[0]);  // input chain.
  Ops.push_back(InOps[1]);  // input asm string.

  unsigned i = 2, e = InOps.size();
  if (InOps[e-1].getValueType() == MVT::Flag)
    --e;  // Don't process a flag operand if it is here.
  
  while (i != e) {
    unsigned Flags = cast<ConstantSDNode>(InOps[i])->getValue();
    if ((Flags & 7) != 4 /*MEM*/) {
      // Just skip over this operand, copying the operands verbatim.
      Ops.insert(Ops.end(), InOps.begin()+i, InOps.begin()+i+(Flags >> 3) + 1);
      i += (Flags >> 3) + 1;
    } else {
      assert((Flags >> 3) == 1 && "Memory operand with multiple values?");
      // Otherwise, this is a memory operand.  Ask the target to select it.
      std::vector<SDOperand> SelOps;
      if (SelectInlineAsmMemoryOperand(InOps[i+1], 'm', SelOps, DAG)) {
        cerr << "Could not match memory address.  Inline asm failure!\n";
        exit(1);
      }
      
      // Add this to the output node.
      MVT::ValueType IntPtrTy = DAG.getTargetLoweringInfo().getPointerTy();
      Ops.push_back(DAG.getTargetConstant(4/*MEM*/ | (SelOps.size() << 3),
                                          IntPtrTy));
      Ops.insert(Ops.end(), SelOps.begin(), SelOps.end());
      i += 2;
    }
  }
  
  // Add the flag input back if present.
  if (e != InOps.size())
    Ops.push_back(InOps.back());
}

char SelectionDAGISel::ID = 0;
