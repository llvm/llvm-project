//===-- InstSelectSimple.cpp - A simple instruction selector for x86 ------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a simple peephole instruction selector for the x86 target
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrBuilder.h"
#include "X86InstrInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/IntrinsicLowering.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/MRegisterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/GetElementPtrTypeIterator.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/CFG.h"
#include "Support/Statistic.h"
using namespace llvm;

namespace {
  Statistic<>
  NumFPKill("x86-codegen", "Number of FP_REG_KILL instructions added");
}

namespace {
  struct ISel : public FunctionPass, InstVisitor<ISel> {
    TargetMachine &TM;
    MachineFunction *F;                 // The function we are compiling into
    MachineBasicBlock *BB;              // The current MBB we are compiling
    int VarArgsFrameIndex;              // FrameIndex for start of varargs area
    int ReturnAddressIndex;             // FrameIndex for the return address

    std::map<Value*, unsigned> RegMap;  // Mapping between Val's and SSA Regs

    // MBBMap - Mapping between LLVM BB -> Machine BB
    std::map<const BasicBlock*, MachineBasicBlock*> MBBMap;

    ISel(TargetMachine &tm) : TM(tm), F(0), BB(0) {}

    /// runOnFunction - Top level implementation of instruction selection for
    /// the entire function.
    ///
    bool runOnFunction(Function &Fn) {
      // First pass over the function, lower any unknown intrinsic functions
      // with the IntrinsicLowering class.
      LowerUnknownIntrinsicFunctionCalls(Fn);

      F = &MachineFunction::construct(&Fn, TM);

      // Create all of the machine basic blocks for the function...
      for (Function::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I)
        F->getBasicBlockList().push_back(MBBMap[I] = new MachineBasicBlock(I));

      BB = &F->front();

      // Set up a frame object for the return address.  This is used by the
      // llvm.returnaddress & llvm.frameaddress intrinisics.
      ReturnAddressIndex = F->getFrameInfo()->CreateFixedObject(4, -4);

      // Copy incoming arguments off of the stack...
      LoadArgumentsToVirtualRegs(Fn);

      // Instruction select everything except PHI nodes
      visit(Fn);

      // Select the PHI nodes
      SelectPHINodes();

      // Insert the FP_REG_KILL instructions into blocks that need them.
      InsertFPRegKills();

      RegMap.clear();
      MBBMap.clear();
      F = 0;
      // We always build a machine code representation for the function
      return true;
    }

    virtual const char *getPassName() const {
      return "X86 Simple Instruction Selection";
    }

    /// visitBasicBlock - This method is called when we are visiting a new basic
    /// block.  This simply creates a new MachineBasicBlock to emit code into
    /// and adds it to the current MachineFunction.  Subsequent visit* for
    /// instructions will be invoked for all instructions in the basic block.
    ///
    void visitBasicBlock(BasicBlock &LLVM_BB) {
      BB = MBBMap[&LLVM_BB];
    }

    /// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
    /// function, lowering any calls to unknown intrinsic functions into the
    /// equivalent LLVM code.
    ///
    void LowerUnknownIntrinsicFunctionCalls(Function &F);

    /// LoadArgumentsToVirtualRegs - Load all of the arguments to this function
    /// from the stack into virtual registers.
    ///
    void LoadArgumentsToVirtualRegs(Function &F);

    /// SelectPHINodes - Insert machine code to generate phis.  This is tricky
    /// because we have to generate our sources into the source basic blocks,
    /// not the current one.
    ///
    void SelectPHINodes();

    /// InsertFPRegKills - Insert FP_REG_KILL instructions into basic blocks
    /// that need them.  This only occurs due to the floating point stackifier
    /// not being aggressive enough to handle arbitrary global stackification.
    ///
    void InsertFPRegKills();

    // Visitation methods for various instructions.  These methods simply emit
    // fixed X86 code for each instruction.
    //

    // Control flow operators
    void visitReturnInst(ReturnInst &RI);
    void visitBranchInst(BranchInst &BI);

    struct ValueRecord {
      Value *Val;
      unsigned Reg;
      const Type *Ty;
      ValueRecord(unsigned R, const Type *T) : Val(0), Reg(R), Ty(T) {}
      ValueRecord(Value *V) : Val(V), Reg(0), Ty(V->getType()) {}
    };
    void doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                const std::vector<ValueRecord> &Args);
    void visitCallInst(CallInst &I);
    void visitIntrinsicCall(Intrinsic::ID ID, CallInst &I);

    // Arithmetic operators
    void visitSimpleBinary(BinaryOperator &B, unsigned OpcodeClass);
    void visitAdd(BinaryOperator &B) { visitSimpleBinary(B, 0); }
    void visitSub(BinaryOperator &B) { visitSimpleBinary(B, 1); }
    void doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, const Type *DestTy,
                    unsigned Op0Reg, unsigned Op1Reg);
    void doMultiplyConst(MachineBasicBlock *MBB, 
                         MachineBasicBlock::iterator MBBI,
                         unsigned DestReg, const Type *DestTy,
                         unsigned Op0Reg, unsigned Op1Val);
    void visitMul(BinaryOperator &B);

    void visitDiv(BinaryOperator &B) { visitDivRem(B); }
    void visitRem(BinaryOperator &B) { visitDivRem(B); }
    void visitDivRem(BinaryOperator &B);

    // Bitwise operators
    void visitAnd(BinaryOperator &B) { visitSimpleBinary(B, 2); }
    void visitOr (BinaryOperator &B) { visitSimpleBinary(B, 3); }
    void visitXor(BinaryOperator &B) { visitSimpleBinary(B, 4); }

    // Comparison operators...
    void visitSetCondInst(SetCondInst &I);
    unsigned EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                            MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator MBBI);
    
    // Memory Instructions
    void visitLoadInst(LoadInst &I);
    void visitStoreInst(StoreInst &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitMallocInst(MallocInst &I);
    void visitFreeInst(FreeInst &I);
    
    // Other operators
    void visitShiftInst(ShiftInst &I);
    void visitPHINode(PHINode &I) {}      // PHI nodes handled by second pass
    void visitCastInst(CastInst &I);
    void visitVANextInst(VANextInst &I);
    void visitVAArgInst(VAArgInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "Cannot instruction select: " << I;
      abort();
    }

    /// promote32 - Make a value 32-bits wide, and put it somewhere.
    ///
    void promote32(unsigned targetReg, const ValueRecord &VR);

    /// getAddressingMode - Get the addressing mode to use to address the
    /// specified value.  The returned value should be used with addFullAddress.
    void getAddressingMode(Value *Addr, unsigned &BaseReg, unsigned &Scale,
                           unsigned &IndexReg, unsigned &Disp);


    /// getGEPIndex - This is used to fold GEP instructions into X86 addressing
    /// expressions.
    void getGEPIndex(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                     std::vector<Value*> &GEPOps,
                     std::vector<const Type*> &GEPTypes, unsigned &BaseReg,
                     unsigned &Scale, unsigned &IndexReg, unsigned &Disp);

    /// isGEPFoldable - Return true if the specified GEP can be completely
    /// folded into the addressing mode of a load/store or lea instruction.
    bool isGEPFoldable(MachineBasicBlock *MBB,
                       Value *Src, User::op_iterator IdxBegin,
                       User::op_iterator IdxEnd, unsigned &BaseReg,
                       unsigned &Scale, unsigned &IndexReg, unsigned &Disp);

    /// emitGEPOperation - Common code shared between visitGetElementPtrInst and
    /// constant expression GEP support.
    ///
    void emitGEPOperation(MachineBasicBlock *BB, MachineBasicBlock::iterator IP,
                          Value *Src, User::op_iterator IdxBegin,
                          User::op_iterator IdxEnd, unsigned TargetReg);

    /// emitCastOperation - Common code shared between visitCastInst and
    /// constant expression cast support.
    ///
    void emitCastOperation(MachineBasicBlock *BB,MachineBasicBlock::iterator IP,
                           Value *Src, const Type *DestTy, unsigned TargetReg);

    /// emitSimpleBinaryOperation - Common code shared between visitSimpleBinary
    /// and constant expression support.
    ///
    void emitSimpleBinaryOperation(MachineBasicBlock *BB,
                                   MachineBasicBlock::iterator IP,
                                   Value *Op0, Value *Op1,
                                   unsigned OperatorClass, unsigned TargetReg);

    void emitDivRemOperation(MachineBasicBlock *BB,
                             MachineBasicBlock::iterator IP,
                             unsigned Op0Reg, unsigned Op1Reg, bool isDiv,
                             const Type *Ty, unsigned TargetReg);

    /// emitSetCCOperation - Common code shared between visitSetCondInst and
    /// constant expression support.
    ///
    void emitSetCCOperation(MachineBasicBlock *BB,
                            MachineBasicBlock::iterator IP,
                            Value *Op0, Value *Op1, unsigned Opcode,
                            unsigned TargetReg);

    /// emitShiftOperation - Common code shared between visitShiftInst and
    /// constant expression support.
    ///
    void emitShiftOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator IP,
                            Value *Op, Value *ShiftAmount, bool isLeftShift,
                            const Type *ResultTy, unsigned DestReg);
      

    /// copyConstantToRegister - Output the instructions required to put the
    /// specified constant into the specified register.
    ///
    void copyConstantToRegister(MachineBasicBlock *MBB,
                                MachineBasicBlock::iterator MBBI,
                                Constant *C, unsigned Reg);

    /// makeAnotherReg - This method returns the next register number we haven't
    /// yet used.
    ///
    /// Long values are handled somewhat specially.  They are always allocated
    /// as pairs of 32 bit integer values.  The register number returned is the
    /// lower 32 bits of the long value, and the regNum+1 is the upper 32 bits
    /// of the long value.
    ///
    unsigned makeAnotherReg(const Type *Ty) {
      assert(dynamic_cast<const X86RegisterInfo*>(TM.getRegisterInfo()) &&
             "Current target doesn't have X86 reg info??");
      const X86RegisterInfo *MRI =
        static_cast<const X86RegisterInfo*>(TM.getRegisterInfo());
      if (Ty == Type::LongTy || Ty == Type::ULongTy) {
        const TargetRegisterClass *RC = MRI->getRegClassForType(Type::IntTy);
        // Create the lower part
        F->getSSARegMap()->createVirtualRegister(RC);
        // Create the upper part.
        return F->getSSARegMap()->createVirtualRegister(RC)-1;
      }

      // Add the mapping of regnumber => reg class to MachineFunction
      const TargetRegisterClass *RC = MRI->getRegClassForType(Ty);
      return F->getSSARegMap()->createVirtualRegister(RC);
    }

    /// getReg - This method turns an LLVM value into a register number.  This
    /// is guaranteed to produce the same register number for a particular value
    /// every time it is queried.
    ///
    unsigned getReg(Value &V) { return getReg(&V); }  // Allow references
    unsigned getReg(Value *V) {
      // Just append to the end of the current bb.
      MachineBasicBlock::iterator It = BB->end();
      return getReg(V, BB, It);
    }
    unsigned getReg(Value *V, MachineBasicBlock *MBB,
                    MachineBasicBlock::iterator IPt) {
      unsigned &Reg = RegMap[V];
      if (Reg == 0) {
        Reg = makeAnotherReg(V->getType());
        RegMap[V] = Reg;
      }

      // If this operand is a constant, emit the code to copy the constant into
      // the register here...
      //
      if (Constant *C = dyn_cast<Constant>(V)) {
        copyConstantToRegister(MBB, IPt, C, Reg);
        RegMap.erase(V);  // Assign a new name to this constant if ref'd again
      } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
        // Move the address of the global into the register
        BuildMI(*MBB, IPt, X86::MOV32ri, 1, Reg).addGlobalAddress(GV);
        RegMap.erase(V);  // Assign a new name to this address if ref'd again
      }

      return Reg;
    }
  };
}

/// TypeClass - Used by the X86 backend to group LLVM types by their basic X86
/// Representation.
///
enum TypeClass {
  cByte, cShort, cInt, cFP, cLong
};

/// getClass - Turn a primitive type into a "class" number which is based on the
/// size of the type, and whether or not it is floating point.
///
static inline TypeClass getClass(const Type *Ty) {
  switch (Ty->getPrimitiveID()) {
  case Type::SByteTyID:
  case Type::UByteTyID:   return cByte;      // Byte operands are class #0
  case Type::ShortTyID:
  case Type::UShortTyID:  return cShort;     // Short operands are class #1
  case Type::IntTyID:
  case Type::UIntTyID:
  case Type::PointerTyID: return cInt;       // Int's and pointers are class #2

  case Type::FloatTyID:
  case Type::DoubleTyID:  return cFP;        // Floating Point is #3

  case Type::LongTyID:
  case Type::ULongTyID:   return cLong;      // Longs are class #4
  default:
    assert(0 && "Invalid type to getClass!");
    return cByte;  // not reached
  }
}

// getClassB - Just like getClass, but treat boolean values as bytes.
static inline TypeClass getClassB(const Type *Ty) {
  if (Ty == Type::BoolTy) return cByte;
  return getClass(Ty);
}


/// copyConstantToRegister - Output the instructions required to put the
/// specified constant into the specified register.
///
void ISel::copyConstantToRegister(MachineBasicBlock *MBB,
                                  MachineBasicBlock::iterator IP,
                                  Constant *C, unsigned R) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    unsigned Class = 0;
    switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      emitGEPOperation(MBB, IP, CE->getOperand(0),
                       CE->op_begin()+1, CE->op_end(), R);
      return;
    case Instruction::Cast:
      emitCastOperation(MBB, IP, CE->getOperand(0), CE->getType(), R);
      return;

    case Instruction::Xor: ++Class; // FALL THROUGH
    case Instruction::Or:  ++Class; // FALL THROUGH
    case Instruction::And: ++Class; // FALL THROUGH
    case Instruction::Sub: ++Class; // FALL THROUGH
    case Instruction::Add:
      emitSimpleBinaryOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                                Class, R);
      return;

    case Instruction::Mul: {
      unsigned Op0Reg = getReg(CE->getOperand(0), MBB, IP);
      unsigned Op1Reg = getReg(CE->getOperand(1), MBB, IP);
      doMultiply(MBB, IP, R, CE->getType(), Op0Reg, Op1Reg);
      return;
    }
    case Instruction::Div:
    case Instruction::Rem: {
      unsigned Op0Reg = getReg(CE->getOperand(0), MBB, IP);
      unsigned Op1Reg = getReg(CE->getOperand(1), MBB, IP);
      emitDivRemOperation(MBB, IP, Op0Reg, Op1Reg,
                          CE->getOpcode() == Instruction::Div,
                          CE->getType(), R);
      return;
    }

    case Instruction::SetNE:
    case Instruction::SetEQ:
    case Instruction::SetLT:
    case Instruction::SetGT:
    case Instruction::SetLE:
    case Instruction::SetGE:
      emitSetCCOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                         CE->getOpcode(), R);
      return;

    case Instruction::Shl:
    case Instruction::Shr:
      emitShiftOperation(MBB, IP, CE->getOperand(0), CE->getOperand(1),
                         CE->getOpcode() == Instruction::Shl, CE->getType(), R);
      return;

    default:
      std::cerr << "Offending expr: " << C << "\n";
      assert(0 && "Constant expression not yet handled!\n");
    }
  }

  if (C->getType()->isIntegral()) {
    unsigned Class = getClassB(C->getType());

    if (Class == cLong) {
      // Copy the value into the register pair.
      uint64_t Val = cast<ConstantInt>(C)->getRawValue();
      BuildMI(*MBB, IP, X86::MOV32ri, 1, R).addImm(Val & 0xFFFFFFFF);
      BuildMI(*MBB, IP, X86::MOV32ri, 1, R+1).addImm(Val >> 32);
      return;
    }

    assert(Class <= cInt && "Type not handled yet!");

    static const unsigned IntegralOpcodeTab[] = {
      X86::MOV8ri, X86::MOV16ri, X86::MOV32ri
    };

    if (C->getType() == Type::BoolTy) {
      BuildMI(*MBB, IP, X86::MOV8ri, 1, R).addImm(C == ConstantBool::True);
    } else {
      ConstantInt *CI = cast<ConstantInt>(C);
      BuildMI(*MBB, IP, IntegralOpcodeTab[Class],1,R).addImm(CI->getRawValue());
    }
  } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
    if (CFP->isExactlyValue(+0.0))
      BuildMI(*MBB, IP, X86::FLD0, 0, R);
    else if (CFP->isExactlyValue(+1.0))
      BuildMI(*MBB, IP, X86::FLD1, 0, R);
    else {
      // Otherwise we need to spill the constant to memory...
      MachineConstantPool *CP = F->getConstantPool();
      unsigned CPI = CP->getConstantPoolIndex(CFP);
      const Type *Ty = CFP->getType();

      assert(Ty == Type::FloatTy || Ty == Type::DoubleTy && "Unknown FP type!");
      unsigned LoadOpcode = Ty == Type::FloatTy ? X86::FLD32m : X86::FLD64m;
      addConstantPoolReference(BuildMI(*MBB, IP, LoadOpcode, 4, R), CPI);
    }

  } else if (isa<ConstantPointerNull>(C)) {
    // Copy zero (null pointer) to the register.
    BuildMI(*MBB, IP, X86::MOV32ri, 1, R).addImm(0);
  } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(C)) {
    BuildMI(*MBB, IP, X86::MOV32ri, 1, R).addGlobalAddress(CPR->getValue());
  } else {
    std::cerr << "Offending constant: " << C << "\n";
    assert(0 && "Type not handled yet!");
  }
}

/// LoadArgumentsToVirtualRegs - Load all of the arguments to this function from
/// the stack into virtual registers.
///
void ISel::LoadArgumentsToVirtualRegs(Function &Fn) {
  // Emit instructions to load the arguments...  On entry to a function on the
  // X86, the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically)
  // [ESP + 8] -- second argument, if first argument is four bytes in size
  //    ... 
  //
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  MachineFrameInfo *MFI = F->getFrameInfo();

  for (Function::aiterator I = Fn.abegin(), E = Fn.aend(); I != E; ++I) {
    unsigned Reg = getReg(*I);
    
    int FI;          // Frame object index
    switch (getClassB(I->getType())) {
    case cByte:
      FI = MFI->CreateFixedObject(1, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOV8rm, 4, Reg), FI);
      break;
    case cShort:
      FI = MFI->CreateFixedObject(2, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOV16rm, 4, Reg), FI);
      break;
    case cInt:
      FI = MFI->CreateFixedObject(4, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOV32rm, 4, Reg), FI);
      break;
    case cLong:
      FI = MFI->CreateFixedObject(8, ArgOffset);
      addFrameReference(BuildMI(BB, X86::MOV32rm, 4, Reg), FI);
      addFrameReference(BuildMI(BB, X86::MOV32rm, 4, Reg+1), FI, 4);
      ArgOffset += 4;   // longs require 4 additional bytes
      break;
    case cFP:
      unsigned Opcode;
      if (I->getType() == Type::FloatTy) {
        Opcode = X86::FLD32m;
        FI = MFI->CreateFixedObject(4, ArgOffset);
      } else {
        Opcode = X86::FLD64m;
        FI = MFI->CreateFixedObject(8, ArgOffset);
        ArgOffset += 4;   // doubles require 4 additional bytes
      }
      addFrameReference(BuildMI(BB, Opcode, 4, Reg), FI);
      break;
    default:
      assert(0 && "Unhandled argument type!");
    }
    ArgOffset += 4;  // Each argument takes at least 4 bytes on the stack...
  }

  // If the function takes variable number of arguments, add a frame offset for
  // the start of the first vararg value... this is used to expand
  // llvm.va_start.
  if (Fn.getFunctionType()->isVarArg())
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
}


/// SelectPHINodes - Insert machine code to generate phis.  This is tricky
/// because we have to generate our sources into the source basic blocks, not
/// the current one.
///
void ISel::SelectPHINodes() {
  const TargetInstrInfo &TII = TM.getInstrInfo();
  const Function &LF = *F->getFunction();  // The LLVM function...
  for (Function::const_iterator I = LF.begin(), E = LF.end(); I != E; ++I) {
    const BasicBlock *BB = I;
    MachineBasicBlock &MBB = *MBBMap[I];

    // Loop over all of the PHI nodes in the LLVM basic block...
    MachineBasicBlock::iterator PHIInsertPoint = MBB.begin();
    for (BasicBlock::const_iterator I = BB->begin();
         PHINode *PN = const_cast<PHINode*>(dyn_cast<PHINode>(I)); ++I) {

      // Create a new machine instr PHI node, and insert it.
      unsigned PHIReg = getReg(*PN);
      MachineInstr *PhiMI = BuildMI(MBB, PHIInsertPoint,
                                    X86::PHI, PN->getNumOperands(), PHIReg);

      MachineInstr *LongPhiMI = 0;
      if (PN->getType() == Type::LongTy || PN->getType() == Type::ULongTy)
        LongPhiMI = BuildMI(MBB, PHIInsertPoint,
                            X86::PHI, PN->getNumOperands(), PHIReg+1);

      // PHIValues - Map of blocks to incoming virtual registers.  We use this
      // so that we only initialize one incoming value for a particular block,
      // even if the block has multiple entries in the PHI node.
      //
      std::map<MachineBasicBlock*, unsigned> PHIValues;

      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        MachineBasicBlock *PredMBB = MBBMap[PN->getIncomingBlock(i)];
        unsigned ValReg;
        std::map<MachineBasicBlock*, unsigned>::iterator EntryIt =
          PHIValues.lower_bound(PredMBB);

        if (EntryIt != PHIValues.end() && EntryIt->first == PredMBB) {
          // We already inserted an initialization of the register for this
          // predecessor.  Recycle it.
          ValReg = EntryIt->second;

        } else {        
          // Get the incoming value into a virtual register.
          //
          Value *Val = PN->getIncomingValue(i);

          // If this is a constant or GlobalValue, we may have to insert code
          // into the basic block to compute it into a virtual register.
          if (isa<Constant>(Val) || isa<GlobalValue>(Val)) {
            // Because we don't want to clobber any values which might be in
            // physical registers with the computation of this constant (which
            // might be arbitrarily complex if it is a constant expression),
            // just insert the computation at the top of the basic block.
            MachineBasicBlock::iterator PI = PredMBB->begin();

            // Skip over any PHI nodes though!
            while (PI != PredMBB->end() && PI->getOpcode() == X86::PHI)
              ++PI;

            ValReg = getReg(Val, PredMBB, PI);
          } else {
            ValReg = getReg(Val);
          }

          // Remember that we inserted a value for this PHI for this predecessor
          PHIValues.insert(EntryIt, std::make_pair(PredMBB, ValReg));
        }

        PhiMI->addRegOperand(ValReg);
        PhiMI->addMachineBasicBlockOperand(PredMBB);
        if (LongPhiMI) {
          LongPhiMI->addRegOperand(ValReg+1);
          LongPhiMI->addMachineBasicBlockOperand(PredMBB);
        }
      }

      // Now that we emitted all of the incoming values for the PHI node, make
      // sure to reposition the InsertPoint after the PHI that we just added.
      // This is needed because we might have inserted a constant into this
      // block, right after the PHI's which is before the old insert point!
      PHIInsertPoint = LongPhiMI ? LongPhiMI : PhiMI;
      ++PHIInsertPoint;
    }
  }
}

/// RequiresFPRegKill - The floating point stackifier pass cannot insert
/// compensation code on critical edges.  As such, it requires that we kill all
/// FP registers on the exit from any blocks that either ARE critical edges, or
/// branch to a block that has incoming critical edges.
///
/// Note that this kill instruction will eventually be eliminated when
/// restrictions in the stackifier are relaxed.
///
static bool RequiresFPRegKill(const BasicBlock *BB) {
#if 0
  for (succ_const_iterator SI = succ_begin(BB), E = succ_end(BB); SI!=E; ++SI) {
    const BasicBlock *Succ = *SI;
    pred_const_iterator PI = pred_begin(Succ), PE = pred_end(Succ);
    ++PI;  // Block have at least one predecessory
    if (PI != PE) {             // If it has exactly one, this isn't crit edge
      // If this block has more than one predecessor, check all of the
      // predecessors to see if they have multiple successors.  If so, then the
      // block we are analyzing needs an FPRegKill.
      for (PI = pred_begin(Succ); PI != PE; ++PI) {
        const BasicBlock *Pred = *PI;
        succ_const_iterator SI2 = succ_begin(Pred);
        ++SI2;  // There must be at least one successor of this block.
        if (SI2 != succ_end(Pred))
          return true;   // Yes, we must insert the kill on this edge.
      }
    }
  }
  // If we got this far, there is no need to insert the kill instruction.
  return false;
#else
  return true;
#endif
}

// InsertFPRegKills - Insert FP_REG_KILL instructions into basic blocks that
// need them.  This only occurs due to the floating point stackifier not being
// aggressive enough to handle arbitrary global stackification.
//
// Currently we insert an FP_REG_KILL instruction into each block that uses or
// defines a floating point virtual register.
//
// When the global register allocators (like linear scan) finally update live
// variable analysis, we can keep floating point values in registers across
// portions of the CFG that do not involve critical edges.  This will be a big
// win, but we are waiting on the global allocators before we can do this.
//
// With a bit of work, the floating point stackifier pass can be enhanced to
// break critical edges as needed (to make a place to put compensation code),
// but this will require some infrastructure improvements as well.
//
void ISel::InsertFPRegKills() {
  SSARegMap &RegMap = *F->getSSARegMap();

  for (MachineFunction::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    for (MachineBasicBlock::iterator I = BB->begin(), E = BB->end(); I!=E; ++I)
      for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      MachineOperand& MO = I->getOperand(i);
        if (MO.isRegister() && MO.getReg()) {
          unsigned Reg = MO.getReg();
          if (MRegisterInfo::isVirtualRegister(Reg))
            if (RegMap.getRegClass(Reg)->getSize() == 10)
              goto UsesFPReg;
        }
      }
    // If we haven't found an FP register use or def in this basic block, check
    // to see if any of our successors has an FP PHI node, which will cause a
    // copy to be inserted into this block.
    for (succ_const_iterator SI = succ_begin(BB->getBasicBlock()),
           E = succ_end(BB->getBasicBlock()); SI != E; ++SI) {
      MachineBasicBlock *SBB = MBBMap[*SI];
      for (MachineBasicBlock::iterator I = SBB->begin();
           I != SBB->end() && I->getOpcode() == X86::PHI; ++I) {
        if (RegMap.getRegClass(I->getOperand(0).getReg())->getSize() == 10)
          goto UsesFPReg;
      }
    }
    continue;
  UsesFPReg:
    // Okay, this block uses an FP register.  If the block has successors (ie,
    // it's not an unwind/return), insert the FP_REG_KILL instruction.
    if (BB->getBasicBlock()->getTerminator()->getNumSuccessors() &&
        RequiresFPRegKill(BB->getBasicBlock())) {
      BuildMI(*BB, BB->getFirstTerminator(), X86::FP_REG_KILL, 0);
      ++NumFPKill;
    }
  }
}


// canFoldSetCCIntoBranch - Return the setcc instruction if we can fold it into
// the conditional branch instruction which is the only user of the cc
// instruction.  This is the case if the conditional branch is the only user of
// the setcc, and if the setcc is in the same basic block as the conditional
// branch.  We also don't handle long arguments below, so we reject them here as
// well.
//
static SetCondInst *canFoldSetCCIntoBranch(Value *V) {
  if (SetCondInst *SCI = dyn_cast<SetCondInst>(V))
    if (SCI->hasOneUse() && isa<BranchInst>(SCI->use_back()) &&
        SCI->getParent() == cast<BranchInst>(SCI->use_back())->getParent()) {
      const Type *Ty = SCI->getOperand(0)->getType();
      if (Ty != Type::LongTy && Ty != Type::ULongTy)
        return SCI;
    }
  return 0;
}

// Return a fixed numbering for setcc instructions which does not depend on the
// order of the opcodes.
//
static unsigned getSetCCNumber(unsigned Opcode) {
  switch(Opcode) {
  default: assert(0 && "Unknown setcc instruction!");
  case Instruction::SetEQ: return 0;
  case Instruction::SetNE: return 1;
  case Instruction::SetLT: return 2;
  case Instruction::SetGE: return 3;
  case Instruction::SetGT: return 4;
  case Instruction::SetLE: return 5;
  }
}

// LLVM  -> X86 signed  X86 unsigned
// -----    ----------  ------------
// seteq -> sete        sete
// setne -> setne       setne
// setlt -> setl        setb
// setge -> setge       setae
// setgt -> setg        seta
// setle -> setle       setbe
// ----
//          sets                       // Used by comparison with 0 optimization
//          setns
static const unsigned SetCCOpcodeTab[2][8] = {
  { X86::SETEr, X86::SETNEr, X86::SETBr, X86::SETAEr, X86::SETAr, X86::SETBEr,
    0, 0 },
  { X86::SETEr, X86::SETNEr, X86::SETLr, X86::SETGEr, X86::SETGr, X86::SETLEr,
    X86::SETSr, X86::SETNSr },
};

// EmitComparison - This function emits a comparison of the two operands,
// returning the extended setcc code to use.
unsigned ISel::EmitComparison(unsigned OpNum, Value *Op0, Value *Op1,
                              MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP) {
  // The arguments are already supposed to be of the same type.
  const Type *CompTy = Op0->getType();
  unsigned Class = getClassB(CompTy);
  unsigned Op0r = getReg(Op0, MBB, IP);

  // Special case handling of: cmp R, i
  if (Class == cByte || Class == cShort || Class == cInt)
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op1)) {
      uint64_t Op1v = cast<ConstantInt>(CI)->getRawValue();

      // Mask off any upper bits of the constant, if there are any...
      Op1v &= (1ULL << (8 << Class)) - 1;

      // If this is a comparison against zero, emit more efficient code.  We
      // can't handle unsigned comparisons against zero unless they are == or
      // !=.  These should have been strength reduced already anyway.
      if (Op1v == 0 && (CompTy->isSigned() || OpNum < 2)) {
        static const unsigned TESTTab[] = {
          X86::TEST8rr, X86::TEST16rr, X86::TEST32rr
        };
        BuildMI(*MBB, IP, TESTTab[Class], 2).addReg(Op0r).addReg(Op0r);

        if (OpNum == 2) return 6;   // Map jl -> js
        if (OpNum == 3) return 7;   // Map jg -> jns
        return OpNum;
      }

      static const unsigned CMPTab[] = {
        X86::CMP8ri, X86::CMP16ri, X86::CMP32ri
      };

      BuildMI(*MBB, IP, CMPTab[Class], 2).addReg(Op0r).addImm(Op1v);
      return OpNum;
    }

  // Special case handling of comparison against +/- 0.0
  if (ConstantFP *CFP = dyn_cast<ConstantFP>(Op1))
    if (CFP->isExactlyValue(+0.0) || CFP->isExactlyValue(-0.0)) {
      BuildMI(*MBB, IP, X86::FTST, 1).addReg(Op0r);
      BuildMI(*MBB, IP, X86::FNSTSW8r, 0);
      BuildMI(*MBB, IP, X86::SAHF, 1);
      return OpNum;
    }

  unsigned Op1r = getReg(Op1, MBB, IP);
  switch (Class) {
  default: assert(0 && "Unknown type class!");
    // Emit: cmp <var1>, <var2> (do the comparison).  We can
    // compare 8-bit with 8-bit, 16-bit with 16-bit, 32-bit with
    // 32-bit.
  case cByte:
    BuildMI(*MBB, IP, X86::CMP8rr, 2).addReg(Op0r).addReg(Op1r);
    break;
  case cShort:
    BuildMI(*MBB, IP, X86::CMP16rr, 2).addReg(Op0r).addReg(Op1r);
    break;
  case cInt:
    BuildMI(*MBB, IP, X86::CMP32rr, 2).addReg(Op0r).addReg(Op1r);
    break;
  case cFP:
    BuildMI(*MBB, IP, X86::FpUCOM, 2).addReg(Op0r).addReg(Op1r);
    BuildMI(*MBB, IP, X86::FNSTSW8r, 0);
    BuildMI(*MBB, IP, X86::SAHF, 1);
    break;

  case cLong:
    if (OpNum < 2) {    // seteq, setne
      unsigned LoTmp = makeAnotherReg(Type::IntTy);
      unsigned HiTmp = makeAnotherReg(Type::IntTy);
      unsigned FinalTmp = makeAnotherReg(Type::IntTy);
      BuildMI(*MBB, IP, X86::XOR32rr, 2, LoTmp).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, X86::XOR32rr, 2, HiTmp).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, X86::OR32rr,  2, FinalTmp).addReg(LoTmp).addReg(HiTmp);
      break;  // Allow the sete or setne to be generated from flags set by OR
    } else {
      // Emit a sequence of code which compares the high and low parts once
      // each, then uses a conditional move to handle the overflow case.  For
      // example, a setlt for long would generate code like this:
      //
      // AL = lo(op1) < lo(op2)   // Signedness depends on operands
      // BL = hi(op1) < hi(op2)   // Always unsigned comparison
      // dest = hi(op1) == hi(op2) ? AL : BL;
      //

      // FIXME: This would be much better if we had hierarchical register
      // classes!  Until then, hardcode registers so that we can deal with their
      // aliases (because we don't have conditional byte moves).
      //
      BuildMI(*MBB, IP, X86::CMP32rr, 2).addReg(Op0r).addReg(Op1r);
      BuildMI(*MBB, IP, SetCCOpcodeTab[0][OpNum], 0, X86::AL);
      BuildMI(*MBB, IP, X86::CMP32rr, 2).addReg(Op0r+1).addReg(Op1r+1);
      BuildMI(*MBB, IP, SetCCOpcodeTab[CompTy->isSigned()][OpNum], 0, X86::BL);
      BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, X86::BH);
      BuildMI(*MBB, IP, X86::IMPLICIT_DEF, 0, X86::AH);
      BuildMI(*MBB, IP, X86::CMOVE16rr, 2, X86::BX).addReg(X86::BX)
                                                   .addReg(X86::AX);
      // NOTE: visitSetCondInst knows that the value is dumped into the BL
      // register at this point for long values...
      return OpNum;
    }
  }
  return OpNum;
}


/// SetCC instructions - Here we just emit boilerplate code to set a byte-sized
/// register, then move it to wherever the result should be. 
///
void ISel::visitSetCondInst(SetCondInst &I) {
  if (canFoldSetCCIntoBranch(&I)) return;  // Fold this into a branch...

  unsigned DestReg = getReg(I);
  MachineBasicBlock::iterator MII = BB->end();
  emitSetCCOperation(BB, MII, I.getOperand(0), I.getOperand(1), I.getOpcode(),
                     DestReg);
}

/// emitSetCCOperation - Common code shared between visitSetCondInst and
/// constant expression support.
///
void ISel::emitSetCCOperation(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP,
                              Value *Op0, Value *Op1, unsigned Opcode,
                              unsigned TargetReg) {
  unsigned OpNum = getSetCCNumber(Opcode);
  OpNum = EmitComparison(OpNum, Op0, Op1, MBB, IP);

  const Type *CompTy = Op0->getType();
  unsigned CompClass = getClassB(CompTy);
  bool isSigned = CompTy->isSigned() && CompClass != cFP;

  if (CompClass != cLong || OpNum < 2) {
    // Handle normal comparisons with a setcc instruction...
    BuildMI(*MBB, IP, SetCCOpcodeTab[isSigned][OpNum], 0, TargetReg);
  } else {
    // Handle long comparisons by copying the value which is already in BL into
    // the register we want...
    BuildMI(*MBB, IP, X86::MOV8rr, 1, TargetReg).addReg(X86::BL);
  }
}




/// promote32 - Emit instructions to turn a narrow operand into a 32-bit-wide
/// operand, in the specified target register.
///
void ISel::promote32(unsigned targetReg, const ValueRecord &VR) {
  bool isUnsigned = VR.Ty->isUnsigned();

  // Make sure we have the register number for this value...
  unsigned Reg = VR.Val ? getReg(VR.Val) : VR.Reg;

  switch (getClassB(VR.Ty)) {
  case cByte:
    // Extend value into target register (8->32)
    if (isUnsigned)
      BuildMI(BB, X86::MOVZX32rr8, 1, targetReg).addReg(Reg);
    else
      BuildMI(BB, X86::MOVSX32rr8, 1, targetReg).addReg(Reg);
    break;
  case cShort:
    // Extend value into target register (16->32)
    if (isUnsigned)
      BuildMI(BB, X86::MOVZX32rr16, 1, targetReg).addReg(Reg);
    else
      BuildMI(BB, X86::MOVSX32rr16, 1, targetReg).addReg(Reg);
    break;
  case cInt:
    // Move value into target register (32->32)
    BuildMI(BB, X86::MOV32rr, 1, targetReg).addReg(Reg);
    break;
  default:
    assert(0 && "Unpromotable operand class in promote32");
  }
}

/// 'ret' instruction - Here we are interested in meeting the x86 ABI.  As such,
/// we have the following possibilities:
///
///   ret void: No return value, simply emit a 'ret' instruction
///   ret sbyte, ubyte : Extend value into EAX and return
///   ret short, ushort: Extend value into EAX and return
///   ret int, uint    : Move value into EAX and return
///   ret pointer      : Move value into EAX and return
///   ret long, ulong  : Move value into EAX/EDX and return
///   ret float/double : Top of FP stack
///
void ISel::visitReturnInst(ReturnInst &I) {
  if (I.getNumOperands() == 0) {
    BuildMI(BB, X86::RET, 0); // Just emit a 'ret' instruction
    return;
  }

  Value *RetVal = I.getOperand(0);
  unsigned RetReg = getReg(RetVal);
  switch (getClassB(RetVal->getType())) {
  case cByte:   // integral return values: extend or move into EAX and return
  case cShort:
  case cInt:
    promote32(X86::EAX, ValueRecord(RetReg, RetVal->getType()));
    // Declare that EAX is live on exit
    BuildMI(BB, X86::IMPLICIT_USE, 2).addReg(X86::EAX).addReg(X86::ESP);
    break;
  case cFP:                   // Floats & Doubles: Return in ST(0)
    BuildMI(BB, X86::FpSETRESULT, 1).addReg(RetReg);
    // Declare that top-of-stack is live on exit
    BuildMI(BB, X86::IMPLICIT_USE, 2).addReg(X86::ST0).addReg(X86::ESP);
    break;
  case cLong:
    BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(RetReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDX).addReg(RetReg+1);
    // Declare that EAX & EDX are live on exit
    BuildMI(BB, X86::IMPLICIT_USE, 3).addReg(X86::EAX).addReg(X86::EDX)
      .addReg(X86::ESP);
    break;
  default:
    visitInstruction(I);
  }
  // Emit a 'ret' instruction
  BuildMI(BB, X86::RET, 0);
}

// getBlockAfter - Return the basic block which occurs lexically after the
// specified one.
static inline BasicBlock *getBlockAfter(BasicBlock *BB) {
  Function::iterator I = BB; ++I;  // Get iterator to next block
  return I != BB->getParent()->end() ? &*I : 0;
}

/// visitBranchInst - Handle conditional and unconditional branches here.  Note
/// that since code layout is frozen at this point, that if we are trying to
/// jump to a block that is the immediate successor of the current block, we can
/// just make a fall-through (but we don't currently).
///
void ISel::visitBranchInst(BranchInst &BI) {
  BasicBlock *NextBB = getBlockAfter(BI.getParent());  // BB after current one

  if (!BI.isConditional()) {  // Unconditional branch?
    if (BI.getSuccessor(0) != NextBB)
      BuildMI(BB, X86::JMP, 1).addPCDisp(BI.getSuccessor(0));
    return;
  }

  // See if we can fold the setcc into the branch itself...
  SetCondInst *SCI = canFoldSetCCIntoBranch(BI.getCondition());
  if (SCI == 0) {
    // Nope, cannot fold setcc into this branch.  Emit a branch on a condition
    // computed some other way...
    unsigned condReg = getReg(BI.getCondition());
    BuildMI(BB, X86::CMP8ri, 2).addReg(condReg).addImm(0);
    if (BI.getSuccessor(1) == NextBB) {
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, X86::JNE, 1).addPCDisp(BI.getSuccessor(0));
    } else {
      BuildMI(BB, X86::JE, 1).addPCDisp(BI.getSuccessor(1));
      
      if (BI.getSuccessor(0) != NextBB)
        BuildMI(BB, X86::JMP, 1).addPCDisp(BI.getSuccessor(0));
    }
    return;
  }

  unsigned OpNum = getSetCCNumber(SCI->getOpcode());
  MachineBasicBlock::iterator MII = BB->end();
  OpNum = EmitComparison(OpNum, SCI->getOperand(0), SCI->getOperand(1), BB,MII);

  const Type *CompTy = SCI->getOperand(0)->getType();
  bool isSigned = CompTy->isSigned() && getClassB(CompTy) != cFP;
  

  // LLVM  -> X86 signed  X86 unsigned
  // -----    ----------  ------------
  // seteq -> je          je
  // setne -> jne         jne
  // setlt -> jl          jb
  // setge -> jge         jae
  // setgt -> jg          ja
  // setle -> jle         jbe
  // ----
  //          js                  // Used by comparison with 0 optimization
  //          jns

  static const unsigned OpcodeTab[2][8] = {
    { X86::JE, X86::JNE, X86::JB, X86::JAE, X86::JA, X86::JBE, 0, 0 },
    { X86::JE, X86::JNE, X86::JL, X86::JGE, X86::JG, X86::JLE,
      X86::JS, X86::JNS },
  };
  
  if (BI.getSuccessor(0) != NextBB) {
    BuildMI(BB, OpcodeTab[isSigned][OpNum], 1).addPCDisp(BI.getSuccessor(0));
    if (BI.getSuccessor(1) != NextBB)
      BuildMI(BB, X86::JMP, 1).addPCDisp(BI.getSuccessor(1));
  } else {
    // Change to the inverse condition...
    if (BI.getSuccessor(1) != NextBB) {
      OpNum ^= 1;
      BuildMI(BB, OpcodeTab[isSigned][OpNum], 1).addPCDisp(BI.getSuccessor(1));
    }
  }
}


/// doCall - This emits an abstract call instruction, setting up the arguments
/// and the return value as appropriate.  For the actual function call itself,
/// it inserts the specified CallMI instruction into the stream.
///
void ISel::doCall(const ValueRecord &Ret, MachineInstr *CallMI,
                  const std::vector<ValueRecord> &Args) {

  // Count how many bytes are to be pushed on the stack...
  unsigned NumBytes = 0;

  if (!Args.empty()) {
    for (unsigned i = 0, e = Args.size(); i != e; ++i)
      switch (getClassB(Args[i].Ty)) {
      case cByte: case cShort: case cInt:
        NumBytes += 4; break;
      case cLong:
        NumBytes += 8; break;
      case cFP:
        NumBytes += Args[i].Ty == Type::FloatTy ? 4 : 8;
        break;
      default: assert(0 && "Unknown class!");
      }

    // Adjust the stack pointer for the new arguments...
    BuildMI(BB, X86::ADJCALLSTACKDOWN, 1).addImm(NumBytes);

    // Arguments go on the stack in reverse order, as specified by the ABI.
    unsigned ArgOffset = 0;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
      unsigned ArgReg;
      switch (getClassB(Args[i].Ty)) {
      case cByte:
      case cShort:
        if (Args[i].Val && isa<ConstantInt>(Args[i].Val)) {
          // Zero/Sign extend constant, then stuff into memory.
          ConstantInt *Val = cast<ConstantInt>(Args[i].Val);
          Val = cast<ConstantInt>(ConstantExpr::getCast(Val, Type::IntTy));
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5), X86::ESP, ArgOffset)
            .addImm(Val->getRawValue() & 0xFFFFFFFF);
        } else {
          // Promote arg to 32 bits wide into a temporary register...
          ArgReg = makeAnotherReg(Type::UIntTy);
          promote32(ArgReg, Args[i]);
          addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
        }
        break;
      case cInt:
        if (Args[i].Val && isa<ConstantInt>(Args[i].Val)) {
          unsigned Val = cast<ConstantInt>(Args[i].Val)->getRawValue();
          addRegOffset(BuildMI(BB, X86::MOV32mi, 5),
                       X86::ESP, ArgOffset).addImm(Val);
        } else {
          ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
          addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
        }
        break;
      case cLong:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                     X86::ESP, ArgOffset).addReg(ArgReg);
        addRegOffset(BuildMI(BB, X86::MOV32mr, 5),
                     X86::ESP, ArgOffset+4).addReg(ArgReg+1);
        ArgOffset += 4;        // 8 byte entry, not 4.
        break;
        
      case cFP:
        ArgReg = Args[i].Val ? getReg(Args[i].Val) : Args[i].Reg;
        if (Args[i].Ty == Type::FloatTy) {
          addRegOffset(BuildMI(BB, X86::FST32m, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
        } else {
          assert(Args[i].Ty == Type::DoubleTy && "Unknown FP type!");
          addRegOffset(BuildMI(BB, X86::FST64m, 5),
                       X86::ESP, ArgOffset).addReg(ArgReg);
          ArgOffset += 4;       // 8 byte entry, not 4.
        }
        break;

      default: assert(0 && "Unknown class!");
      }
      ArgOffset += 4;
    }
  } else {
    BuildMI(BB, X86::ADJCALLSTACKDOWN, 1).addImm(0);
  }

  BB->push_back(CallMI);

  BuildMI(BB, X86::ADJCALLSTACKUP, 1).addImm(NumBytes);

  // If there is a return value, scavenge the result from the location the call
  // leaves it in...
  //
  if (Ret.Ty != Type::VoidTy) {
    unsigned DestClass = getClassB(Ret.Ty);
    switch (DestClass) {
    case cByte:
    case cShort:
    case cInt: {
      // Integral results are in %eax, or the appropriate portion
      // thereof.
      static const unsigned regRegMove[] = {
        X86::MOV8rr, X86::MOV16rr, X86::MOV32rr
      };
      static const unsigned AReg[] = { X86::AL, X86::AX, X86::EAX };
      BuildMI(BB, regRegMove[DestClass], 1, Ret.Reg).addReg(AReg[DestClass]);
      break;
    }
    case cFP:     // Floating-point return values live in %ST(0)
      BuildMI(BB, X86::FpGETRESULT, 1, Ret.Reg);
      break;
    case cLong:   // Long values are left in EDX:EAX
      BuildMI(BB, X86::MOV32rr, 1, Ret.Reg).addReg(X86::EAX);
      BuildMI(BB, X86::MOV32rr, 1, Ret.Reg+1).addReg(X86::EDX);
      break;
    default: assert(0 && "Unknown class!");
    }
  }
}


/// visitCallInst - Push args on stack and do a procedure call instruction.
void ISel::visitCallInst(CallInst &CI) {
  MachineInstr *TheCall;
  if (Function *F = CI.getCalledFunction()) {
    // Is it an intrinsic function call?
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID()) {
      visitIntrinsicCall(ID, CI);   // Special intrinsics are not handled here
      return;
    }

    // Emit a CALL instruction with PC-relative displacement.
    TheCall = BuildMI(X86::CALLpcrel32, 1).addGlobalAddress(F, true);
  } else {  // Emit an indirect call...
    unsigned Reg = getReg(CI.getCalledValue());
    TheCall = BuildMI(X86::CALL32r, 1).addReg(Reg);
  }

  std::vector<ValueRecord> Args;
  for (unsigned i = 1, e = CI.getNumOperands(); i != e; ++i)
    Args.push_back(ValueRecord(CI.getOperand(i)));

  unsigned DestReg = CI.getType() != Type::VoidTy ? getReg(CI) : 0;
  doCall(ValueRecord(DestReg, CI.getType()), TheCall, Args);
}         


/// LowerUnknownIntrinsicFunctionCalls - This performs a prepass over the
/// function, lowering any calls to unknown intrinsic functions into the
/// equivalent LLVM code.
///
void ISel::LowerUnknownIntrinsicFunctionCalls(Function &F) {
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; )
      if (CallInst *CI = dyn_cast<CallInst>(I++))
        if (Function *F = CI->getCalledFunction())
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
          case Intrinsic::returnaddress:
          case Intrinsic::frameaddress:
          case Intrinsic::memcpy:
          case Intrinsic::memset:
            // We directly implement these intrinsics
            break;
          default:
            // All other intrinsic calls we must lower.
            Instruction *Before = CI->getPrev();
            TM.getIntrinsicLowering().LowerIntrinsicCall(CI);
            if (Before) {        // Move iterator to instruction after call
              I = Before;  ++I;
            } else {
              I = BB->begin();
            }
          }

}

void ISel::visitIntrinsicCall(Intrinsic::ID ID, CallInst &CI) {
  unsigned TmpReg1, TmpReg2;
  switch (ID) {
  case Intrinsic::vastart:
    // Get the address of the first vararg value...
    TmpReg1 = getReg(CI);
    addFrameReference(BuildMI(BB, X86::LEA32r, 5, TmpReg1), VarArgsFrameIndex);
    return;

  case Intrinsic::vacopy:
    TmpReg1 = getReg(CI);
    TmpReg2 = getReg(CI.getOperand(1));
    BuildMI(BB, X86::MOV32rr, 1, TmpReg1).addReg(TmpReg2);
    return;
  case Intrinsic::vaend: return;   // Noop on X86

  case Intrinsic::returnaddress:
  case Intrinsic::frameaddress:
    TmpReg1 = getReg(CI);
    if (cast<Constant>(CI.getOperand(1))->isNullValue()) {
      if (ID == Intrinsic::returnaddress) {
        // Just load the return address
        addFrameReference(BuildMI(BB, X86::MOV32rm, 4, TmpReg1),
                          ReturnAddressIndex);
      } else {
        addFrameReference(BuildMI(BB, X86::LEA32r, 4, TmpReg1),
                          ReturnAddressIndex, -4);
      }
    } else {
      // Values other than zero are not implemented yet.
      BuildMI(BB, X86::MOV32ri, 1, TmpReg1).addImm(0);
    }
    return;

  case Intrinsic::memcpy: {
    assert(CI.getNumOperands() == 5 && "Illegal llvm.memcpy call!");
    unsigned Align = 1;
    if (ConstantInt *AlignC = dyn_cast<ConstantInt>(CI.getOperand(4))) {
      Align = AlignC->getRawValue();
      if (Align == 0) Align = 1;
    }

    // Turn the byte code into # iterations
    unsigned CountReg;
    unsigned Opcode;
    switch (Align & 3) {
    case 2:   // WORD aligned
      if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
        CountReg = getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/2));
      } else {
        CountReg = makeAnotherReg(Type::IntTy);
        unsigned ByteReg = getReg(CI.getOperand(3));
        BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(1);
      }
      Opcode = X86::REP_MOVSW;
      break;
    case 0:   // DWORD aligned
      if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
        CountReg = getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/4));
      } else {
        CountReg = makeAnotherReg(Type::IntTy);
        unsigned ByteReg = getReg(CI.getOperand(3));
        BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(2);
      }
      Opcode = X86::REP_MOVSD;
      break;
    default:  // BYTE aligned
      CountReg = getReg(CI.getOperand(3));
      Opcode = X86::REP_MOVSB;
      break;
    }

    // No matter what the alignment is, we put the source in ESI, the
    // destination in EDI, and the count in ECX.
    TmpReg1 = getReg(CI.getOperand(1));
    TmpReg2 = getReg(CI.getOperand(2));
    BuildMI(BB, X86::MOV32rr, 1, X86::ECX).addReg(CountReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDI).addReg(TmpReg1);
    BuildMI(BB, X86::MOV32rr, 1, X86::ESI).addReg(TmpReg2);
    BuildMI(BB, Opcode, 0);
    return;
  }
  case Intrinsic::memset: {
    assert(CI.getNumOperands() == 5 && "Illegal llvm.memset call!");
    unsigned Align = 1;
    if (ConstantInt *AlignC = dyn_cast<ConstantInt>(CI.getOperand(4))) {
      Align = AlignC->getRawValue();
      if (Align == 0) Align = 1;
    }

    // Turn the byte code into # iterations
    unsigned CountReg;
    unsigned Opcode;
    if (ConstantInt *ValC = dyn_cast<ConstantInt>(CI.getOperand(2))) {
      unsigned Val = ValC->getRawValue() & 255;

      // If the value is a constant, then we can potentially use larger copies.
      switch (Align & 3) {
      case 2:   // WORD aligned
        if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
          CountReg =getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/2));
        } else {
          CountReg = makeAnotherReg(Type::IntTy);
          unsigned ByteReg = getReg(CI.getOperand(3));
          BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(1);
        }
        BuildMI(BB, X86::MOV16ri, 1, X86::AX).addImm((Val << 8) | Val);
        Opcode = X86::REP_STOSW;
        break;
      case 0:   // DWORD aligned
        if (ConstantInt *I = dyn_cast<ConstantInt>(CI.getOperand(3))) {
          CountReg =getReg(ConstantUInt::get(Type::UIntTy, I->getRawValue()/4));
        } else {
          CountReg = makeAnotherReg(Type::IntTy);
          unsigned ByteReg = getReg(CI.getOperand(3));
          BuildMI(BB, X86::SHR32ri, 2, CountReg).addReg(ByteReg).addImm(2);
        }
        Val = (Val << 8) | Val;
        BuildMI(BB, X86::MOV32ri, 1, X86::EAX).addImm((Val << 16) | Val);
        Opcode = X86::REP_STOSD;
        break;
      default:  // BYTE aligned
        CountReg = getReg(CI.getOperand(3));
        BuildMI(BB, X86::MOV8ri, 1, X86::AL).addImm(Val);
        Opcode = X86::REP_STOSB;
        break;
      }
    } else {
      // If it's not a constant value we are storing, just fall back.  We could
      // try to be clever to form 16 bit and 32 bit values, but we don't yet.
      unsigned ValReg = getReg(CI.getOperand(2));
      BuildMI(BB, X86::MOV8rr, 1, X86::AL).addReg(ValReg);
      CountReg = getReg(CI.getOperand(3));
      Opcode = X86::REP_STOSB;
    }

    // No matter what the alignment is, we put the source in ESI, the
    // destination in EDI, and the count in ECX.
    TmpReg1 = getReg(CI.getOperand(1));
    //TmpReg2 = getReg(CI.getOperand(2));
    BuildMI(BB, X86::MOV32rr, 1, X86::ECX).addReg(CountReg);
    BuildMI(BB, X86::MOV32rr, 1, X86::EDI).addReg(TmpReg1);
    BuildMI(BB, Opcode, 0);
    return;
  }

  default: assert(0 && "Error: unknown intrinsics should have been lowered!");
  }
}

static bool isSafeToFoldLoadIntoInstruction(LoadInst &LI, Instruction &User) {
  if (LI.getParent() != User.getParent())
    return false;
  BasicBlock::iterator It = &LI;
  // Check all of the instructions between the load and the user.  We should
  // really use alias analysis here, but for now we just do something simple.
  for (++It; It != BasicBlock::iterator(&User); ++It) {
    switch (It->getOpcode()) {
    case Instruction::Malloc:
    case Instruction::Free:
    case Instruction::Store:
    case Instruction::Call:
    case Instruction::Invoke:
      return false;
    }
  }
  return true;
}


/// visitSimpleBinary - Implement simple binary operators for integral types...
/// OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for Or, 4 for
/// Xor.
///
void ISel::visitSimpleBinary(BinaryOperator &B, unsigned OperatorClass) {
  unsigned DestReg = getReg(B);
  MachineBasicBlock::iterator MI = BB->end();
  Value *Op0 = B.getOperand(0), *Op1 = B.getOperand(1);

  // Special case: op Reg, load [mem]
  if (isa<LoadInst>(Op0) && !isa<LoadInst>(Op1))
    if (!B.swapOperands())
      std::swap(Op0, Op1);  // Make sure any loads are in the RHS.

  unsigned Class = getClassB(B.getType());
  if (isa<LoadInst>(Op1) && Class < cFP &&
      isSafeToFoldLoadIntoInstruction(*cast<LoadInst>(Op1), B)) {

    static const unsigned OpcodeTab[][3] = {
      // Arithmetic operators
      { X86::ADD8rm, X86::ADD16rm, X86::ADD32rm },  // ADD
      { X86::SUB8rm, X86::SUB16rm, X86::SUB32rm },  // SUB
      
      // Bitwise operators
      { X86::AND8rm, X86::AND16rm, X86::AND32rm },  // AND
      { X86:: OR8rm, X86:: OR16rm, X86:: OR32rm },  // OR
      { X86::XOR8rm, X86::XOR16rm, X86::XOR32rm },  // XOR
    };
  
    assert(Class < cFP && "General code handles 64-bit integer types!");
    unsigned Opcode = OpcodeTab[OperatorClass][Class];

    unsigned BaseReg, Scale, IndexReg, Disp;
    getAddressingMode(cast<LoadInst>(Op1)->getOperand(0), BaseReg,
                      Scale, IndexReg, Disp);

    unsigned Op0r = getReg(Op0);
    addFullAddress(BuildMI(BB, Opcode, 2, DestReg).addReg(Op0r),
                   BaseReg, Scale, IndexReg, Disp);
    return;
  }

  emitSimpleBinaryOperation(BB, MI, Op0, Op1, OperatorClass, DestReg);
}

/// emitSimpleBinaryOperation - Implement simple binary operators for integral
/// types...  OperatorClass is one of: 0 for Add, 1 for Sub, 2 for And, 3 for
/// Or, 4 for Xor.
///
/// emitSimpleBinaryOperation - Common code shared between visitSimpleBinary
/// and constant expression support.
///
void ISel::emitSimpleBinaryOperation(MachineBasicBlock *MBB,
                                     MachineBasicBlock::iterator IP,
                                     Value *Op0, Value *Op1,
                                     unsigned OperatorClass, unsigned DestReg) {
  unsigned Class = getClassB(Op0->getType());

  // sub 0, X -> neg X
  if (OperatorClass == 1 && Class != cLong)
    if (ConstantInt *CI = dyn_cast<ConstantInt>(Op0)) {
      if (CI->isNullValue()) {
        unsigned op1Reg = getReg(Op1, MBB, IP);
        switch (Class) {
        default: assert(0 && "Unknown class for this function!");
        case cByte:
          BuildMI(*MBB, IP, X86::NEG8r, 1, DestReg).addReg(op1Reg);
          return;
        case cShort:
          BuildMI(*MBB, IP, X86::NEG16r, 1, DestReg).addReg(op1Reg);
          return;
        case cInt:
          BuildMI(*MBB, IP, X86::NEG32r, 1, DestReg).addReg(op1Reg);
          return;
        }
      }
    } else if (ConstantFP *CFP = dyn_cast<ConstantFP>(Op0))
      if (CFP->isExactlyValue(-0.0)) {
        // -0.0 - X === -X
        unsigned op1Reg = getReg(Op1, MBB, IP);
        BuildMI(*MBB, IP, X86::FCHS, 1, DestReg).addReg(op1Reg);
        return;
      }

  // Special case: op Reg, <const>
  if (Class != cLong && isa<ConstantInt>(Op1)) {
    ConstantInt *Op1C = cast<ConstantInt>(Op1);
    unsigned Op0r = getReg(Op0, MBB, IP);

    // xor X, -1 -> not X
    if (OperatorClass == 4 && Op1C->isAllOnesValue()) {
      static unsigned const NOTTab[] = { X86::NOT8r, X86::NOT16r, X86::NOT32r };
      BuildMI(*MBB, IP, NOTTab[Class], 1, DestReg).addReg(Op0r);
      return;
    }

    // add X, -1 -> dec X
    if (OperatorClass == 0 && Op1C->isAllOnesValue()) {
      static unsigned const DECTab[] = { X86::DEC8r, X86::DEC16r, X86::DEC32r };
      BuildMI(*MBB, IP, DECTab[Class], 1, DestReg).addReg(Op0r);
      return;
    }

    // add X, 1 -> inc X
    if (OperatorClass == 0 && Op1C->equalsInt(1)) {
      static unsigned const DECTab[] = { X86::INC8r, X86::INC16r, X86::INC32r };
      BuildMI(*MBB, IP, DECTab[Class], 1, DestReg).addReg(Op0r);
      return;
    }
  
    static const unsigned OpcodeTab[][3] = {
      // Arithmetic operators
      { X86::ADD8ri, X86::ADD16ri, X86::ADD32ri },  // ADD
      { X86::SUB8ri, X86::SUB16ri, X86::SUB32ri },  // SUB
    
      // Bitwise operators
      { X86::AND8ri, X86::AND16ri, X86::AND32ri },  // AND
      { X86:: OR8ri, X86:: OR16ri, X86:: OR32ri },  // OR
      { X86::XOR8ri, X86::XOR16ri, X86::XOR32ri },  // XOR
    };
  
    assert(Class < cFP && "General code handles 64-bit integer types!");
    unsigned Opcode = OpcodeTab[OperatorClass][Class];


    uint64_t Op1v = cast<ConstantInt>(Op1C)->getRawValue();
    BuildMI(*MBB, IP, Opcode, 5, DestReg).addReg(Op0r).addImm(Op1v);
    return;
  }

  // Finally, handle the general case now.
  static const unsigned OpcodeTab[][4] = {
    // Arithmetic operators
    { X86::ADD8rr, X86::ADD16rr, X86::ADD32rr, X86::FpADD },  // ADD
    { X86::SUB8rr, X86::SUB16rr, X86::SUB32rr, X86::FpSUB },  // SUB
      
    // Bitwise operators
    { X86::AND8rr, X86::AND16rr, X86::AND32rr, 0 },  // AND
    { X86:: OR8rr, X86:: OR16rr, X86:: OR32rr, 0 },  // OR
    { X86::XOR8rr, X86::XOR16rr, X86::XOR32rr, 0 },  // XOR
  };
    
  bool isLong = false;
  if (Class == cLong) {
    isLong = true;
    Class = cInt;          // Bottom 32 bits are handled just like ints
  }
    
  unsigned Opcode = OpcodeTab[OperatorClass][Class];
  assert(Opcode && "Floating point arguments to logical inst?");
  unsigned Op0r = getReg(Op0, MBB, IP);
  unsigned Op1r = getReg(Op1, MBB, IP);
  BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(Op0r).addReg(Op1r);
    
  if (isLong) {        // Handle the upper 32 bits of long values...
    static const unsigned TopTab[] = {
      X86::ADC32rr, X86::SBB32rr, X86::AND32rr, X86::OR32rr, X86::XOR32rr
    };
    BuildMI(*MBB, IP, TopTab[OperatorClass], 2,
            DestReg+1).addReg(Op0r+1).addReg(Op1r+1);
  }
}

/// doMultiply - Emit appropriate instructions to multiply together the
/// registers op0Reg and op1Reg, and put the result in DestReg.  The type of the
/// result should be given as DestTy.
///
void ISel::doMultiply(MachineBasicBlock *MBB, MachineBasicBlock::iterator MBBI,
                      unsigned DestReg, const Type *DestTy,
                      unsigned op0Reg, unsigned op1Reg) {
  unsigned Class = getClass(DestTy);
  switch (Class) {
  case cFP:              // Floating point multiply
    BuildMI(*MBB, MBBI, X86::FpMUL, 2, DestReg).addReg(op0Reg).addReg(op1Reg);
    return;
  case cInt:
  case cShort:
    BuildMI(*MBB, MBBI, Class == cInt ? X86::IMUL32rr:X86::IMUL16rr, 2, DestReg)
      .addReg(op0Reg).addReg(op1Reg);
    return;
  case cByte:
    // Must use the MUL instruction, which forces use of AL...
    BuildMI(*MBB, MBBI, X86::MOV8rr, 1, X86::AL).addReg(op0Reg);
    BuildMI(*MBB, MBBI, X86::MUL8r, 1).addReg(op1Reg);
    BuildMI(*MBB, MBBI, X86::MOV8rr, 1, DestReg).addReg(X86::AL);
    return;
  default:
  case cLong: assert(0 && "doMultiply cannot operate on LONG values!");
  }
}

// ExactLog2 - This function solves for (Val == 1 << (N-1)) and returns N.  It
// returns zero when the input is not exactly a power of two.
static unsigned ExactLog2(unsigned Val) {
  if (Val == 0) return 0;
  unsigned Count = 0;
  while (Val != 1) {
    if (Val & 1) return 0;
    Val >>= 1;
    ++Count;
  }
  return Count+1;
}

void ISel::doMultiplyConst(MachineBasicBlock *MBB,
                           MachineBasicBlock::iterator IP,
                           unsigned DestReg, const Type *DestTy,
                           unsigned op0Reg, unsigned ConstRHS) {
  unsigned Class = getClass(DestTy);

  // If the element size is exactly a power of 2, use a shift to get it.
  if (unsigned Shift = ExactLog2(ConstRHS)) {
    switch (Class) {
    default: assert(0 && "Unknown class for this function!");
    case cByte:
      BuildMI(*MBB, IP, X86::SHL32ri,2, DestReg).addReg(op0Reg).addImm(Shift-1);
      return;
    case cShort:
      BuildMI(*MBB, IP, X86::SHL32ri,2, DestReg).addReg(op0Reg).addImm(Shift-1);
      return;
    case cInt:
      BuildMI(*MBB, IP, X86::SHL32ri,2, DestReg).addReg(op0Reg).addImm(Shift-1);
      return;
    }
  }
  
  if (Class == cShort) {
    BuildMI(*MBB, IP, X86::IMUL16rri,2,DestReg).addReg(op0Reg).addImm(ConstRHS);
    return;
  } else if (Class == cInt) {
    BuildMI(*MBB, IP, X86::IMUL32rri,2,DestReg).addReg(op0Reg).addImm(ConstRHS);
    return;
  }

  // Most general case, emit a normal multiply...
  static const unsigned MOVriTab[] = {
    X86::MOV8ri, X86::MOV16ri, X86::MOV32ri
  };

  unsigned TmpReg = makeAnotherReg(DestTy);
  BuildMI(*MBB, IP, MOVriTab[Class], 1, TmpReg).addImm(ConstRHS);
  
  // Emit a MUL to multiply the register holding the index by
  // elementSize, putting the result in OffsetReg.
  doMultiply(MBB, IP, DestReg, DestTy, op0Reg, TmpReg);
}

/// visitMul - Multiplies are not simple binary operators because they must deal
/// with the EAX register explicitly.
///
void ISel::visitMul(BinaryOperator &I) {
  unsigned Op0Reg  = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  // Simple scalar multiply?
  if (I.getType() != Type::LongTy && I.getType() != Type::ULongTy) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(1))) {
      unsigned Val = (unsigned)CI->getRawValue(); // Cannot be 64-bit constant
      MachineBasicBlock::iterator MBBI = BB->end();
      doMultiplyConst(BB, MBBI, DestReg, I.getType(), Op0Reg, Val);
    } else {
      unsigned Op1Reg  = getReg(I.getOperand(1));
      MachineBasicBlock::iterator MBBI = BB->end();
      doMultiply(BB, MBBI, DestReg, I.getType(), Op0Reg, Op1Reg);
    }
  } else {
    unsigned Op1Reg  = getReg(I.getOperand(1));

    // Long value.  We have to do things the hard way...
    // Multiply the two low parts... capturing carry into EDX
    BuildMI(BB, X86::MOV32rr, 1, X86::EAX).addReg(Op0Reg);
    BuildMI(BB, X86::MUL32r, 1).addReg(Op1Reg);  // AL*BL

    unsigned OverflowReg = makeAnotherReg(Type::UIntTy);
    BuildMI(BB, X86::MOV32rr, 1, DestReg).addReg(X86::EAX);     // AL*BL
    BuildMI(BB, X86::MOV32rr, 1, OverflowReg).addReg(X86::EDX); // AL*BL >> 32

    MachineBasicBlock::iterator MBBI = BB->end();
    unsigned AHBLReg = makeAnotherReg(Type::UIntTy);   // AH*BL
    BuildMI(*BB, MBBI, X86::IMUL32rr,2,AHBLReg).addReg(Op0Reg+1).addReg(Op1Reg);

    unsigned AHBLplusOverflowReg = makeAnotherReg(Type::UIntTy);
    BuildMI(*BB, MBBI, X86::ADD32rr, 2,                  // AH*BL+(AL*BL >> 32)
            AHBLplusOverflowReg).addReg(AHBLReg).addReg(OverflowReg);
    
    MBBI = BB->end();
    unsigned ALBHReg = makeAnotherReg(Type::UIntTy); // AL*BH
    BuildMI(*BB, MBBI, X86::IMUL32rr,2,ALBHReg).addReg(Op0Reg).addReg(Op1Reg+1);
    
    BuildMI(*BB, MBBI, X86::ADD32rr, 2,         // AL*BH + AH*BL + (AL*BL >> 32)
            DestReg+1).addReg(AHBLplusOverflowReg).addReg(ALBHReg);
  }
}


/// visitDivRem - Handle division and remainder instructions... these
/// instruction both require the same instructions to be generated, they just
/// select the result from a different register.  Note that both of these
/// instructions work differently for signed and unsigned operands.
///
void ISel::visitDivRem(BinaryOperator &I) {
  unsigned Op0Reg = getReg(I.getOperand(0));
  unsigned Op1Reg = getReg(I.getOperand(1));
  unsigned ResultReg = getReg(I);

  MachineBasicBlock::iterator IP = BB->end();
  emitDivRemOperation(BB, IP, Op0Reg, Op1Reg, I.getOpcode() == Instruction::Div,
                      I.getType(), ResultReg);
}

void ISel::emitDivRemOperation(MachineBasicBlock *BB,
                               MachineBasicBlock::iterator IP,
                               unsigned Op0Reg, unsigned Op1Reg, bool isDiv,
                               const Type *Ty, unsigned ResultReg) {
  unsigned Class = getClass(Ty);
  switch (Class) {
  case cFP:              // Floating point divide
    if (isDiv) {
      BuildMI(*BB, IP, X86::FpDIV, 2, ResultReg).addReg(Op0Reg).addReg(Op1Reg);
    } else {               // Floating point remainder...
      MachineInstr *TheCall =
        BuildMI(X86::CALLpcrel32, 1).addExternalSymbol("fmod", true);
      std::vector<ValueRecord> Args;
      Args.push_back(ValueRecord(Op0Reg, Type::DoubleTy));
      Args.push_back(ValueRecord(Op1Reg, Type::DoubleTy));
      doCall(ValueRecord(ResultReg, Type::DoubleTy), TheCall, Args);
    }
    return;
  case cLong: {
    static const char *FnName[] =
      { "__moddi3", "__divdi3", "__umoddi3", "__udivdi3" };

    unsigned NameIdx = Ty->isUnsigned()*2 + isDiv;
    MachineInstr *TheCall =
      BuildMI(X86::CALLpcrel32, 1).addExternalSymbol(FnName[NameIdx], true);

    std::vector<ValueRecord> Args;
    Args.push_back(ValueRecord(Op0Reg, Type::LongTy));
    Args.push_back(ValueRecord(Op1Reg, Type::LongTy));
    doCall(ValueRecord(ResultReg, Type::LongTy), TheCall, Args);
    return;
  }
  case cByte: case cShort: case cInt:
    break;          // Small integrals, handled below...
  default: assert(0 && "Unknown class!");
  }

  static const unsigned Regs[]     ={ X86::AL    , X86::AX     , X86::EAX     };
  static const unsigned MovOpcode[]={ X86::MOV8rr, X86::MOV16rr, X86::MOV32rr };
  static const unsigned SarOpcode[]={ X86::SAR8ri, X86::SAR16ri, X86::SAR32ri };
  static const unsigned ClrOpcode[]={ X86::MOV8ri, X86::MOV16ri, X86::MOV32ri };
  static const unsigned ExtRegs[]  ={ X86::AH    , X86::DX     , X86::EDX     };

  static const unsigned DivOpcode[][4] = {
    { X86::DIV8r , X86::DIV16r , X86::DIV32r , 0 },  // Unsigned division
    { X86::IDIV8r, X86::IDIV16r, X86::IDIV32r, 0 },  // Signed division
  };

  bool isSigned   = Ty->isSigned();
  unsigned Reg    = Regs[Class];
  unsigned ExtReg = ExtRegs[Class];

  // Put the first operand into one of the A registers...
  BuildMI(*BB, IP, MovOpcode[Class], 1, Reg).addReg(Op0Reg);

  if (isSigned) {
    // Emit a sign extension instruction...
    unsigned ShiftResult = makeAnotherReg(Ty);
    BuildMI(*BB, IP, SarOpcode[Class], 2,ShiftResult).addReg(Op0Reg).addImm(31);
    BuildMI(*BB, IP, MovOpcode[Class], 1, ExtReg).addReg(ShiftResult);
  } else {
    // If unsigned, emit a zeroing instruction... (reg = 0)
    BuildMI(*BB, IP, ClrOpcode[Class], 2, ExtReg).addImm(0);
  }

  // Emit the appropriate divide or remainder instruction...
  BuildMI(*BB, IP, DivOpcode[isSigned][Class], 1).addReg(Op1Reg);

  // Figure out which register we want to pick the result out of...
  unsigned DestReg = isDiv ? Reg : ExtReg;
  
  // Put the result into the destination register...
  BuildMI(*BB, IP, MovOpcode[Class], 1, ResultReg).addReg(DestReg);
}


/// Shift instructions: 'shl', 'sar', 'shr' - Some special cases here
/// for constant immediate shift values, and for constant immediate
/// shift values equal to 1. Even the general case is sort of special,
/// because the shift amount has to be in CL, not just any old register.
///
void ISel::visitShiftInst(ShiftInst &I) {
  MachineBasicBlock::iterator IP = BB->end ();
  emitShiftOperation (BB, IP, I.getOperand (0), I.getOperand (1),
                      I.getOpcode () == Instruction::Shl, I.getType (),
                      getReg (I));
}

/// emitShiftOperation - Common code shared between visitShiftInst and
/// constant expression support.
void ISel::emitShiftOperation(MachineBasicBlock *MBB,
                              MachineBasicBlock::iterator IP,
                              Value *Op, Value *ShiftAmount, bool isLeftShift,
                              const Type *ResultTy, unsigned DestReg) {
  unsigned SrcReg = getReg (Op, MBB, IP);
  bool isSigned = ResultTy->isSigned ();
  unsigned Class = getClass (ResultTy);
  
  static const unsigned ConstantOperand[][4] = {
    { X86::SHR8ri, X86::SHR16ri, X86::SHR32ri, X86::SHRD32rri8 },  // SHR
    { X86::SAR8ri, X86::SAR16ri, X86::SAR32ri, X86::SHRD32rri8 },  // SAR
    { X86::SHL8ri, X86::SHL16ri, X86::SHL32ri, X86::SHLD32rri8 },  // SHL
    { X86::SHL8ri, X86::SHL16ri, X86::SHL32ri, X86::SHLD32rri8 },  // SAL = SHL
  };

  static const unsigned NonConstantOperand[][4] = {
    { X86::SHR8rCL, X86::SHR16rCL, X86::SHR32rCL },  // SHR
    { X86::SAR8rCL, X86::SAR16rCL, X86::SAR32rCL },  // SAR
    { X86::SHL8rCL, X86::SHL16rCL, X86::SHL32rCL },  // SHL
    { X86::SHL8rCL, X86::SHL16rCL, X86::SHL32rCL },  // SAL = SHL
  };

  // Longs, as usual, are handled specially...
  if (Class == cLong) {
    // If we have a constant shift, we can generate much more efficient code
    // than otherwise...
    //
    if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
      unsigned Amount = CUI->getValue();
      if (Amount < 32) {
        const unsigned *Opc = ConstantOperand[isLeftShift*2+isSigned];
        if (isLeftShift) {
          BuildMI(*MBB, IP, Opc[3], 3, 
              DestReg+1).addReg(SrcReg+1).addReg(SrcReg).addImm(Amount);
          BuildMI(*MBB, IP, Opc[2], 2, DestReg).addReg(SrcReg).addImm(Amount);
        } else {
          BuildMI(*MBB, IP, Opc[3], 3,
              DestReg).addReg(SrcReg  ).addReg(SrcReg+1).addImm(Amount);
          BuildMI(*MBB, IP, Opc[2],2,DestReg+1).addReg(SrcReg+1).addImm(Amount);
        }
      } else {                 // Shifting more than 32 bits
        Amount -= 32;
        if (isLeftShift) {
          BuildMI(*MBB, IP, X86::SHL32ri, 2,
              DestReg + 1).addReg(SrcReg).addImm(Amount);
          BuildMI(*MBB, IP, X86::MOV32ri, 1,
              DestReg).addImm(0);
        } else {
          unsigned Opcode = isSigned ? X86::SAR32ri : X86::SHR32ri;
          BuildMI(*MBB, IP, Opcode, 2, DestReg).addReg(SrcReg+1).addImm(Amount);
          BuildMI(*MBB, IP, X86::MOV32ri, 1, DestReg+1).addImm(0);
        }
      }
    } else {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);

      if (!isLeftShift && isSigned) {
        // If this is a SHR of a Long, then we need to do funny sign extension
        // stuff.  TmpReg gets the value to use as the high-part if we are
        // shifting more than 32 bits.
        BuildMI(*MBB, IP, X86::SAR32ri, 2, TmpReg).addReg(SrcReg).addImm(31);
      } else {
        // Other shifts use a fixed zero value if the shift is more than 32
        // bits.
        BuildMI(*MBB, IP, X86::MOV32ri, 1, TmpReg).addImm(0);
      }

      // Initialize CL with the shift amount...
      unsigned ShiftAmountReg = getReg(ShiftAmount, MBB, IP);
      BuildMI(*MBB, IP, X86::MOV8rr, 1, X86::CL).addReg(ShiftAmountReg);

      unsigned TmpReg2 = makeAnotherReg(Type::IntTy);
      unsigned TmpReg3 = makeAnotherReg(Type::IntTy);
      if (isLeftShift) {
        // TmpReg2 = shld inHi, inLo
        BuildMI(*MBB, IP, X86::SHLD32rrCL,2,TmpReg2).addReg(SrcReg+1)
                                                    .addReg(SrcReg);
        // TmpReg3 = shl  inLo, CL
        BuildMI(*MBB, IP, X86::SHL32rCL, 1, TmpReg3).addReg(SrcReg);

        // Set the flags to indicate whether the shift was by more than 32 bits.
        BuildMI(*MBB, IP, X86::TEST8ri, 2).addReg(X86::CL).addImm(32);

        // DestHi = (>32) ? TmpReg3 : TmpReg2;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2, 
                DestReg+1).addReg(TmpReg2).addReg(TmpReg3);
        // DestLo = (>32) ? TmpReg : TmpReg3;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2,
            DestReg).addReg(TmpReg3).addReg(TmpReg);
      } else {
        // TmpReg2 = shrd inLo, inHi
        BuildMI(*MBB, IP, X86::SHRD32rrCL,2,TmpReg2).addReg(SrcReg)
                                                    .addReg(SrcReg+1);
        // TmpReg3 = s[ah]r  inHi, CL
        BuildMI(*MBB, IP, isSigned ? X86::SAR32rCL : X86::SHR32rCL, 1, TmpReg3)
                       .addReg(SrcReg+1);

        // Set the flags to indicate whether the shift was by more than 32 bits.
        BuildMI(*MBB, IP, X86::TEST8ri, 2).addReg(X86::CL).addImm(32);

        // DestLo = (>32) ? TmpReg3 : TmpReg2;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2, 
                DestReg).addReg(TmpReg2).addReg(TmpReg3);

        // DestHi = (>32) ? TmpReg : TmpReg3;
        BuildMI(*MBB, IP, X86::CMOVNE32rr, 2, 
                DestReg+1).addReg(TmpReg3).addReg(TmpReg);
      }
    }
    return;
  }

  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(ShiftAmount)) {
    // The shift amount is constant, guaranteed to be a ubyte. Get its value.
    assert(CUI->getType() == Type::UByteTy && "Shift amount not a ubyte?");

    const unsigned *Opc = ConstantOperand[isLeftShift*2+isSigned];
    BuildMI(*MBB, IP, Opc[Class], 2,
        DestReg).addReg(SrcReg).addImm(CUI->getValue());
  } else {                  // The shift amount is non-constant.
    unsigned ShiftAmountReg = getReg (ShiftAmount, MBB, IP);
    BuildMI(*MBB, IP, X86::MOV8rr, 1, X86::CL).addReg(ShiftAmountReg);

    const unsigned *Opc = NonConstantOperand[isLeftShift*2+isSigned];
    BuildMI(*MBB, IP, Opc[Class], 1, DestReg).addReg(SrcReg);
  }
}


void ISel::getAddressingMode(Value *Addr, unsigned &BaseReg, unsigned &Scale,
                             unsigned &IndexReg, unsigned &Disp) {
  BaseReg = 0; Scale = 1; IndexReg = 0; Disp = 0;
  if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Addr)) {
    if (isGEPFoldable(BB, GEP->getOperand(0), GEP->op_begin()+1, GEP->op_end(),
                       BaseReg, Scale, IndexReg, Disp))
      return;
  } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Addr)) {
    if (CE->getOpcode() == Instruction::GetElementPtr)
      if (isGEPFoldable(BB, CE->getOperand(0), CE->op_begin()+1, CE->op_end(),
                        BaseReg, Scale, IndexReg, Disp))
        return;
  }

  // If it's not foldable, reset addr mode.
  BaseReg = getReg(Addr);
  Scale = 1; IndexReg = 0; Disp = 0;
}


/// visitLoadInst - Implement LLVM load instructions in terms of the x86 'mov'
/// instruction.  The load and store instructions are the only place where we
/// need to worry about the memory layout of the target machine.
///
void ISel::visitLoadInst(LoadInst &I) {
  // Check to see if this load instruction is going to be folded into a binary
  // instruction, like add.  If so, we don't want to emit it.  Wouldn't a real
  // pattern matching instruction selector be nice?
  if (I.hasOneUse() && getClassB(I.getType()) < cFP) {
    Instruction *User = cast<Instruction>(I.use_back());
    switch (User->getOpcode()) {
    default: User = 0; break;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
      break;
    }

    if (User) {
      // Okay, we found a user.  If the load is the first operand and there is
      // no second operand load, reverse the operand ordering.  Note that this
      // can fail for a subtract (ie, no change will be made).
      if (!isa<LoadInst>(User->getOperand(1)))
        cast<BinaryOperator>(User)->swapOperands();
      
      // Okay, now that everything is set up, if this load is used by the second
      // operand, and if there are no instructions that invalidate the load
      // before the binary operator, eliminate the load.
      if (User->getOperand(1) == &I &&
          isSafeToFoldLoadIntoInstruction(I, *User))
        return;   // Eliminate the load!
    }
  }

  unsigned DestReg = getReg(I);
  unsigned BaseReg = 0, Scale = 1, IndexReg = 0, Disp = 0;
  getAddressingMode(I.getOperand(0), BaseReg, Scale, IndexReg, Disp);

  unsigned Class = getClassB(I.getType());
  if (Class == cLong) {
    addFullAddress(BuildMI(BB, X86::MOV32rm, 4, DestReg),
                   BaseReg, Scale, IndexReg, Disp);
    addFullAddress(BuildMI(BB, X86::MOV32rm, 4, DestReg+1),
                   BaseReg, Scale, IndexReg, Disp+4);
    return;
  }

  static const unsigned Opcodes[] = {
    X86::MOV8rm, X86::MOV16rm, X86::MOV32rm, X86::FLD32m
  };
  unsigned Opcode = Opcodes[Class];
  if (I.getType() == Type::DoubleTy) Opcode = X86::FLD64m;
  addFullAddress(BuildMI(BB, Opcode, 4, DestReg),
                 BaseReg, Scale, IndexReg, Disp);
}

/// visitStoreInst - Implement LLVM store instructions in terms of the x86 'mov'
/// instruction.
///
void ISel::visitStoreInst(StoreInst &I) {
  unsigned BaseReg, Scale, IndexReg, Disp;
  getAddressingMode(I.getOperand(1), BaseReg, Scale, IndexReg, Disp);

  const Type *ValTy = I.getOperand(0)->getType();
  unsigned Class = getClassB(ValTy);

  if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(0))) {
    uint64_t Val = CI->getRawValue();
    if (Class == cLong) {
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5),
                     BaseReg, Scale, IndexReg, Disp).addImm(Val & ~0U);
      addFullAddress(BuildMI(BB, X86::MOV32mi, 5),
                     BaseReg, Scale, IndexReg, Disp+4).addImm(Val>>32);
    } else {
      static const unsigned Opcodes[] = {
        X86::MOV8mi, X86::MOV16mi, X86::MOV32mi
      };
      unsigned Opcode = Opcodes[Class];
      addFullAddress(BuildMI(BB, Opcode, 5),
                     BaseReg, Scale, IndexReg, Disp).addImm(Val);
    }
  } else if (ConstantBool *CB = dyn_cast<ConstantBool>(I.getOperand(0))) {
    addFullAddress(BuildMI(BB, X86::MOV8mi, 5),
                   BaseReg, Scale, IndexReg, Disp).addImm(CB->getValue());
  } else {    
    if (Class == cLong) {
      unsigned ValReg = getReg(I.getOperand(0));
      addFullAddress(BuildMI(BB, X86::MOV32mr, 5),
                     BaseReg, Scale, IndexReg, Disp).addReg(ValReg);
      addFullAddress(BuildMI(BB, X86::MOV32mr, 5),
                     BaseReg, Scale, IndexReg, Disp+4).addReg(ValReg+1);
    } else {
      unsigned ValReg = getReg(I.getOperand(0));
      static const unsigned Opcodes[] = {
        X86::MOV8mr, X86::MOV16mr, X86::MOV32mr, X86::FST32m
      };
      unsigned Opcode = Opcodes[Class];
      if (ValTy == Type::DoubleTy) Opcode = X86::FST64m;
      addFullAddress(BuildMI(BB, Opcode, 1+4),
                     BaseReg, Scale, IndexReg, Disp).addReg(ValReg);
    }
  }
}


/// visitCastInst - Here we have various kinds of copying with or without sign
/// extension going on.
///
void ISel::visitCastInst(CastInst &CI) {
  Value *Op = CI.getOperand(0);
  // If this is a cast from a 32-bit integer to a Long type, and the only uses
  // of the case are GEP instructions, then the cast does not need to be
  // generated explicitly, it will be folded into the GEP.
  if (CI.getType() == Type::LongTy &&
      (Op->getType() == Type::IntTy || Op->getType() == Type::UIntTy)) {
    bool AllUsesAreGEPs = true;
    for (Value::use_iterator I = CI.use_begin(), E = CI.use_end(); I != E; ++I)
      if (!isa<GetElementPtrInst>(*I)) {
        AllUsesAreGEPs = false;
        break;
      }        

    // No need to codegen this cast if all users are getelementptr instrs...
    if (AllUsesAreGEPs) return;
  }

  unsigned DestReg = getReg(CI);
  MachineBasicBlock::iterator MI = BB->end();
  emitCastOperation(BB, MI, Op, CI.getType(), DestReg);
}

/// emitCastOperation - Common code shared between visitCastInst and constant
/// expression cast support.
///
void ISel::emitCastOperation(MachineBasicBlock *BB,
                             MachineBasicBlock::iterator IP,
                             Value *Src, const Type *DestTy,
                             unsigned DestReg) {
  unsigned SrcReg = getReg(Src, BB, IP);
  const Type *SrcTy = Src->getType();
  unsigned SrcClass = getClassB(SrcTy);
  unsigned DestClass = getClassB(DestTy);

  // Implement casts to bool by using compare on the operand followed by set if
  // not zero on the result.
  if (DestTy == Type::BoolTy) {
    switch (SrcClass) {
    case cByte:
      BuildMI(*BB, IP, X86::TEST8rr, 2).addReg(SrcReg).addReg(SrcReg);
      break;
    case cShort:
      BuildMI(*BB, IP, X86::TEST16rr, 2).addReg(SrcReg).addReg(SrcReg);
      break;
    case cInt:
      BuildMI(*BB, IP, X86::TEST32rr, 2).addReg(SrcReg).addReg(SrcReg);
      break;
    case cLong: {
      unsigned TmpReg = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, X86::OR32rr, 2, TmpReg).addReg(SrcReg).addReg(SrcReg+1);
      break;
    }
    case cFP:
      BuildMI(*BB, IP, X86::FTST, 1).addReg(SrcReg);
      BuildMI(*BB, IP, X86::FNSTSW8r, 0);
      BuildMI(*BB, IP, X86::SAHF, 1);
      break;
    }

    // If the zero flag is not set, then the value is true, set the byte to
    // true.
    BuildMI(*BB, IP, X86::SETNEr, 1, DestReg);
    return;
  }

  static const unsigned RegRegMove[] = {
    X86::MOV8rr, X86::MOV16rr, X86::MOV32rr, X86::FpMOV, X86::MOV32rr
  };

  // Implement casts between values of the same type class (as determined by
  // getClass) by using a register-to-register move.
  if (SrcClass == DestClass) {
    if (SrcClass <= cInt || (SrcClass == cFP && SrcTy == DestTy)) {
      BuildMI(*BB, IP, RegRegMove[SrcClass], 1, DestReg).addReg(SrcReg);
    } else if (SrcClass == cFP) {
      if (SrcTy == Type::FloatTy) {  // double -> float
        assert(DestTy == Type::DoubleTy && "Unknown cFP member!");
        BuildMI(*BB, IP, X86::FpMOV, 1, DestReg).addReg(SrcReg);
      } else {                       // float -> double
        assert(SrcTy == Type::DoubleTy && DestTy == Type::FloatTy &&
               "Unknown cFP member!");
        // Truncate from double to float by storing to memory as short, then
        // reading it back.
        unsigned FltAlign = TM.getTargetData().getFloatAlignment();
        int FrameIdx = F->getFrameInfo()->CreateStackObject(4, FltAlign);
        addFrameReference(BuildMI(*BB, IP, X86::FST32m, 5), FrameIdx).addReg(SrcReg);
        addFrameReference(BuildMI(*BB, IP, X86::FLD32m, 5, DestReg), FrameIdx);
      }
    } else if (SrcClass == cLong) {
      BuildMI(*BB, IP, X86::MOV32rr, 1, DestReg).addReg(SrcReg);
      BuildMI(*BB, IP, X86::MOV32rr, 1, DestReg+1).addReg(SrcReg+1);
    } else {
      assert(0 && "Cannot handle this type of cast instruction!");
      abort();
    }
    return;
  }

  // Handle cast of SMALLER int to LARGER int using a move with sign extension
  // or zero extension, depending on whether the source type was signed.
  if (SrcClass <= cInt && (DestClass <= cInt || DestClass == cLong) &&
      SrcClass < DestClass) {
    bool isLong = DestClass == cLong;
    if (isLong) DestClass = cInt;

    static const unsigned Opc[][4] = {
      { X86::MOVSX16rr8, X86::MOVSX32rr8, X86::MOVSX32rr16, X86::MOV32rr }, // s
      { X86::MOVZX16rr8, X86::MOVZX32rr8, X86::MOVZX32rr16, X86::MOV32rr }  // u
    };
    
    bool isUnsigned = SrcTy->isUnsigned();
    BuildMI(*BB, IP, Opc[isUnsigned][SrcClass + DestClass - 1], 1,
        DestReg).addReg(SrcReg);

    if (isLong) {  // Handle upper 32 bits as appropriate...
      if (isUnsigned)     // Zero out top bits...
        BuildMI(*BB, IP, X86::MOV32ri, 1, DestReg+1).addImm(0);
      else                // Sign extend bottom half...
        BuildMI(*BB, IP, X86::SAR32ri, 2, DestReg+1).addReg(DestReg).addImm(31);
    }
    return;
  }

  // Special case long -> int ...
  if (SrcClass == cLong && DestClass == cInt) {
    BuildMI(*BB, IP, X86::MOV32rr, 1, DestReg).addReg(SrcReg);
    return;
  }
  
  // Handle cast of LARGER int to SMALLER int using a move to EAX followed by a
  // move out of AX or AL.
  if ((SrcClass <= cInt || SrcClass == cLong) && DestClass <= cInt
      && SrcClass > DestClass) {
    static const unsigned AReg[] = { X86::AL, X86::AX, X86::EAX, 0, X86::EAX };
    BuildMI(*BB, IP, RegRegMove[SrcClass], 1, AReg[SrcClass]).addReg(SrcReg);
    BuildMI(*BB, IP, RegRegMove[DestClass], 1, DestReg).addReg(AReg[DestClass]);
    return;
  }

  // Handle casts from integer to floating point now...
  if (DestClass == cFP) {
    // Promote the integer to a type supported by FLD.  We do this because there
    // are no unsigned FLD instructions, so we must promote an unsigned value to
    // a larger signed value, then use FLD on the larger value.
    //
    const Type *PromoteType = 0;
    unsigned PromoteOpcode;
    unsigned RealDestReg = DestReg;
    switch (SrcTy->getPrimitiveID()) {
    case Type::BoolTyID:
    case Type::SByteTyID:
      // We don't have the facilities for directly loading byte sized data from
      // memory (even signed).  Promote it to 16 bits.
      PromoteType = Type::ShortTy;
      PromoteOpcode = X86::MOVSX16rr8;
      break;
    case Type::UByteTyID:
      PromoteType = Type::ShortTy;
      PromoteOpcode = X86::MOVZX16rr8;
      break;
    case Type::UShortTyID:
      PromoteType = Type::IntTy;
      PromoteOpcode = X86::MOVZX32rr16;
      break;
    case Type::UIntTyID: {
      // Make a 64 bit temporary... and zero out the top of it...
      unsigned TmpReg = makeAnotherReg(Type::LongTy);
      BuildMI(*BB, IP, X86::MOV32rr, 1, TmpReg).addReg(SrcReg);
      BuildMI(*BB, IP, X86::MOV32ri, 1, TmpReg+1).addImm(0);
      SrcTy = Type::LongTy;
      SrcClass = cLong;
      SrcReg = TmpReg;
      break;
    }
    case Type::ULongTyID:
      // Don't fild into the read destination.
      DestReg = makeAnotherReg(Type::DoubleTy);
      break;
    default:  // No promotion needed...
      break;
    }
    
    if (PromoteType) {
      unsigned TmpReg = makeAnotherReg(PromoteType);
      unsigned Opc = SrcTy->isSigned() ? X86::MOVSX16rr8 : X86::MOVZX16rr8;
      BuildMI(*BB, IP, Opc, 1, TmpReg).addReg(SrcReg);
      SrcTy = PromoteType;
      SrcClass = getClass(PromoteType);
      SrcReg = TmpReg;
    }

    // Spill the integer to memory and reload it from there...
    int FrameIdx =
      F->getFrameInfo()->CreateStackObject(SrcTy, TM.getTargetData());

    if (SrcClass == cLong) {
      addFrameReference(BuildMI(*BB, IP, X86::MOV32mr, 5),
                        FrameIdx).addReg(SrcReg);
      addFrameReference(BuildMI(*BB, IP, X86::MOV32mr, 5),
                        FrameIdx, 4).addReg(SrcReg+1);
    } else {
      static const unsigned Op1[] = { X86::MOV8mr, X86::MOV16mr, X86::MOV32mr };
      addFrameReference(BuildMI(*BB, IP, Op1[SrcClass], 5),
                        FrameIdx).addReg(SrcReg);
    }

    static const unsigned Op2[] =
      { 0/*byte*/, X86::FILD16m, X86::FILD32m, 0/*FP*/, X86::FILD64m };
    addFrameReference(BuildMI(*BB, IP, Op2[SrcClass], 5, DestReg), FrameIdx);

    // We need special handling for unsigned 64-bit integer sources.  If the
    // input number has the "sign bit" set, then we loaded it incorrectly as a
    // negative 64-bit number.  In this case, add an offset value.
    if (SrcTy == Type::ULongTy) {
      // Emit a test instruction to see if the dynamic input value was signed.
      BuildMI(*BB, IP, X86::TEST32rr, 2).addReg(SrcReg+1).addReg(SrcReg+1);

      // If the sign bit is set, get a pointer to an offset, otherwise get a
      // pointer to a zero.
      MachineConstantPool *CP = F->getConstantPool();
      unsigned Zero = makeAnotherReg(Type::IntTy);
      Constant *Null = Constant::getNullValue(Type::UIntTy);
      addConstantPoolReference(BuildMI(*BB, IP, X86::LEA32r, 5, Zero), 
                               CP->getConstantPoolIndex(Null));
      unsigned Offset = makeAnotherReg(Type::IntTy);
      Constant *OffsetCst = ConstantUInt::get(Type::UIntTy, 0x5f800000);
                                             
      addConstantPoolReference(BuildMI(*BB, IP, X86::LEA32r, 5, Offset),
                               CP->getConstantPoolIndex(OffsetCst));
      unsigned Addr = makeAnotherReg(Type::IntTy);
      BuildMI(*BB, IP, X86::CMOVS32rr, 2, Addr).addReg(Zero).addReg(Offset);

      // Load the constant for an add.  FIXME: this could make an 'fadd' that
      // reads directly from memory, but we don't support these yet.
      unsigned ConstReg = makeAnotherReg(Type::DoubleTy);
      addDirectMem(BuildMI(*BB, IP, X86::FLD32m, 4, ConstReg), Addr);

      BuildMI(*BB, IP, X86::FpADD, 2, RealDestReg)
                .addReg(ConstReg).addReg(DestReg);
    }

    return;
  }

  // Handle casts from floating point to integer now...
  if (SrcClass == cFP) {
    // Change the floating point control register to use "round towards zero"
    // mode when truncating to an integer value.
    //
    int CWFrameIdx = F->getFrameInfo()->CreateStackObject(2, 2);
    addFrameReference(BuildMI(*BB, IP, X86::FNSTCW16m, 4), CWFrameIdx);

    // Load the old value of the high byte of the control word...
    unsigned HighPartOfCW = makeAnotherReg(Type::UByteTy);
    addFrameReference(BuildMI(*BB, IP, X86::MOV8rm, 4, HighPartOfCW),
                      CWFrameIdx, 1);

    // Set the high part to be round to zero...
    addFrameReference(BuildMI(*BB, IP, X86::MOV8mi, 5),
                      CWFrameIdx, 1).addImm(12);

    // Reload the modified control word now...
    addFrameReference(BuildMI(*BB, IP, X86::FLDCW16m, 4), CWFrameIdx);
    
    // Restore the memory image of control word to original value
    addFrameReference(BuildMI(*BB, IP, X86::MOV8mr, 5),
                      CWFrameIdx, 1).addReg(HighPartOfCW);

    // We don't have the facilities for directly storing byte sized data to
    // memory.  Promote it to 16 bits.  We also must promote unsigned values to
    // larger classes because we only have signed FP stores.
    unsigned StoreClass  = DestClass;
    const Type *StoreTy  = DestTy;
    if (StoreClass == cByte || DestTy->isUnsigned())
      switch (StoreClass) {
      case cByte:  StoreTy = Type::ShortTy; StoreClass = cShort; break;
      case cShort: StoreTy = Type::IntTy;   StoreClass = cInt;   break;
      case cInt:   StoreTy = Type::LongTy;  StoreClass = cLong;  break;
      // The following treatment of cLong may not be perfectly right,
      // but it survives chains of casts of the form
      // double->ulong->double.
      case cLong:  StoreTy = Type::LongTy;  StoreClass = cLong;  break;
      default: assert(0 && "Unknown store class!");
      }

    // Spill the integer to memory and reload it from there...
    int FrameIdx =
      F->getFrameInfo()->CreateStackObject(StoreTy, TM.getTargetData());

    static const unsigned Op1[] =
      { 0, X86::FIST16m, X86::FIST32m, 0, X86::FISTP64m };
    addFrameReference(BuildMI(*BB, IP, Op1[StoreClass], 5),
                      FrameIdx).addReg(SrcReg);

    if (DestClass == cLong) {
      addFrameReference(BuildMI(*BB, IP, X86::MOV32rm, 4, DestReg), FrameIdx);
      addFrameReference(BuildMI(*BB, IP, X86::MOV32rm, 4, DestReg+1),
                        FrameIdx, 4);
    } else {
      static const unsigned Op2[] = { X86::MOV8rm, X86::MOV16rm, X86::MOV32rm };
      addFrameReference(BuildMI(*BB, IP, Op2[DestClass], 4, DestReg), FrameIdx);
    }

    // Reload the original control word now...
    addFrameReference(BuildMI(*BB, IP, X86::FLDCW16m, 4), CWFrameIdx);
    return;
  }

  // Anything we haven't handled already, we can't (yet) handle at all.
  assert(0 && "Unhandled cast instruction!");
  abort();
}

/// visitVANextInst - Implement the va_next instruction...
///
void ISel::visitVANextInst(VANextInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  unsigned Size;
  switch (I.getArgType()->getPrimitiveID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    Size = 4;
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
  case Type::DoubleTyID:
    Size = 8;
    break;
  }

  // Increment the VAList pointer...
  BuildMI(BB, X86::ADD32ri, 2, DestReg).addReg(VAList).addImm(Size);
}

void ISel::visitVAArgInst(VAArgInst &I) {
  unsigned VAList = getReg(I.getOperand(0));
  unsigned DestReg = getReg(I);

  switch (I.getType()->getPrimitiveID()) {
  default:
    std::cerr << I;
    assert(0 && "Error: bad type for va_next instruction!");
    return;
  case Type::PointerTyID:
  case Type::UIntTyID:
  case Type::IntTyID:
    addDirectMem(BuildMI(BB, X86::MOV32rm, 4, DestReg), VAList);
    break;
  case Type::ULongTyID:
  case Type::LongTyID:
    addDirectMem(BuildMI(BB, X86::MOV32rm, 4, DestReg), VAList);
    addRegOffset(BuildMI(BB, X86::MOV32rm, 4, DestReg+1), VAList, 4);
    break;
  case Type::DoubleTyID:
    addDirectMem(BuildMI(BB, X86::FLD64m, 4, DestReg), VAList);
    break;
  }
}

/// visitGetElementPtrInst - instruction-select GEP instructions
///
void ISel::visitGetElementPtrInst(GetElementPtrInst &I) {
  // If this GEP instruction will be folded into all of its users, we don't need
  // to explicitly calculate it!
  unsigned A, B, C, D;
  if (isGEPFoldable(0, I.getOperand(0), I.op_begin()+1, I.op_end(), A,B,C,D)) {
    // Check all of the users of the instruction to see if they are loads and
    // stores.
    bool AllWillFold = true;
    for (Value::use_iterator UI = I.use_begin(), E = I.use_end(); UI != E; ++UI)
      if (cast<Instruction>(*UI)->getOpcode() != Instruction::Load)
        if (cast<Instruction>(*UI)->getOpcode() != Instruction::Store ||
            cast<Instruction>(*UI)->getOperand(0) == &I) {
          AllWillFold = false;
          break;
        }

    // If the instruction is foldable, and will be folded into all users, don't
    // emit it!
    if (AllWillFold) return;
  }

  unsigned outputReg = getReg(I);
  emitGEPOperation(BB, BB->end(), I.getOperand(0),
                   I.op_begin()+1, I.op_end(), outputReg);
}

/// getGEPIndex - Inspect the getelementptr operands specified with GEPOps and
/// GEPTypes (the derived types being stepped through at each level).  On return
/// from this function, if some indexes of the instruction are representable as
/// an X86 lea instruction, the machine operands are put into the Ops
/// instruction and the consumed indexes are poped from the GEPOps/GEPTypes
/// lists.  Otherwise, GEPOps.size() is returned.  If this returns a an
/// addressing mode that only partially consumes the input, the BaseReg input of
/// the addressing mode must be left free.
///
/// Note that there is one fewer entry in GEPTypes than there is in GEPOps.
///
void ISel::getGEPIndex(MachineBasicBlock *MBB, MachineBasicBlock::iterator IP,
                       std::vector<Value*> &GEPOps,
                       std::vector<const Type*> &GEPTypes, unsigned &BaseReg,
                       unsigned &Scale, unsigned &IndexReg, unsigned &Disp) {
  const TargetData &TD = TM.getTargetData();

  // Clear out the state we are working with...
  BaseReg = 0;    // No base register
  Scale = 1;      // Unit scale
  IndexReg = 0;   // No index register
  Disp = 0;       // No displacement

  // While there are GEP indexes that can be folded into the current address,
  // keep processing them.
  while (!GEPTypes.empty()) {
    if (const StructType *StTy = dyn_cast<StructType>(GEPTypes.back())) {
      // It's a struct access.  CUI is the index into the structure,
      // which names the field. This index must have unsigned type.
      const ConstantUInt *CUI = cast<ConstantUInt>(GEPOps.back());
      
      // Use the TargetData structure to pick out what the layout of the
      // structure is in memory.  Since the structure index must be constant, we
      // can get its value and use it to find the right byte offset from the
      // StructLayout class's list of structure member offsets.
      Disp += TD.getStructLayout(StTy)->MemberOffsets[CUI->getValue()];
      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();
    } else {
      // It's an array or pointer access: [ArraySize x ElementType].
      const SequentialType *SqTy = cast<SequentialType>(GEPTypes.back());
      Value *idx = GEPOps.back();

      // idx is the index into the array.  Unlike with structure
      // indices, we may not know its actual value at code-generation
      // time.
      assert(idx->getType() == Type::LongTy && "Bad GEP array index!");

      // If idx is a constant, fold it into the offset.
      unsigned TypeSize = TD.getTypeSize(SqTy->getElementType());
      if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(idx)) {
        Disp += TypeSize*CSI->getValue();
      } else {
        // If the index reg is already taken, we can't handle this index.
        if (IndexReg) return;

        // If this is a size that we can handle, then add the index as 
        switch (TypeSize) {
        case 1: case 2: case 4: case 8:
          // These are all acceptable scales on X86.
          Scale = TypeSize;
          break;
        default:
          // Otherwise, we can't handle this scale
          return;
        }

        if (CastInst *CI = dyn_cast<CastInst>(idx))
          if (CI->getOperand(0)->getType() == Type::IntTy ||
              CI->getOperand(0)->getType() == Type::UIntTy)
            idx = CI->getOperand(0);

        IndexReg = MBB ? getReg(idx, MBB, IP) : 1;
      }

      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();
    }
  }

  // GEPTypes is empty, which means we have a single operand left.  See if we
  // can set it as the base register.
  //
  // FIXME: When addressing modes are more powerful/correct, we could load
  // global addresses directly as 32-bit immediates.
  assert(BaseReg == 0);
  BaseReg = MBB ? getReg(GEPOps[0], MBB, IP) : 1;
  GEPOps.pop_back();        // Consume the last GEP operand
}


/// isGEPFoldable - Return true if the specified GEP can be completely
/// folded into the addressing mode of a load/store or lea instruction.
bool ISel::isGEPFoldable(MachineBasicBlock *MBB,
                         Value *Src, User::op_iterator IdxBegin,
                         User::op_iterator IdxEnd, unsigned &BaseReg,
                         unsigned &Scale, unsigned &IndexReg, unsigned &Disp) {
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Src))
    Src = CPR->getValue();

  std::vector<Value*> GEPOps;
  GEPOps.resize(IdxEnd-IdxBegin+1);
  GEPOps[0] = Src;
  std::copy(IdxBegin, IdxEnd, GEPOps.begin()+1);
  
  std::vector<const Type*> GEPTypes;
  GEPTypes.assign(gep_type_begin(Src->getType(), IdxBegin, IdxEnd),
                  gep_type_end(Src->getType(), IdxBegin, IdxEnd));

  MachineBasicBlock::iterator IP;
  if (MBB) IP = MBB->end();
  getGEPIndex(MBB, IP, GEPOps, GEPTypes, BaseReg, Scale, IndexReg, Disp);

  // We can fold it away iff the getGEPIndex call eliminated all operands.
  return GEPOps.empty();
}

void ISel::emitGEPOperation(MachineBasicBlock *MBB,
                            MachineBasicBlock::iterator IP,
                            Value *Src, User::op_iterator IdxBegin,
                            User::op_iterator IdxEnd, unsigned TargetReg) {
  const TargetData &TD = TM.getTargetData();
  if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Src))
    Src = CPR->getValue();

  std::vector<Value*> GEPOps;
  GEPOps.resize(IdxEnd-IdxBegin+1);
  GEPOps[0] = Src;
  std::copy(IdxBegin, IdxEnd, GEPOps.begin()+1);
  
  std::vector<const Type*> GEPTypes;
  GEPTypes.assign(gep_type_begin(Src->getType(), IdxBegin, IdxEnd),
                  gep_type_end(Src->getType(), IdxBegin, IdxEnd));

  // Keep emitting instructions until we consume the entire GEP instruction.
  while (!GEPOps.empty()) {
    unsigned OldSize = GEPOps.size();
    unsigned BaseReg, Scale, IndexReg, Disp;
    getGEPIndex(MBB, IP, GEPOps, GEPTypes, BaseReg, Scale, IndexReg, Disp);
    
    if (GEPOps.size() != OldSize) {
      // getGEPIndex consumed some of the input.  Build an LEA instruction here.
      unsigned NextTarget = 0;
      if (!GEPOps.empty()) {
        assert(BaseReg == 0 &&
           "getGEPIndex should have left the base register open for chaining!");
        NextTarget = BaseReg = makeAnotherReg(Type::UIntTy);
      }

      if (IndexReg == 0 && Disp == 0)
        BuildMI(*MBB, IP, X86::MOV32rr, 1, TargetReg).addReg(BaseReg);
      else
        addFullAddress(BuildMI(*MBB, IP, X86::LEA32r, 5, TargetReg),
                       BaseReg, Scale, IndexReg, Disp);
      --IP;
      TargetReg = NextTarget;
    } else if (GEPTypes.empty()) {
      // The getGEPIndex operation didn't want to build an LEA.  Check to see if
      // all operands are consumed but the base pointer.  If so, just load it
      // into the register.
      if (GlobalValue *GV = dyn_cast<GlobalValue>(GEPOps[0])) {
        BuildMI(*MBB, IP, X86::MOV32ri, 1, TargetReg).addGlobalAddress(GV);
      } else {
        unsigned BaseReg = getReg(GEPOps[0], MBB, IP);
        BuildMI(*MBB, IP, X86::MOV32rr, 1, TargetReg).addReg(BaseReg);
      }
      break;                // we are now done

    } else {
      // It's an array or pointer access: [ArraySize x ElementType].
      const SequentialType *SqTy = cast<SequentialType>(GEPTypes.back());
      Value *idx = GEPOps.back();
      GEPOps.pop_back();        // Consume a GEP operand
      GEPTypes.pop_back();

      // idx is the index into the array.  Unlike with structure
      // indices, we may not know its actual value at code-generation
      // time.
      assert(idx->getType() == Type::LongTy && "Bad GEP array index!");

      // Most GEP instructions use a [cast (int/uint) to LongTy] as their
      // operand on X86.  Handle this case directly now...
      if (CastInst *CI = dyn_cast<CastInst>(idx))
        if (CI->getOperand(0)->getType() == Type::IntTy ||
            CI->getOperand(0)->getType() == Type::UIntTy)
          idx = CI->getOperand(0);

      // We want to add BaseReg to(idxReg * sizeof ElementType). First, we
      // must find the size of the pointed-to type (Not coincidentally, the next
      // type is the type of the elements in the array).
      const Type *ElTy = SqTy->getElementType();
      unsigned elementSize = TD.getTypeSize(ElTy);

      // If idxReg is a constant, we don't need to perform the multiply!
      if (ConstantSInt *CSI = dyn_cast<ConstantSInt>(idx)) {
        if (!CSI->isNullValue()) {
          unsigned Offset = elementSize*CSI->getValue();
          unsigned Reg = makeAnotherReg(Type::UIntTy);
          BuildMI(*MBB, IP, X86::ADD32ri, 2, TargetReg)
                                .addReg(Reg).addImm(Offset);
          --IP;            // Insert the next instruction before this one.
          TargetReg = Reg; // Codegen the rest of the GEP into this
        }
      } else if (elementSize == 1) {
        // If the element size is 1, we don't have to multiply, just add
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned Reg = makeAnotherReg(Type::UIntTy);
        BuildMI(*MBB, IP, X86::ADD32rr, 2,TargetReg).addReg(Reg).addReg(idxReg);
        --IP;            // Insert the next instruction before this one.
        TargetReg = Reg; // Codegen the rest of the GEP into this
      } else {
        unsigned idxReg = getReg(idx, MBB, IP);
        unsigned OffsetReg = makeAnotherReg(Type::UIntTy);

        // Make sure we can back the iterator up to point to the first
        // instruction emitted.
        MachineBasicBlock::iterator BeforeIt = IP;
        if (IP == MBB->begin())
          BeforeIt = MBB->end();
        else
          --BeforeIt;
        doMultiplyConst(MBB, IP, OffsetReg, Type::IntTy, idxReg, elementSize);

        // Emit an ADD to add OffsetReg to the basePtr.
        unsigned Reg = makeAnotherReg(Type::UIntTy);
        BuildMI(*MBB, IP, X86::ADD32rr, 2, TargetReg)
                          .addReg(Reg).addReg(OffsetReg);

        // Step to the first instruction of the multiply.
        if (BeforeIt == MBB->end())
          IP = MBB->begin();
        else
          IP = ++BeforeIt;

        TargetReg = Reg; // Codegen the rest of the GEP into this
      }
    }
  }
}


/// visitAllocaInst - If this is a fixed size alloca, allocate space from the
/// frame manager, otherwise do it the hard way.
///
void ISel::visitAllocaInst(AllocaInst &I) {
  // Find the data size of the alloca inst's getAllocatedType.
  const Type *Ty = I.getAllocatedType();
  unsigned TySize = TM.getTargetData().getTypeSize(Ty);

  // If this is a fixed size alloca in the entry block for the function,
  // statically stack allocate the space.
  //
  if (ConstantUInt *CUI = dyn_cast<ConstantUInt>(I.getArraySize())) {
    if (I.getParent() == I.getParent()->getParent()->begin()) {
      TySize *= CUI->getValue();   // Get total allocated size...
      unsigned Alignment = TM.getTargetData().getTypeAlignment(Ty);
      
      // Create a new stack object using the frame manager...
      int FrameIdx = F->getFrameInfo()->CreateStackObject(TySize, Alignment);
      addFrameReference(BuildMI(BB, X86::LEA32r, 5, getReg(I)), FrameIdx);
      return;
    }
  }
  
  // Create a register to hold the temporary result of multiplying the type size
  // constant by the variable amount.
  unsigned TotalSizeReg = makeAnotherReg(Type::UIntTy);
  unsigned SrcReg1 = getReg(I.getArraySize());
  
  // TotalSizeReg = mul <numelements>, <TypeSize>
  MachineBasicBlock::iterator MBBI = BB->end();
  doMultiplyConst(BB, MBBI, TotalSizeReg, Type::UIntTy, SrcReg1, TySize);

  // AddedSize = add <TotalSizeReg>, 15
  unsigned AddedSizeReg = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::ADD32ri, 2, AddedSizeReg).addReg(TotalSizeReg).addImm(15);

  // AlignedSize = and <AddedSize>, ~15
  unsigned AlignedSize = makeAnotherReg(Type::UIntTy);
  BuildMI(BB, X86::AND32ri, 2, AlignedSize).addReg(AddedSizeReg).addImm(~15);
  
  // Subtract size from stack pointer, thereby allocating some space.
  BuildMI(BB, X86::SUB32rr, 2, X86::ESP).addReg(X86::ESP).addReg(AlignedSize);

  // Put a pointer to the space into the result register, by copying
  // the stack pointer.
  BuildMI(BB, X86::MOV32rr, 1, getReg(I)).addReg(X86::ESP);

  // Inform the Frame Information that we have just allocated a variable-sized
  // object.
  F->getFrameInfo()->CreateVariableSizedObject();
}

/// visitMallocInst - Malloc instructions are code generated into direct calls
/// to the library malloc.
///
void ISel::visitMallocInst(MallocInst &I) {
  unsigned AllocSize = TM.getTargetData().getTypeSize(I.getAllocatedType());
  unsigned Arg;

  if (ConstantUInt *C = dyn_cast<ConstantUInt>(I.getOperand(0))) {
    Arg = getReg(ConstantUInt::get(Type::UIntTy, C->getValue() * AllocSize));
  } else {
    Arg = makeAnotherReg(Type::UIntTy);
    unsigned Op0Reg = getReg(I.getOperand(0));
    MachineBasicBlock::iterator MBBI = BB->end();
    doMultiplyConst(BB, MBBI, Arg, Type::UIntTy, Op0Reg, AllocSize);
  }

  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(Arg, Type::UIntTy));
  MachineInstr *TheCall = BuildMI(X86::CALLpcrel32,
                                  1).addExternalSymbol("malloc", true);
  doCall(ValueRecord(getReg(I), I.getType()), TheCall, Args);
}


/// visitFreeInst - Free instructions are code gen'd to call the free libc
/// function.
///
void ISel::visitFreeInst(FreeInst &I) {
  std::vector<ValueRecord> Args;
  Args.push_back(ValueRecord(I.getOperand(0)));
  MachineInstr *TheCall = BuildMI(X86::CALLpcrel32,
                                  1).addExternalSymbol("free", true);
  doCall(ValueRecord(0, Type::VoidTy), TheCall, Args);
}
   
/// createX86SimpleInstructionSelector - This pass converts an LLVM function
/// into a machine code representation is a very simple peep-hole fashion.  The
/// generated code sucks but the implementation is nice and simple.
///
FunctionPass *llvm::createX86SimpleInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}
