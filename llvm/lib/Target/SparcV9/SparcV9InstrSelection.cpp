//===-- SparcV9InstrSelection.cpp -------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
//  BURS instruction selection for SPARC V9 architecture.      
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/InstrForest.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/InstrSelectionSupport.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "MachineInstrAnnot.h"
#include "SparcV9InstrSelectionSupport.h"
#include "SparcV9Internals.h"
#include "SparcV9RegClassInfo.h"
#include "SparcV9RegInfo.h"
#include "Support/MathExtras.h"
#include <algorithm>
#include <cmath>

namespace llvm {

static inline void Add3OperandInstr(unsigned Opcode, InstructionNode* Node,
                                    std::vector<MachineInstr*>& mvec) {
  mvec.push_back(BuildMI(Opcode, 3).addReg(Node->leftChild()->getValue())
                                   .addReg(Node->rightChild()->getValue())
                                   .addRegDef(Node->getValue()));
}


//---------------------------------------------------------------------------
// Function: FoldGetElemChain
// 
// Purpose:
//   Fold a chain of GetElementPtr instructions containing only
//   constant offsets into an equivalent (Pointer, IndexVector) pair.
//   Returns the pointer Value, and stores the resulting IndexVector
//   in argument chainIdxVec. This is a helper function for
//   FoldConstantIndices that does the actual folding. 
//---------------------------------------------------------------------------


// Check for a constant 0.
static inline bool
IsZero(Value* idx)
{
  return (idx == ConstantSInt::getNullValue(idx->getType()));
}

static Value*
FoldGetElemChain(InstrTreeNode* ptrNode, std::vector<Value*>& chainIdxVec,
                 bool lastInstHasLeadingNonZero)
{
  InstructionNode* gepNode = dyn_cast<InstructionNode>(ptrNode);
  GetElementPtrInst* gepInst =
    dyn_cast_or_null<GetElementPtrInst>(gepNode ? gepNode->getInstruction() :0);

  // ptr value is not computed in this tree or ptr value does not come from GEP
  // instruction
  if (gepInst == NULL)
    return NULL;

  // Return NULL if we don't fold any instructions in.
  Value* ptrVal = NULL;

  // Now chase the chain of getElementInstr instructions, if any.
  // Check for any non-constant indices and stop there.
  // Also, stop if the first index of child is a non-zero array index
  // and the last index of the current node is a non-array index:
  // in that case, a non-array declared type is being accessed as an array
  // which is not type-safe, but could be legal.
  // 
  InstructionNode* ptrChild = gepNode;
  while (ptrChild && (ptrChild->getOpLabel() == Instruction::GetElementPtr ||
                      ptrChild->getOpLabel() == GetElemPtrIdx))
  {
    // Child is a GetElemPtr instruction
    gepInst = cast<GetElementPtrInst>(ptrChild->getValue());
    User::op_iterator OI, firstIdx = gepInst->idx_begin();
    User::op_iterator lastIdx = gepInst->idx_end();
    bool allConstantOffsets = true;

    // The first index of every GEP must be an array index.
    assert((*firstIdx)->getType() == Type::LongTy &&
           "INTERNAL ERROR: Structure index for a pointer type!");

    // If the last instruction had a leading non-zero index, check if the
    // current one references a sequential (i.e., indexable) type.
    // If not, the code is not type-safe and we would create an illegal GEP
    // by folding them, so don't fold any more instructions.
    // 
    if (lastInstHasLeadingNonZero)
      if (! isa<SequentialType>(gepInst->getType()->getElementType()))
        break;   // cannot fold in any preceding getElementPtr instrs.

    // Check that all offsets are constant for this instruction
    for (OI = firstIdx; allConstantOffsets && OI != lastIdx; ++OI)
      allConstantOffsets = isa<ConstantInt>(*OI);

    if (allConstantOffsets) {
      // Get pointer value out of ptrChild.
      ptrVal = gepInst->getPointerOperand();

      // Insert its index vector at the start, skipping any leading [0]
      // Remember the old size to check if anything was inserted.
      unsigned oldSize = chainIdxVec.size();
      int firstIsZero = IsZero(*firstIdx);
      chainIdxVec.insert(chainIdxVec.begin(), firstIdx + firstIsZero, lastIdx);

      // Remember if it has leading zero index: it will be discarded later.
      if (oldSize < chainIdxVec.size())
        lastInstHasLeadingNonZero = !firstIsZero;

      // Mark the folded node so no code is generated for it.
      ((InstructionNode*) ptrChild)->markFoldedIntoParent();

      // Get the previous GEP instruction and continue trying to fold
      ptrChild = dyn_cast<InstructionNode>(ptrChild->leftChild());
    } else // cannot fold this getElementPtr instr. or any preceding ones
      break;
  }

  // If the first getElementPtr instruction had a leading [0], add it back.
  // Note that this instruction is the *last* one that was successfully
  // folded *and* contributed any indices, in the loop above.
  // 
  if (ptrVal && ! lastInstHasLeadingNonZero) 
    chainIdxVec.insert(chainIdxVec.begin(), ConstantSInt::get(Type::LongTy,0));

  return ptrVal;
}


//---------------------------------------------------------------------------
// Function: GetGEPInstArgs
// 
// Purpose:
//   Helper function for GetMemInstArgs that handles the final getElementPtr
//   instruction used by (or same as) the memory operation.
//   Extracts the indices of the current instruction and tries to fold in
//   preceding ones if all indices of the current one are constant.
//---------------------------------------------------------------------------

static Value *
GetGEPInstArgs(InstructionNode* gepNode,
               std::vector<Value*>& idxVec,
               bool& allConstantIndices)
{
  allConstantIndices = true;
  GetElementPtrInst* gepI = cast<GetElementPtrInst>(gepNode->getInstruction());

  // Default pointer is the one from the current instruction.
  Value* ptrVal = gepI->getPointerOperand();
  InstrTreeNode* ptrChild = gepNode->leftChild(); 

  // Extract the index vector of the GEP instruction.
  // If all indices are constant and first index is zero, try to fold
  // in preceding GEPs with all constant indices.
  for (User::op_iterator OI=gepI->idx_begin(),  OE=gepI->idx_end();
       allConstantIndices && OI != OE; ++OI)
    if (! isa<Constant>(*OI))
      allConstantIndices = false;     // note: this also terminates loop!

  // If we have only constant indices, fold chains of constant indices
  // in this and any preceding GetElemPtr instructions.
  bool foldedGEPs = false;
  bool leadingNonZeroIdx = gepI && ! IsZero(*gepI->idx_begin());
  if (allConstantIndices)
    if (Value* newPtr = FoldGetElemChain(ptrChild, idxVec, leadingNonZeroIdx)) {
      ptrVal = newPtr;
      foldedGEPs = true;
    }

  // Append the index vector of the current instruction.
  // Skip the leading [0] index if preceding GEPs were folded into this.
  idxVec.insert(idxVec.end(),
                gepI->idx_begin() + (foldedGEPs && !leadingNonZeroIdx),
                gepI->idx_end());

  return ptrVal;
}

//---------------------------------------------------------------------------
// Function: GetMemInstArgs
// 
// Purpose:
//   Get the pointer value and the index vector for a memory operation
//   (GetElementPtr, Load, or Store).  If all indices of the given memory
//   operation are constant, fold in constant indices in a chain of
//   preceding GetElementPtr instructions (if any), and return the
//   pointer value of the first instruction in the chain.
//   All folded instructions are marked so no code is generated for them.
//
// Return values:
//   Returns the pointer Value to use.
//   Returns the resulting IndexVector in idxVec.
//   Returns true/false in allConstantIndices if all indices are/aren't const.
//---------------------------------------------------------------------------

static Value*
GetMemInstArgs(InstructionNode* memInstrNode,
               std::vector<Value*>& idxVec,
               bool& allConstantIndices)
{
  allConstantIndices = false;
  Instruction* memInst = memInstrNode->getInstruction();
  assert(idxVec.size() == 0 && "Need empty vector to return indices");

  // If there is a GetElemPtr instruction to fold in to this instr,
  // it must be in the left child for Load and GetElemPtr, and in the
  // right child for Store instructions.
  InstrTreeNode* ptrChild = (memInst->getOpcode() == Instruction::Store
                             ? memInstrNode->rightChild()
                             : memInstrNode->leftChild()); 
  
  // Default pointer is the one from the current instruction.
  Value* ptrVal = ptrChild->getValue(); 

  // Find the "last" GetElemPtr instruction: this one or the immediate child.
  // There will be none if this is a load or a store from a scalar pointer.
  InstructionNode* gepNode = NULL;
  if (isa<GetElementPtrInst>(memInst))
    gepNode = memInstrNode;
  else if (isa<InstructionNode>(ptrChild) && isa<GetElementPtrInst>(ptrVal)) {
    // Child of load/store is a GEP and memInst is its only use.
    // Use its indices and mark it as folded.
    gepNode = cast<InstructionNode>(ptrChild);
    gepNode->markFoldedIntoParent();
  }

  // If there are no indices, return the current pointer.
  // Else extract the pointer from the GEP and fold the indices.
  return gepNode ? GetGEPInstArgs(gepNode, idxVec, allConstantIndices)
                 : ptrVal;
}


//************************ Internal Functions ******************************/


static inline MachineOpCode 
ChooseBprInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode;
  
  Instruction* setCCInstr =
    ((InstructionNode*) instrNode->leftChild())->getInstruction();
  
  switch(setCCInstr->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::BRZ;   break;
  case Instruction::SetNE: opCode = V9::BRNZ;  break;
  case Instruction::SetLE: opCode = V9::BRLEZ; break;
  case Instruction::SetGE: opCode = V9::BRGEZ; break;
  case Instruction::SetLT: opCode = V9::BRLZ;  break;
  case Instruction::SetGT: opCode = V9::BRGZ;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    opCode = V9::INVALID_OPCODE;
    break; 
  }
  
  return opCode;
}


static inline MachineOpCode 
ChooseBpccInstruction(const InstructionNode* instrNode,
                      const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  bool isSigned = setCCInstr->getOperand(0)->getType()->isSigned();
  
  if (isSigned) {
    switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = V9::BE;  break;
    case Instruction::SetNE: opCode = V9::BNE; break;
    case Instruction::SetLE: opCode = V9::BLE; break;
    case Instruction::SetGE: opCode = V9::BGE; break;
    case Instruction::SetLT: opCode = V9::BL;  break;
    case Instruction::SetGT: opCode = V9::BG;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  } else {
    switch(setCCInstr->getOpcode())
    {
    case Instruction::SetEQ: opCode = V9::BE;   break;
    case Instruction::SetNE: opCode = V9::BNE;  break;
    case Instruction::SetLE: opCode = V9::BLEU; break;
    case Instruction::SetGE: opCode = V9::BCC;  break;
    case Instruction::SetLT: opCode = V9::BCS;  break;
    case Instruction::SetGT: opCode = V9::BGU;  break;
    default:
      assert(0 && "Unrecognized VM instruction!");
      break; 
    }
  }
  
  return opCode;
}

static inline MachineOpCode 
ChooseBFpccInstruction(const InstructionNode* instrNode,
                       const BinaryOperator* setCCInstr)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  switch(setCCInstr->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::FBE;  break;
  case Instruction::SetNE: opCode = V9::FBNE; break;
  case Instruction::SetLE: opCode = V9::FBLE; break;
  case Instruction::SetGE: opCode = V9::FBGE; break;
  case Instruction::SetLT: opCode = V9::FBL;  break;
  case Instruction::SetGT: opCode = V9::FBG;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    break; 
  }
  
  return opCode;
}


// Create a unique TmpInstruction for a boolean value,
// representing the CC register used by a branch on that value.
// For now, hack this using a little static cache of TmpInstructions.
// Eventually the entire BURG instruction selection should be put
// into a separate class that can hold such information.
// The static cache is not too bad because the memory for these
// TmpInstructions will be freed along with the rest of the Function anyway.
// 
static TmpInstruction*
GetTmpForCC(Value* boolVal, const Function *F, const Type* ccType,
            MachineCodeForInstruction& mcfi)
{
  typedef hash_map<const Value*, TmpInstruction*> BoolTmpCache;
  static BoolTmpCache boolToTmpCache;     // Map boolVal -> TmpInstruction*
  static const Function *lastFunction = 0;// Use to flush cache between funcs
  
  assert(boolVal->getType() == Type::BoolTy && "Weird but ok! Delete assert");
  
  if (lastFunction != F) {
    lastFunction = F;
    boolToTmpCache.clear();
  }
  
  // Look for tmpI and create a new one otherwise.  The new value is
  // directly written to map using the ref returned by operator[].
  TmpInstruction*& tmpI = boolToTmpCache[boolVal];
  if (tmpI == NULL)
    tmpI = new TmpInstruction(mcfi, ccType, boolVal);
  
  return tmpI;
}


static inline MachineOpCode 
ChooseBccInstruction(const InstructionNode* instrNode,
                     const Type*& setCCType)
{
  InstructionNode* setCCNode = (InstructionNode*) instrNode->leftChild();
  assert(setCCNode->getOpLabel() == SetCCOp);
  BinaryOperator* setCCInstr =cast<BinaryOperator>(setCCNode->getInstruction());
  setCCType = setCCInstr->getOperand(0)->getType();
  
  if (setCCType->isFloatingPoint())
    return ChooseBFpccInstruction(instrNode, setCCInstr);
  else
    return ChooseBpccInstruction(instrNode, setCCInstr);
}


// WARNING: since this function has only one caller, it always returns
// the opcode that expects an immediate and a register. If this function
// is ever used in cases where an opcode that takes two registers is required,
// then modify this function and use convertOpcodeFromRegToImm() where required.
//
// It will be necessary to expand convertOpcodeFromRegToImm() to handle the
// new cases of opcodes.
static inline MachineOpCode 
ChooseMovFpcciInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  switch(instrNode->getInstruction()->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::MOVFEi;  break;
  case Instruction::SetNE: opCode = V9::MOVFNEi; break;
  case Instruction::SetLE: opCode = V9::MOVFLEi; break;
  case Instruction::SetGE: opCode = V9::MOVFGEi; break;
  case Instruction::SetLT: opCode = V9::MOVFLi;  break;
  case Instruction::SetGT: opCode = V9::MOVFGi;  break;
  default:
    assert(0 && "Unrecognized VM instruction!");
    break; 
  }
  
  return opCode;
}


// ChooseMovpcciForSetCC -- Choose a conditional-move instruction
// based on the type of SetCC operation.
// 
// WARNING: since this function has only one caller, it always returns
// the opcode that expects an immediate and a register. If this function
// is ever used in cases where an opcode that takes two registers is required,
// then modify this function and use convertOpcodeFromRegToImm() where required.
//
// It will be necessary to expand convertOpcodeFromRegToImm() to handle the
// new cases of opcodes.
// 
static MachineOpCode
ChooseMovpcciForSetCC(const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;

  const Type* opType = instrNode->leftChild()->getValue()->getType();
  assert(opType->isIntegral() || isa<PointerType>(opType));
  bool noSign = opType->isUnsigned() || isa<PointerType>(opType);
  
  switch(instrNode->getInstruction()->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::MOVEi;                        break;
  case Instruction::SetLE: opCode = noSign? V9::MOVLEUi : V9::MOVLEi; break;
  case Instruction::SetGE: opCode = noSign? V9::MOVCCi  : V9::MOVGEi; break;
  case Instruction::SetLT: opCode = noSign? V9::MOVCSi  : V9::MOVLi;  break;
  case Instruction::SetGT: opCode = noSign? V9::MOVGUi  : V9::MOVGi;  break;
  case Instruction::SetNE: opCode = V9::MOVNEi;                       break;
  default: assert(0 && "Unrecognized LLVM instr!"); break; 
  }
  
  return opCode;
}


// ChooseMovpregiForSetCC -- Choose a conditional-move-on-register-value
// instruction based on the type of SetCC operation.  These instructions
// compare a register with 0 and perform the move is the comparison is true.
// 
// WARNING: like the previous function, this function it always returns
// the opcode that expects an immediate and a register.  See above.
// 
static MachineOpCode
ChooseMovpregiForSetCC(const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  switch(instrNode->getInstruction()->getOpcode())
  {
  case Instruction::SetEQ: opCode = V9::MOVRZi;  break;
  case Instruction::SetLE: opCode = V9::MOVRLEZi; break;
  case Instruction::SetGE: opCode = V9::MOVRGEZi; break;
  case Instruction::SetLT: opCode = V9::MOVRLZi;  break;
  case Instruction::SetGT: opCode = V9::MOVRGZi;  break;
  case Instruction::SetNE: opCode = V9::MOVRNZi; break;
  default: assert(0 && "Unrecognized VM instr!"); break; 
  }
  
  return opCode;
}


static inline MachineOpCode
ChooseConvertToFloatInstr(const TargetMachine& target,
                          OpLabel vopCode, const Type* opType)
{
  assert((vopCode == ToFloatTy || vopCode == ToDoubleTy) &&
         "Unrecognized convert-to-float opcode!");
  assert((opType->isIntegral() || opType->isFloatingPoint() ||
          isa<PointerType>(opType))
         && "Trying to convert a non-scalar type to FLOAT/DOUBLE?");

  MachineOpCode opCode = V9::INVALID_OPCODE;

  unsigned opSize = target.getTargetData().getTypeSize(opType);

  if (opType == Type::FloatTy)
    opCode = (vopCode == ToFloatTy? V9::NOP : V9::FSTOD);
  else if (opType == Type::DoubleTy)
    opCode = (vopCode == ToFloatTy? V9::FDTOS : V9::NOP);
  else if (opSize <= 4)
    opCode = (vopCode == ToFloatTy? V9::FITOS : V9::FITOD);
  else {
    assert(opSize == 8 && "Unrecognized type size > 4 and < 8!");
    opCode = (vopCode == ToFloatTy? V9::FXTOS : V9::FXTOD);
  }
  
  return opCode;
}

static inline MachineOpCode 
ChooseConvertFPToIntInstr(const TargetMachine& target,
                          const Type* destType, const Type* opType)
{
  assert((opType == Type::FloatTy || opType == Type::DoubleTy)
         && "This function should only be called for FLOAT or DOUBLE");
  assert((destType->isIntegral() || isa<PointerType>(destType))
         && "Trying to convert FLOAT/DOUBLE to a non-scalar type?");

  MachineOpCode opCode = V9::INVALID_OPCODE;

  unsigned destSize = target.getTargetData().getTypeSize(destType);

  if (destType == Type::UIntTy)
    assert(destType != Type::UIntTy && "Expand FP-to-uint beforehand.");
  else if (destSize <= 4)
    opCode = (opType == Type::FloatTy)? V9::FSTOI : V9::FDTOI;
  else {
    assert(destSize == 8 && "Unrecognized type size > 4 and < 8!");
    opCode = (opType == Type::FloatTy)? V9::FSTOX : V9::FDTOX;
  }

  return opCode;
}

static MachineInstr*
CreateConvertFPToIntInstr(const TargetMachine& target,
                          Value* srcVal,
                          Value* destVal,
                          const Type* destType)
{
  MachineOpCode opCode = ChooseConvertFPToIntInstr(target, destType,
                                                   srcVal->getType());
  assert(opCode != V9::INVALID_OPCODE && "Expected to need conversion!");
  return BuildMI(opCode, 2).addReg(srcVal).addRegDef(destVal);
}

// CreateCodeToConvertFloatToInt: Convert FP value to signed or unsigned integer
// The FP value must be converted to the dest type in an FP register,
// and the result is then copied from FP to int register via memory.
// SPARC does not have a float-to-uint conversion, only a float-to-int (fdtoi).
// Since fdtoi converts to signed integers, any FP value V between MAXINT+1
// and MAXUNSIGNED (i.e., 2^31 <= V <= 2^32-1) would be converted incorrectly.
// Therefore, for converting an FP value to uint32_t, we first need to convert
// to uint64_t and then to uint32_t.
// 
static void
CreateCodeToConvertFloatToInt(const TargetMachine& target,
                              Value* opVal,
                              Instruction* destI,
                              std::vector<MachineInstr*>& mvec,
                              MachineCodeForInstruction& mcfi)
{
  Function* F = destI->getParent()->getParent();

  // Create a temporary to represent the FP register into which the
  // int value will placed after conversion.  The type of this temporary
  // depends on the type of FP register to use: single-prec for a 32-bit
  // int or smaller; double-prec for a 64-bit int.
  // 
  size_t destSize = target.getTargetData().getTypeSize(destI->getType());

  const Type* castDestType = destI->getType(); // type for the cast instr result
  const Type* castDestRegType;          // type for cast instruction result reg
  TmpInstruction* destForCast;          // dest for cast instruction
  Instruction* fpToIntCopyDest = destI; // dest for fp-reg-to-int-reg copy instr

  // For converting an FP value to uint32_t, we first need to convert to
  // uint64_t and then to uint32_t, as explained above.
  if (destI->getType() == Type::UIntTy) {
    castDestType    = Type::ULongTy;       // use this instead of type of destI
    castDestRegType = Type::DoubleTy;      // uint64_t needs 64-bit FP register.
    destForCast     = new TmpInstruction(mcfi, castDestRegType, opVal);
    fpToIntCopyDest = new TmpInstruction(mcfi, castDestType, destForCast);
  }
  else {
    castDestRegType = (destSize > 4)? Type::DoubleTy : Type::FloatTy;
    destForCast = new TmpInstruction(mcfi, castDestRegType, opVal);
  }

  // Create the fp-to-int conversion instruction (src and dest regs are FP regs)
  mvec.push_back(CreateConvertFPToIntInstr(target, opVal, destForCast,
                                           castDestType));

  // Create the fpreg-to-intreg copy code
  target.getInstrInfo().CreateCodeToCopyFloatToInt(target, F, destForCast,
                                                   fpToIntCopyDest, mvec, mcfi);

  // Create the uint64_t to uint32_t conversion, if needed
  if (destI->getType() == Type::UIntTy)
    target.getInstrInfo().
      CreateZeroExtensionInstructions(target, F, fpToIntCopyDest, destI,
                                      /*numLowBits*/ 32, mvec, mcfi);
}


static inline MachineOpCode 
ChooseAddInstruction(const InstructionNode* instrNode)
{
  return ChooseAddInstructionByType(instrNode->getInstruction()->getType());
}


static inline MachineInstr* 
CreateMovFloatInstruction(const InstructionNode* instrNode,
                          const Type* resultType)
{
  return BuildMI((resultType == Type::FloatTy) ? V9::FMOVS : V9::FMOVD, 2)
                   .addReg(instrNode->leftChild()->getValue())
                   .addRegDef(instrNode->getValue());
}

static inline MachineInstr* 
CreateAddConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));
  
  // Cases worth optimizing are:
  // (1) Add with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FADD (1 vs 3 cycles).  There is no integer MOV.
  // 
  if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
    double dval = FPC->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode,
                                        instrNode->getInstruction()->getType());
  }
  
  return minstr;
}


static inline MachineOpCode 
ChooseSubInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  if (resultType->isInteger() || isa<PointerType>(resultType)) {
      opCode = V9::SUBr;
  } else {
    switch(resultType->getPrimitiveID())
    {
    case Type::FloatTyID:  opCode = V9::FSUBS; break;
    case Type::DoubleTyID: opCode = V9::FSUBD; break;
    default: assert(0 && "Invalid type for SUB instruction"); break; 
    }
  }

  return opCode;
}


static inline MachineInstr* 
CreateSubConstInstruction(const InstructionNode* instrNode)
{
  MachineInstr* minstr = NULL;
  
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  assert(isa<Constant>(constOp));
  
  // Cases worth optimizing are:
  // (1) Sub with 0 for float or double: use an FMOV of appropriate type,
  //	 instead of an FSUB (1 vs 3 cycles).  There is no integer MOV.
  // 
  if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
    double dval = FPC->getValue();
    if (dval == 0.0)
      minstr = CreateMovFloatInstruction(instrNode,
                                        instrNode->getInstruction()->getType());
  }
  
  return minstr;
}


static inline MachineOpCode 
ChooseFcmpInstruction(const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  Value* operand = ((InstrTreeNode*) instrNode->leftChild())->getValue();
  switch(operand->getType()->getPrimitiveID()) {
  case Type::FloatTyID:  opCode = V9::FCMPS; break;
  case Type::DoubleTyID: opCode = V9::FCMPD; break;
  default: assert(0 && "Invalid type for FCMP instruction"); break; 
  }
  
  return opCode;
}


// Assumes that leftArg and rightArg are both cast instructions.
//
static inline bool
BothFloatToDouble(const InstructionNode* instrNode)
{
  InstrTreeNode* leftArg = instrNode->leftChild();
  InstrTreeNode* rightArg = instrNode->rightChild();
  InstrTreeNode* leftArgArg = leftArg->leftChild();
  InstrTreeNode* rightArgArg = rightArg->leftChild();
  assert(leftArg->getValue()->getType() == rightArg->getValue()->getType());
  
  // Check if both arguments are floats cast to double
  return (leftArg->getValue()->getType() == Type::DoubleTy &&
          leftArgArg->getValue()->getType() == Type::FloatTy &&
          rightArgArg->getValue()->getType() == Type::FloatTy);
}


static inline MachineOpCode 
ChooseMulInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  if (resultType->isInteger())
    opCode = V9::MULXr;
  else
    switch(resultType->getPrimitiveID())
    {
    case Type::FloatTyID:  opCode = V9::FMULS; break;
    case Type::DoubleTyID: opCode = V9::FMULD; break;
    default: assert(0 && "Invalid type for MUL instruction"); break; 
    }
  
  return opCode;
}



static inline MachineInstr*
CreateIntNegInstruction(const TargetMachine& target,
                        Value* vreg)
{
  return BuildMI(V9::SUBr, 3).addMReg(target.getRegInfo().getZeroRegNum())
    .addReg(vreg).addRegDef(vreg);
}


// Create instruction sequence for any shift operation.
// SLL or SLLX on an operand smaller than the integer reg. size (64bits)
// requires a second instruction for explicit sign-extension.
// Note that we only have to worry about a sign-bit appearing in the
// most significant bit of the operand after shifting (e.g., bit 32 of
// Int or bit 16 of Short), so we do not have to worry about results
// that are as large as a normal integer register.
// 
static inline void
CreateShiftInstructions(const TargetMachine& target,
                        Function* F,
                        MachineOpCode shiftOpCode,
                        Value* argVal1,
                        Value* optArgVal2, /* Use optArgVal2 if not NULL */
                        unsigned optShiftNum, /* else use optShiftNum */
                        Instruction* destVal,
                        std::vector<MachineInstr*>& mvec,
                        MachineCodeForInstruction& mcfi)
{
  assert((optArgVal2 != NULL || optShiftNum <= 64) &&
         "Large shift sizes unexpected, but can be handled below: "
         "You need to check whether or not it fits in immed field below");
  
  // If this is a logical left shift of a type smaller than the standard
  // integer reg. size, we have to extend the sign-bit into upper bits
  // of dest, so we need to put the result of the SLL into a temporary.
  // 
  Value* shiftDest = destVal;
  unsigned opSize = target.getTargetData().getTypeSize(argVal1->getType());

  if ((shiftOpCode == V9::SLLr5 || shiftOpCode == V9::SLLXr6) && opSize < 8) {
    // put SLL result into a temporary
    shiftDest = new TmpInstruction(mcfi, argVal1, optArgVal2, "sllTmp");
  }
  
  MachineInstr* M = (optArgVal2 != NULL)
    ? BuildMI(shiftOpCode, 3).addReg(argVal1).addReg(optArgVal2)
                             .addReg(shiftDest, MachineOperand::Def)
    : BuildMI(shiftOpCode, 3).addReg(argVal1).addZImm(optShiftNum)
                             .addReg(shiftDest, MachineOperand::Def);
  mvec.push_back(M);
  
  if (shiftDest != destVal) {
    // extend the sign-bit of the result into all upper bits of dest
    assert(8*opSize <= 32 && "Unexpected type size > 4 and < IntRegSize?");
    target.getInstrInfo().
      CreateSignExtensionInstructions(target, F, shiftDest, destVal,
                                      8*opSize, mvec, mcfi);
  }
}


// Does not create any instructions if we cannot exploit constant to
// create a cheaper instruction.
// This returns the approximate cost of the instructions generated,
// which is used to pick the cheapest when both operands are constant.
static unsigned
CreateMulConstInstruction(const TargetMachine &target, Function* F,
                          Value* lval, Value* rval, Instruction* destVal,
                          std::vector<MachineInstr*>& mvec,
                          MachineCodeForInstruction& mcfi)
{
  /* Use max. multiply cost, viz., cost of MULX */
  unsigned cost = target.getInstrInfo().minLatency(V9::MULXr);
  unsigned firstNewInstr = mvec.size();
  
  Value* constOp = rval;
  if (! isa<Constant>(constOp))
    return cost;
  
  // Cases worth optimizing are:
  // (1) Multiply by 0 or 1 for any type: replace with copy (ADD or FMOV)
  // (2) Multiply by 2^x for integer types: replace with Shift
  // 
  const Type* resultType = destVal->getType();
  
  if (resultType->isInteger() || isa<PointerType>(resultType)) {
    bool isValidConst;
    int64_t C = (int64_t) target.getInstrInfo().ConvertConstantToIntType(target,
                                     constOp, constOp->getType(), isValidConst);
    if (isValidConst) {
      unsigned pow;
      bool needNeg = false;
      if (C < 0) {
        needNeg = true;
        C = -C;
      }
          
      if (C == 0 || C == 1) {
        cost = target.getInstrInfo().minLatency(V9::ADDr);
        unsigned Zero = target.getRegInfo().getZeroRegNum();
        MachineInstr* M;
        if (C == 0)
          M =BuildMI(V9::ADDr,3).addMReg(Zero).addMReg(Zero).addRegDef(destVal);
        else
          M = BuildMI(V9::ADDr,3).addReg(lval).addMReg(Zero).addRegDef(destVal);
        mvec.push_back(M);
      } else if (isPowerOf2(C, pow)) {
        unsigned opSize = target.getTargetData().getTypeSize(resultType);
        MachineOpCode opCode = (opSize <= 32)? V9::SLLr5 : V9::SLLXr6;
        CreateShiftInstructions(target, F, opCode, lval, NULL, pow,
                                destVal, mvec, mcfi);
      }
          
      if (mvec.size() > 0 && needNeg) {
        // insert <reg = SUB 0, reg> after the instr to flip the sign
        MachineInstr* M = CreateIntNegInstruction(target, destVal);
        mvec.push_back(M);
      }
    }
  } else {
    if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
      double dval = FPC->getValue();
      if (fabs(dval) == 1) {
        MachineOpCode opCode =  (dval < 0)
          ? (resultType == Type::FloatTy? V9::FNEGS : V9::FNEGD)
          : (resultType == Type::FloatTy? V9::FMOVS : V9::FMOVD);
        mvec.push_back(BuildMI(opCode,2).addReg(lval).addRegDef(destVal));
      } 
    }
  }
  
  if (firstNewInstr < mvec.size()) {
    cost = 0;
    for (unsigned i=firstNewInstr; i < mvec.size(); ++i)
      cost += target.getInstrInfo().minLatency(mvec[i]->getOpcode());
  }
  
  return cost;
}


// Does not create any instructions if we cannot exploit constant to
// create a cheaper instruction.
// 
static inline void
CreateCheapestMulConstInstruction(const TargetMachine &target,
                                  Function* F,
                                  Value* lval, Value* rval,
                                  Instruction* destVal,
                                  std::vector<MachineInstr*>& mvec,
                                  MachineCodeForInstruction& mcfi)
{
  Value* constOp;
  if (isa<Constant>(lval) && isa<Constant>(rval)) {
    // both operands are constant: evaluate and "set" in dest
    Constant* P = ConstantExpr::get(Instruction::Mul,
                                    cast<Constant>(lval),
                                    cast<Constant>(rval));
    target.getInstrInfo().CreateCodeToLoadConst(target,F,P,destVal,mvec,mcfi);
  }
  else if (isa<Constant>(rval))         // rval is constant, but not lval
    CreateMulConstInstruction(target, F, lval, rval, destVal, mvec, mcfi);
  else if (isa<Constant>(lval))         // lval is constant, but not rval
    CreateMulConstInstruction(target, F, lval, rval, destVal, mvec, mcfi);
  
  // else neither is constant
  return;
}

// Return NULL if we cannot exploit constant to create a cheaper instruction
static inline void
CreateMulInstruction(const TargetMachine &target, Function* F,
                     Value* lval, Value* rval, Instruction* destVal,
                     std::vector<MachineInstr*>& mvec,
                     MachineCodeForInstruction& mcfi,
                     MachineOpCode forceMulOp = -1)
{
  unsigned L = mvec.size();
  CreateCheapestMulConstInstruction(target,F, lval, rval, destVal, mvec, mcfi);
  if (mvec.size() == L) {
    // no instructions were added so create MUL reg, reg, reg.
    // Use FSMULD if both operands are actually floats cast to doubles.
    // Otherwise, use the default opcode for the appropriate type.
    MachineOpCode mulOp = ((forceMulOp != -1)
                           ? forceMulOp 
                           : ChooseMulInstructionByType(destVal->getType()));
    mvec.push_back(BuildMI(mulOp, 3).addReg(lval).addReg(rval)
                   .addRegDef(destVal));
  }
}


// Generate a divide instruction for Div or Rem.
// For Rem, this assumes that the operand type will be signed if the result
// type is signed.  This is correct because they must have the same sign.
// 
static inline MachineOpCode 
ChooseDivInstruction(TargetMachine &target,
                     const InstructionNode* instrNode)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  const Type* resultType = instrNode->getInstruction()->getType();
  
  if (resultType->isInteger())
    opCode = resultType->isSigned()? V9::SDIVXr : V9::UDIVXr;
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = V9::FDIVS; break;
      case Type::DoubleTyID: opCode = V9::FDIVD; break;
      default: assert(0 && "Invalid type for DIV instruction"); break; 
      }
  
  return opCode;
}


// Return if we cannot exploit constant to create a cheaper instruction
static void
CreateDivConstInstruction(TargetMachine &target,
                          const InstructionNode* instrNode,
                          std::vector<MachineInstr*>& mvec)
{
  Value* LHS  = instrNode->leftChild()->getValue();
  Value* constOp = ((InstrTreeNode*) instrNode->rightChild())->getValue();
  if (!isa<Constant>(constOp))
    return;

  Instruction* destVal = instrNode->getInstruction();
  unsigned ZeroReg = target.getRegInfo().getZeroRegNum();
  
  // Cases worth optimizing are:
  // (1) Divide by 1 for any type: replace with copy (ADD or FMOV)
  // (2) Divide by 2^x for integer types: replace with SR[L or A]{X}
  // 
  const Type* resultType = instrNode->getInstruction()->getType();
 
  if (resultType->isInteger()) {
    unsigned pow;
    bool isValidConst;
    int64_t C = (int64_t) target.getInstrInfo().ConvertConstantToIntType(target,
                                     constOp, constOp->getType(), isValidConst);
    if (isValidConst) {
      bool needNeg = false;
      if (C < 0) {
        needNeg = true;
        C = -C;
      }
      
      if (C == 1) {
        mvec.push_back(BuildMI(V9::ADDr, 3).addReg(LHS).addMReg(ZeroReg)
                       .addRegDef(destVal));
      } else if (isPowerOf2(C, pow)) {
        unsigned opCode;
        Value* shiftOperand;
        unsigned opSize = target.getTargetData().getTypeSize(resultType);

        if (resultType->isSigned()) {
          // For N / 2^k, if the operand N is negative,
          // we need to add (2^k - 1) before right-shifting by k, i.e.,
          // 
          //    (N / 2^k) = N >> k,               if N >= 0;
          //                (N + 2^k - 1) >> k,   if N < 0
          // 
          // If N is <= 32 bits, use:
          //    sra N, 31, t1           // t1 = ~0,         if N < 0,  0 else
          //    srl t1, 32-k, t2        // t2 = 2^k - 1,    if N < 0,  0 else
          //    add t2, N, t3           // t3 = N + 2^k -1, if N < 0,  N else
	  //    sra t3, k, result       // result = N / 2^k
          // 
          // If N is 64 bits, use:
          //    srax N,  k-1,  t1       // t1 = sign bit in high k positions
          //    srlx t1, 64-k, t2       // t2 = 2^k - 1,    if N < 0,  0 else
          //    add t2, N, t3           // t3 = N + 2^k -1, if N < 0,  N else
	  //    sra t3, k, result       // result = N / 2^k
          //
          TmpInstruction *sraTmp, *srlTmp, *addTmp;
          MachineCodeForInstruction& mcfi
            = MachineCodeForInstruction::get(destVal);
          sraTmp = new TmpInstruction(mcfi, resultType, LHS, 0, "getSign");
          srlTmp = new TmpInstruction(mcfi, resultType, LHS, 0, "getPlus2km1");
          addTmp = new TmpInstruction(mcfi, resultType, LHS, srlTmp,"incIfNeg");

          // Create the SRA or SRAX instruction to get the sign bit
          mvec.push_back(BuildMI((opSize > 4)? V9::SRAXi6 : V9::SRAi5, 3)
                         .addReg(LHS)
                         .addSImm((resultType==Type::LongTy)? pow-1 : 31)
                         .addRegDef(sraTmp));

          // Create the SRL or SRLX instruction to get the sign bit
          mvec.push_back(BuildMI((opSize > 4)? V9::SRLXi6 : V9::SRLi5, 3)
                         .addReg(sraTmp)
                         .addSImm((resultType==Type::LongTy)? 64-pow : 32-pow)
                         .addRegDef(srlTmp));

          // Create the ADD instruction to add 2^pow-1 for negative values
          mvec.push_back(BuildMI(V9::ADDr, 3).addReg(LHS).addReg(srlTmp)
                         .addRegDef(addTmp));

          // Get the shift operand and "right-shift" opcode to do the divide
          shiftOperand = addTmp;
          opCode = (opSize > 4)? V9::SRAXi6 : V9::SRAi5;
        } else {
          // Get the shift operand and "right-shift" opcode to do the divide
          shiftOperand = LHS;
          opCode = (opSize > 4)? V9::SRLXi6 : V9::SRLi5;
        }

        // Now do the actual shift!
        mvec.push_back(BuildMI(opCode, 3).addReg(shiftOperand).addZImm(pow)
                       .addRegDef(destVal));
      }
          
      if (needNeg && (C == 1 || isPowerOf2(C, pow))) {
        // insert <reg = SUB 0, reg> after the instr to flip the sign
        mvec.push_back(CreateIntNegInstruction(target, destVal));
      }
    }
  } else {
    if (ConstantFP *FPC = dyn_cast<ConstantFP>(constOp)) {
      double dval = FPC->getValue();
      if (fabs(dval) == 1) {
        unsigned opCode = 
          (dval < 0) ? (resultType == Type::FloatTy? V9::FNEGS : V9::FNEGD)
          : (resultType == Type::FloatTy? V9::FMOVS : V9::FMOVD);
              
        mvec.push_back(BuildMI(opCode, 2).addReg(LHS).addRegDef(destVal));
      } 
    }
  }
}


static void
CreateCodeForVariableSizeAlloca(const TargetMachine& target,
                                Instruction* result,
                                unsigned tsize,
                                Value* numElementsVal,
                                std::vector<MachineInstr*>& getMvec)
{
  Value* totalSizeVal;
  MachineInstr* M;
  MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(result);
  Function *F = result->getParent()->getParent();

  // Enforce the alignment constraints on the stack pointer at
  // compile time if the total size is a known constant.
  if (isa<Constant>(numElementsVal)) {
    bool isValid;
    int64_t numElem = (int64_t) target.getInstrInfo().
      ConvertConstantToIntType(target, numElementsVal,
                               numElementsVal->getType(), isValid);
    assert(isValid && "Unexpectedly large array dimension in alloca!");
    int64_t total = numElem * tsize;
    if (int extra= total % target.getFrameInfo().getStackFrameSizeAlignment())
      total += target.getFrameInfo().getStackFrameSizeAlignment() - extra;
    totalSizeVal = ConstantSInt::get(Type::IntTy, total);
  } else {
    // The size is not a constant.  Generate code to compute it and
    // code to pad the size for stack alignment.
    // Create a Value to hold the (constant) element size
    Value* tsizeVal = ConstantSInt::get(Type::IntTy, tsize);

    // Create temporary values to hold the result of MUL, SLL, SRL
    // To pad `size' to next smallest multiple of 16:
    //          size = (size + 15) & (-16 = 0xfffffffffffffff0)
    // 
    TmpInstruction* tmpProd = new TmpInstruction(mcfi,numElementsVal, tsizeVal);
    TmpInstruction* tmpAdd15= new TmpInstruction(mcfi,numElementsVal, tmpProd);
    TmpInstruction* tmpAndf0= new TmpInstruction(mcfi,numElementsVal, tmpAdd15);

    // Instruction 1: mul numElements, typeSize -> tmpProd
    // This will optimize the MUL as far as possible.
    CreateMulInstruction(target, F, numElementsVal, tsizeVal, tmpProd, getMvec,
                         mcfi, -1);

    // Instruction 2: andn tmpProd, 0x0f -> tmpAndn
    getMvec.push_back(BuildMI(V9::ADDi, 3).addReg(tmpProd).addSImm(15)
                      .addReg(tmpAdd15, MachineOperand::Def));

    // Instruction 3: add tmpAndn, 0x10 -> tmpAdd16
    getMvec.push_back(BuildMI(V9::ANDi, 3).addReg(tmpAdd15).addSImm(-16)
                      .addReg(tmpAndf0, MachineOperand::Def));

    totalSizeVal = tmpAndf0;
  }

  // Get the constant offset from SP for dynamically allocated storage
  // and create a temporary Value to hold it.
  MachineFunction& mcInfo = MachineFunction::get(F);
  bool growUp;
  ConstantSInt* dynamicAreaOffset =
    ConstantSInt::get(Type::IntTy,
                     target.getFrameInfo().getDynamicAreaOffset(mcInfo,growUp));
  assert(! growUp && "Has SPARC v9 stack frame convention changed?");

  unsigned SPReg = target.getRegInfo().getStackPointer();

  // Instruction 2: sub %sp, totalSizeVal -> %sp
  getMvec.push_back(BuildMI(V9::SUBr, 3).addMReg(SPReg).addReg(totalSizeVal)
                    .addMReg(SPReg,MachineOperand::Def));

  // Instruction 3: add %sp, frameSizeBelowDynamicArea -> result
  getMvec.push_back(BuildMI(V9::ADDr,3).addMReg(SPReg).addReg(dynamicAreaOffset)
                    .addRegDef(result));
}        


static void
CreateCodeForFixedSizeAlloca(const TargetMachine& target,
                             Instruction* result,
                             unsigned tsize,
                             unsigned numElements,
                             std::vector<MachineInstr*>& getMvec)
{
  assert(result && result->getParent() &&
         "Result value is not part of a function?");
  Function *F = result->getParent()->getParent();
  MachineFunction &mcInfo = MachineFunction::get(F);

  // If the alloca is of zero bytes (which is perfectly legal) we bump it up to
  // one byte.  This is unnecessary, but I really don't want to break any
  // fragile logic in this code.  FIXME.
  if (tsize == 0)
    tsize = 1;


  // Put the variable in the dynamically sized area of the frame if either:
  // (a) The offset is too large to use as an immediate in load/stores
  //     (check LDX because all load/stores have the same-size immed. field).
  // (b) The object is "large", so it could cause many other locals,
  //     spills, and temporaries to have large offsets.
  //     NOTE: We use LARGE = 8 * argSlotSize = 64 bytes.
  // You've gotta love having only 13 bits for constant offset values :-|.
  // 
  unsigned paddedSize;
  int offsetFromFP = mcInfo.getInfo()->computeOffsetforLocalVar(result,
                                                                paddedSize,
                                                         tsize * numElements);

  if (((int)paddedSize) > 8 * target.getFrameInfo().getSizeOfEachArgOnStack() ||
      ! target.getInstrInfo().constantFitsInImmedField(V9::LDXi,offsetFromFP)) {
    CreateCodeForVariableSizeAlloca(target, result, tsize, 
				    ConstantSInt::get(Type::IntTy,numElements),
				    getMvec);
    return;
  }
  
  // else offset fits in immediate field so go ahead and allocate it.
  offsetFromFP = mcInfo.getInfo()->allocateLocalVar(result, tsize *numElements);
  
  // Create a temporary Value to hold the constant offset.
  // This is needed because it may not fit in the immediate field.
  ConstantSInt* offsetVal = ConstantSInt::get(Type::IntTy, offsetFromFP);
  
  // Instruction 1: add %fp, offsetFromFP -> result
  unsigned FPReg = target.getRegInfo().getFramePointer();
  getMvec.push_back(BuildMI(V9::ADDr, 3).addMReg(FPReg).addReg(offsetVal)
                    .addRegDef(result));
}


//------------------------------------------------------------------------ 
// Function SetOperandsForMemInstr
//
// Choose addressing mode for the given load or store instruction.
// Use [reg+reg] if it is an indexed reference, and the index offset is
//		 not a constant or if it cannot fit in the offset field.
// Use [reg+offset] in all other cases.
// 
// This assumes that all array refs are "lowered" to one of these forms:
//	%x = load (subarray*) ptr, constant	; single constant offset
//	%x = load (subarray*) ptr, offsetVal	; single non-constant offset
// Generally, this should happen via strength reduction + LICM.
// Also, strength reduction should take care of using the same register for
// the loop index variable and an array index, when that is profitable.
//------------------------------------------------------------------------ 

static void
SetOperandsForMemInstr(unsigned Opcode,
                       std::vector<MachineInstr*>& mvec,
                       InstructionNode* vmInstrNode,
                       const TargetMachine& target)
{
  Instruction* memInst = vmInstrNode->getInstruction();
  // Index vector, ptr value, and flag if all indices are const.
  std::vector<Value*> idxVec;
  bool allConstantIndices;
  Value* ptrVal = GetMemInstArgs(vmInstrNode, idxVec, allConstantIndices);

  // Now create the appropriate operands for the machine instruction.
  // First, initialize so we default to storing the offset in a register.
  int64_t smallConstOffset = 0;
  Value* valueForRegOffset = NULL;
  MachineOperand::MachineOperandType offsetOpType =
    MachineOperand::MO_VirtualRegister;

  // Check if there is an index vector and if so, compute the
  // right offset for structures and for arrays 
  // 
  if (!idxVec.empty()) {
    const PointerType* ptrType = cast<PointerType>(ptrVal->getType());
      
    // If all indices are constant, compute the combined offset directly.
    if (allConstantIndices) {
      // Compute the offset value using the index vector. Create a
      // virtual reg. for it since it may not fit in the immed field.
      uint64_t offset = target.getTargetData().getIndexedOffset(ptrType,idxVec);
      valueForRegOffset = ConstantSInt::get(Type::LongTy, offset);
    } else {
      // There is at least one non-constant offset.  Therefore, this must
      // be an array ref, and must have been lowered to a single non-zero
      // offset.  (An extra leading zero offset, if any, can be ignored.)
      // Generate code sequence to compute address from index.
      // 
      bool firstIdxIsZero = IsZero(idxVec[0]);
      assert(idxVec.size() == 1U + firstIdxIsZero 
             && "Array refs must be lowered before Instruction Selection");

      Value* idxVal = idxVec[firstIdxIsZero];

      std::vector<MachineInstr*> mulVec;
      Instruction* addr =
        new TmpInstruction(MachineCodeForInstruction::get(memInst),
                           Type::ULongTy, memInst);

      // Get the array type indexed by idxVal, and compute its element size.
      // The call to getTypeSize() will fail if size is not constant.
      const Type* vecType = (firstIdxIsZero
                             ? GetElementPtrInst::getIndexedType(ptrType,
                                           std::vector<Value*>(1U, idxVec[0]),
                                           /*AllowCompositeLeaf*/ true)
                                 : ptrType);
      const Type* eltType = cast<SequentialType>(vecType)->getElementType();
      ConstantUInt* eltSizeVal = ConstantUInt::get(Type::ULongTy,
                                   target.getTargetData().getTypeSize(eltType));

      // CreateMulInstruction() folds constants intelligently enough.
      CreateMulInstruction(target, memInst->getParent()->getParent(),
                           idxVal,         /* lval, not likely to be const*/
                           eltSizeVal,     /* rval, likely to be constant */
                           addr,           /* result */
                           mulVec, MachineCodeForInstruction::get(memInst),
                           -1);

      assert(mulVec.size() > 0 && "No multiply code created?");
      mvec.insert(mvec.end(), mulVec.begin(), mulVec.end());
      
      valueForRegOffset = addr;
    }
  } else {
    offsetOpType = MachineOperand::MO_SignExtendedImmed;
    smallConstOffset = 0;
  }

  // For STORE:
  //   Operand 0 is value, operand 1 is ptr, operand 2 is offset
  // For LOAD or GET_ELEMENT_PTR,
  //   Operand 0 is ptr, operand 1 is offset, operand 2 is result.
  // 
  unsigned offsetOpNum, ptrOpNum;
  MachineInstr *MI;
  if (memInst->getOpcode() == Instruction::Store) {
    if (offsetOpType == MachineOperand::MO_VirtualRegister) {
      MI = BuildMI(Opcode, 3).addReg(vmInstrNode->leftChild()->getValue())
                             .addReg(ptrVal).addReg(valueForRegOffset);
    } else {
      Opcode = convertOpcodeFromRegToImm(Opcode);
      MI = BuildMI(Opcode, 3).addReg(vmInstrNode->leftChild()->getValue())
                             .addReg(ptrVal).addSImm(smallConstOffset);
    }
  } else {
    if (offsetOpType == MachineOperand::MO_VirtualRegister) {
      MI = BuildMI(Opcode, 3).addReg(ptrVal).addReg(valueForRegOffset)
                             .addRegDef(memInst);
    } else {
      Opcode = convertOpcodeFromRegToImm(Opcode);
      MI = BuildMI(Opcode, 3).addReg(ptrVal).addSImm(smallConstOffset)
                             .addRegDef(memInst);
    }
  }
  mvec.push_back(MI);
}


// 
// Substitute operand `operandNum' of the instruction in node `treeNode'
// in place of the use(s) of that instruction in node `parent'.
// Check both explicit and implicit operands!
// Also make sure to skip over a parent who:
// (1) is a list node in the Burg tree, or
// (2) itself had its results forwarded to its parent
// 
static void
ForwardOperand(InstructionNode* treeNode,
               InstrTreeNode*   parent,
               int operandNum)
{
  assert(treeNode && parent && "Invalid invocation of ForwardOperand");
  
  Instruction* unusedOp = treeNode->getInstruction();
  Value* fwdOp = unusedOp->getOperand(operandNum);

  // The parent itself may be a list node, so find the real parent instruction
  while (parent->getNodeType() != InstrTreeNode::NTInstructionNode)
    {
      parent = parent->parent();
      assert(parent && "ERROR: Non-instruction node has no parent in tree.");
    }
  InstructionNode* parentInstrNode = (InstructionNode*) parent;
  
  Instruction* userInstr = parentInstrNode->getInstruction();
  MachineCodeForInstruction &mvec = MachineCodeForInstruction::get(userInstr);

  // The parent's mvec would be empty if it was itself forwarded.
  // Recursively call ForwardOperand in that case...
  //
  if (mvec.size() == 0) {
    assert(parent->parent() != NULL &&
           "Parent could not have been forwarded, yet has no instructions?");
    ForwardOperand(treeNode, parent->parent(), operandNum);
  } else {
    for (unsigned i=0, N=mvec.size(); i < N; i++) {
      MachineInstr* minstr = mvec[i];
      for (unsigned i=0, numOps=minstr->getNumOperands(); i < numOps; ++i) {
        const MachineOperand& mop = minstr->getOperand(i);
        if (mop.getType() == MachineOperand::MO_VirtualRegister &&
            mop.getVRegValue() == unusedOp)
        {
          minstr->SetMachineOperandVal(i, MachineOperand::MO_VirtualRegister,
                                       fwdOp);
        }
      }
          
      for (unsigned i=0,numOps=minstr->getNumImplicitRefs(); i<numOps; ++i)
        if (minstr->getImplicitRef(i) == unusedOp)
          minstr->setImplicitRef(i, fwdOp);
    }
  }
}


inline bool
AllUsesAreBranches(const Instruction* setccI)
{
  for (Value::use_const_iterator UI=setccI->use_begin(), UE=setccI->use_end();
       UI != UE; ++UI)
    if (! isa<TmpInstruction>(*UI)     // ignore tmp instructions here
        && cast<Instruction>(*UI)->getOpcode() != Instruction::Br)
      return false;
  return true;
}

// Generate code for any intrinsic that needs a special code sequence
// instead of a regular call.  If not that kind of intrinsic, do nothing.
// Returns true if code was generated, otherwise false.
// 
static bool CodeGenIntrinsic(Intrinsic::ID iid, CallInst &callInstr,
                             TargetMachine &target,
                             std::vector<MachineInstr*>& mvec) {
  switch (iid) {
  default:
    assert(0 && "Unknown intrinsic function call should have been lowered!");
  case Intrinsic::vastart: {
    // Get the address of the first incoming vararg argument on the stack
    bool ignore;
    Function* func = cast<Function>(callInstr.getParent()->getParent());
    int numFixedArgs   = func->getFunctionType()->getNumParams();
    int fpReg          = target.getFrameInfo().getIncomingArgBaseRegNum();
    int argSize        = target.getFrameInfo().getSizeOfEachArgOnStack();
    int firstVarArgOff = numFixedArgs * argSize + target.getFrameInfo().
      getFirstIncomingArgOffset(MachineFunction::get(func), ignore);
    mvec.push_back(BuildMI(V9::ADDi, 3).addMReg(fpReg).addSImm(firstVarArgOff).
                   addRegDef(&callInstr));
    return true;
  }

  case Intrinsic::vaend:
    return true;                        // no-op on SparcV9

  case Intrinsic::vacopy:
    // Simple copy of current va_list (arg1) to new va_list (result)
    mvec.push_back(BuildMI(V9::ORr, 3).
                   addMReg(target.getRegInfo().getZeroRegNum()).
                   addReg(callInstr.getOperand(1)).
                   addRegDef(&callInstr));
    return true;
  }
}

//******************* Externally Visible Functions *************************/

//------------------------------------------------------------------------ 
// External Function: ThisIsAChainRule
//
// Purpose:
//   Check if a given BURG rule is a chain rule.
//------------------------------------------------------------------------ 

extern bool
ThisIsAChainRule(int eruleno)
{
  switch(eruleno)
    {
    case 111:	// stmt:  reg
    case 123:
    case 124:
    case 125:
    case 126:
    case 127:
    case 128:
    case 129:
    case 130:
    case 131:
    case 132:
    case 133:
    case 155:
    case 221:
    case 222:
    case 241:
    case 242:
    case 243:
    case 244:
    case 245:
    case 321:
      return true; break;

    default:
      return false; break;
    }
}


//------------------------------------------------------------------------ 
// External Function: GetInstructionsByRule
//
// Purpose:
//   Choose machine instructions for the SPARC according to the
//   patterns chosen by the BURG-generated parser.
//------------------------------------------------------------------------ 

void
GetInstructionsByRule(InstructionNode* subtreeRoot,
                      int ruleForNode,
                      short* nts,
                      TargetMachine &target,
                      std::vector<MachineInstr*>& mvec)
{
  bool checkCast = false;		// initialize here to use fall-through
  bool maskUnsignedResult = false;
  int nextRule;
  int forwardOperandNum = -1;
  unsigned allocaSize = 0;
  MachineInstr* M, *M2;
  unsigned L;
  bool foldCase = false;

  mvec.clear(); 
  
  // If the code for this instruction was folded into the parent (user),
  // then do nothing!
  if (subtreeRoot->isFoldedIntoParent())
    return;
  
  // 
  // Let's check for chain rules outside the switch so that we don't have
  // to duplicate the list of chain rule production numbers here again
  // 
  if (ThisIsAChainRule(ruleForNode)) {
    // Chain rules have a single nonterminal on the RHS.
    // Get the rule that matches the RHS non-terminal and use that instead.
    // 
    assert(nts[0] && ! nts[1]
           && "A chain rule should have only one RHS non-terminal!");
    nextRule = burm_rule(subtreeRoot->state, nts[0]);
    nts = burm_nts[nextRule];
    GetInstructionsByRule(subtreeRoot, nextRule, nts, target, mvec);
  } else {
    switch(ruleForNode) {
      case 1:   // stmt:   Ret
      case 2:   // stmt:   RetValue(reg)
      {         // NOTE: Prepass of register allocation is responsible
                //	 for moving return value to appropriate register.
                // Copy the return value to the required return register.
                // Mark the return Value as an implicit ref of the RET instr..
                // Mark the return-address register as a hidden virtual reg.
         	// Finally put a NOP in the delay slot.
        ReturnInst *returnInstr=cast<ReturnInst>(subtreeRoot->getInstruction());
        Value* retVal = returnInstr->getReturnValue();
        MachineCodeForInstruction& mcfi =
          MachineCodeForInstruction::get(returnInstr);

        // Create a hidden virtual reg to represent the return address register
        // used by the machine instruction but not represented in LLVM.
        // 
        Instruction* returnAddrTmp = new TmpInstruction(mcfi, returnInstr);

        MachineInstr* retMI = 
          BuildMI(V9::JMPLRETi, 3).addReg(returnAddrTmp).addSImm(8)
          .addMReg(target.getRegInfo().getZeroRegNum(), MachineOperand::Def);
      
        // If there is a value to return, we need to:
        // (a) Sign-extend the value if it is smaller than 8 bytes (reg size)
        // (b) Insert a copy to copy the return value to the appropriate reg.
        //     -- For FP values, create a FMOVS or FMOVD instruction
        //     -- For non-FP values, create an add-with-0 instruction
        // 
        if (retVal != NULL) {
          const SparcV9RegInfo& regInfo =
            (SparcV9RegInfo&) target.getRegInfo();
          const Type* retType = retVal->getType();
          unsigned regClassID = regInfo.getRegClassIDOfType(retType);
          unsigned retRegNum = (retType->isFloatingPoint()
                                ? (unsigned) SparcV9FloatRegClass::f0
                                : (unsigned) SparcV9IntRegClass::i0);
          retRegNum = regInfo.getUnifiedRegNum(regClassID, retRegNum);

          // () Insert sign-extension instructions for small signed values.
          // 
          Value* retValToUse = retVal;
          if (retType->isIntegral() && retType->isSigned()) {
            unsigned retSize = target.getTargetData().getTypeSize(retType);
            if (retSize <= 4) {
              // create a temporary virtual reg. to hold the sign-extension
              retValToUse = new TmpInstruction(mcfi, retVal);

              // sign-extend retVal and put the result in the temporary reg.
              target.getInstrInfo().CreateSignExtensionInstructions
                (target, returnInstr->getParent()->getParent(),
                 retVal, retValToUse, 8*retSize, mvec, mcfi);
            }
          }

          // (b) Now, insert a copy to to the appropriate register:
          //     -- For FP values, create a FMOVS or FMOVD instruction
          //     -- For non-FP values, create an add-with-0 instruction
          // 
          // First, create a virtual register to represent the register and
          // mark this vreg as being an implicit operand of the ret MI.
          TmpInstruction* retVReg = 
            new TmpInstruction(mcfi, retValToUse, NULL, "argReg");
          
          retMI->addImplicitRef(retVReg);
          
          if (retType->isFloatingPoint())
            M = (BuildMI(retType==Type::FloatTy? V9::FMOVS : V9::FMOVD, 2)
                 .addReg(retValToUse).addReg(retVReg, MachineOperand::Def));
          else
            M = (BuildMI(ChooseAddInstructionByType(retType), 3)
                 .addReg(retValToUse).addSImm((int64_t) 0)
                 .addReg(retVReg, MachineOperand::Def));

          // Mark the operand with the register it should be assigned
          M->SetRegForOperand(M->getNumOperands()-1, retRegNum);
          retMI->SetRegForImplicitRef(retMI->getNumImplicitRefs()-1, retRegNum);

          mvec.push_back(M);
        }
        
        // Now insert the RET instruction and a NOP for the delay slot
        mvec.push_back(retMI);
        mvec.push_back(BuildMI(V9::NOP, 0));
        
        break;
      }  
        
      case 3:	// stmt:   Store(reg,reg)
      case 4:	// stmt:   Store(reg,ptrreg)
        SetOperandsForMemInstr(ChooseStoreInstruction(
                        subtreeRoot->leftChild()->getValue()->getType()),
                               mvec, subtreeRoot, target);
        break;

      case 5:	// stmt:   BrUncond
        {
          BranchInst *BI = cast<BranchInst>(subtreeRoot->getInstruction());
          mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(BI->getSuccessor(0)));
        
          // delay slot
          mvec.push_back(BuildMI(V9::NOP, 0));
          break;
        }

      case 206:	// stmt:   BrCond(setCCconst)
      { // setCCconst => boolean was computed with `%b = setCC type reg1 const'
        // If the constant is ZERO, we can use the branch-on-integer-register
        // instructions and avoid the SUBcc instruction entirely.
        // Otherwise this is just the same as case 5, so just fall through.
        // 
        InstrTreeNode* constNode = subtreeRoot->leftChild()->rightChild();
        assert(constNode &&
               constNode->getNodeType() ==InstrTreeNode::NTConstNode);
        Constant *constVal = cast<Constant>(constNode->getValue());
        bool isValidConst;
        
        if ((constVal->getType()->isInteger()
             || isa<PointerType>(constVal->getType()))
            && target.getInstrInfo().ConvertConstantToIntType(target,
                             constVal, constVal->getType(), isValidConst) == 0
            && isValidConst)
          {
            // That constant is a zero after all...
            // Use the left child of setCC as the first argument!
            // Mark the setCC node so that no code is generated for it.
            InstructionNode* setCCNode = (InstructionNode*)
                                         subtreeRoot->leftChild();
            assert(setCCNode->getOpLabel() == SetCCOp);
            setCCNode->markFoldedIntoParent();
            
            BranchInst* brInst=cast<BranchInst>(subtreeRoot->getInstruction());
            
            M = BuildMI(ChooseBprInstruction(subtreeRoot), 2)
                                .addReg(setCCNode->leftChild()->getValue())
                                .addPCDisp(brInst->getSuccessor(0));
            mvec.push_back(M);
            
            // delay slot
            mvec.push_back(BuildMI(V9::NOP, 0));

            // false branch
            mvec.push_back(BuildMI(V9::BA, 1)
                           .addPCDisp(brInst->getSuccessor(1)));
            
            // delay slot
            mvec.push_back(BuildMI(V9::NOP, 0));
            break;
          }
        // ELSE FALL THROUGH
      }

      case 6:	// stmt:   BrCond(setCC)
      { // bool => boolean was computed with SetCC.
        // The branch to use depends on whether it is FP, signed, or unsigned.
        // If it is an integer CC, we also need to find the unique
        // TmpInstruction representing that CC.
        // 
        BranchInst* brInst = cast<BranchInst>(subtreeRoot->getInstruction());
        const Type* setCCType;
        unsigned Opcode = ChooseBccInstruction(subtreeRoot, setCCType);
        Value* ccValue = GetTmpForCC(subtreeRoot->leftChild()->getValue(),
                                     brInst->getParent()->getParent(),
                                     setCCType,
                                     MachineCodeForInstruction::get(brInst));
        M = BuildMI(Opcode, 2).addCCReg(ccValue)
                              .addPCDisp(brInst->getSuccessor(0));
        mvec.push_back(M);

        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));

        // false branch
        mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(brInst->getSuccessor(1)));

        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));
        break;
      }
        
      case 208:	// stmt:   BrCond(boolconst)
      {
        // boolconst => boolean is a constant; use BA to first or second label
        Constant* constVal = 
          cast<Constant>(subtreeRoot->leftChild()->getValue());
        unsigned dest = cast<ConstantBool>(constVal)->getValue()? 0 : 1;
        
        M = BuildMI(V9::BA, 1).addPCDisp(
          cast<BranchInst>(subtreeRoot->getInstruction())->getSuccessor(dest));
        mvec.push_back(M);
        
        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));
        break;
      }
        
      case   8:	// stmt:   BrCond(boolreg)
      { // boolreg   => boolean is recorded in an integer register.
        //              Use branch-on-integer-register instruction.
        // 
        BranchInst *BI = cast<BranchInst>(subtreeRoot->getInstruction());
        M = BuildMI(V9::BRNZ, 2).addReg(subtreeRoot->leftChild()->getValue())
          .addPCDisp(BI->getSuccessor(0));
        mvec.push_back(M);

        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));

        // false branch
        mvec.push_back(BuildMI(V9::BA, 1).addPCDisp(BI->getSuccessor(1)));
        
        // delay slot
        mvec.push_back(BuildMI(V9::NOP, 0));
        break;
      }  
      
      case 9:	// stmt:   Switch(reg)
        assert(0 && "*** SWITCH instruction is not implemented yet.");
        break;

      case 10:	// reg:   VRegList(reg, reg)
        assert(0 && "VRegList should never be the topmost non-chain rule");
        break;

      case 21:	// bool:  Not(bool,reg): Compute with a conditional-move-on-reg
      { // First find the unary operand. It may be left or right, usually right.
        Instruction* notI = subtreeRoot->getInstruction();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(subtreeRoot->getInstruction()));
        unsigned ZeroReg = target.getRegInfo().getZeroRegNum();

        // Unconditionally set register to 0
        mvec.push_back(BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(notI));

        // Now conditionally move 1 into the register.
        // Mark the register as a use (as well as a def) because the old
        // value will be retained if the condition is false.
        mvec.push_back(BuildMI(V9::MOVRZi, 3).addReg(notArg).addZImm(1)
                       .addReg(notI, MachineOperand::UseAndDef));

        break;
      }

      case 421:	// reg:   BNot(reg,reg): Compute as reg = reg XOR-NOT 0
      { // First find the unary operand. It may be left or right, usually right.
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(subtreeRoot->getInstruction()));
        unsigned ZeroReg = target.getRegInfo().getZeroRegNum();
        mvec.push_back(BuildMI(V9::XNORr, 3).addReg(notArg).addMReg(ZeroReg)
                                       .addRegDef(subtreeRoot->getValue()));
        break;
      }

      case 322:	// reg:   Not(tobool, reg):
        // Fold CAST-TO-BOOL with NOT by inverting the sense of cast-to-bool
        foldCase = true;
        // Just fall through!

      case 22:	// reg:   ToBoolTy(reg):
      {
        Instruction* castI = subtreeRoot->getInstruction();
        Value* opVal = subtreeRoot->leftChild()->getValue();
        assert(opVal->getType()->isIntegral() ||
               isa<PointerType>(opVal->getType()));

        // Unconditionally set register to 0
        mvec.push_back(BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(castI));

        // Now conditionally move 1 into the register.
        // Mark the register as a use (as well as a def) because the old
        // value will be retained if the condition is false.
        MachineOpCode opCode = foldCase? V9::MOVRZi : V9::MOVRNZi;
        mvec.push_back(BuildMI(opCode, 3).addReg(opVal).addZImm(1)
                       .addReg(castI, MachineOperand::UseAndDef));

        break;
      }
      
      case 23:	// reg:   ToUByteTy(reg)
      case 24:	// reg:   ToSByteTy(reg)
      case 25:	// reg:   ToUShortTy(reg)
      case 26:	// reg:   ToShortTy(reg)
      case 27:	// reg:   ToUIntTy(reg)
      case 28:	// reg:   ToIntTy(reg)
      case 29:	// reg:   ToULongTy(reg)
      case 30:	// reg:   ToLongTy(reg)
      {
        //======================================================================
        // Rules for integer conversions:
        // 
        //--------
        // From ISO 1998 C++ Standard, Sec. 4.7:
        //
        // 2. If the destination type is unsigned, the resulting value is
        // the least unsigned integer congruent to the source integer
        // (modulo 2n where n is the number of bits used to represent the
        // unsigned type). [Note: In a two s complement representation,
        // this conversion is conceptual and there is no change in the
        // bit pattern (if there is no truncation). ]
        // 
        // 3. If the destination type is signed, the value is unchanged if
        // it can be represented in the destination type (and bitfield width);
        // otherwise, the value is implementation-defined.
        //--------
        // 
        // Since we assume 2s complement representations, this implies:
        // 
        // -- If operand is smaller than destination, zero-extend or sign-extend
        //    according to the signedness of the *operand*: source decides:
        //    (1) If operand is signed, sign-extend it.
        //        If dest is unsigned, zero-ext the result!
        //    (2) If operand is unsigned, our current invariant is that
        //        it's high bits are correct, so zero-extension is not needed.
        // 
        // -- If operand is same size as or larger than destination,
        //    zero-extend or sign-extend according to the signedness of
        //    the *destination*: destination decides:
        //    (1) If destination is signed, sign-extend (truncating if needed)
        //        This choice is implementation defined.  We sign-extend the
        //        operand, which matches both Sun's cc and gcc3.2.
        //    (2) If destination is unsigned, zero-extend (truncating if needed)
        //======================================================================

        Instruction* destI =  subtreeRoot->getInstruction();
        Function* currentFunc = destI->getParent()->getParent();
        MachineCodeForInstruction& mcfi=MachineCodeForInstruction::get(destI);

        Value* opVal = subtreeRoot->leftChild()->getValue();
        const Type* opType = opVal->getType();
        const Type* destType = destI->getType();
        unsigned opSize   = target.getTargetData().getTypeSize(opType);
        unsigned destSize = target.getTargetData().getTypeSize(destType);
        
        bool isIntegral = opType->isIntegral() || isa<PointerType>(opType);

        if (opType == Type::BoolTy ||
            opType == destType ||
            isIntegral && opSize == destSize && opSize == 8) {
          // nothing to do in all these cases
          forwardOperandNum = 0;          // forward first operand to user

        } else if (opType->isFloatingPoint()) {

          CreateCodeToConvertFloatToInt(target, opVal, destI, mvec, mcfi);
          if (destI->getType()->isUnsigned() && destI->getType() !=Type::UIntTy)
            maskUnsignedResult = true; // not handled by fp->int code

        } else if (isIntegral) {

          bool opSigned     = opType->isSigned();
          bool destSigned   = destType->isSigned();
          unsigned extSourceInBits = 8 * std::min<unsigned>(opSize, destSize);

          assert(! (opSize == destSize && opSigned == destSigned) &&
                 "How can different int types have same size and signedness?");

          bool signExtend = (opSize <  destSize && opSigned ||
                             opSize >= destSize && destSigned);

          bool signAndZeroExtend = (opSize < destSize && destSize < 8u &&
                                    opSigned && !destSigned);
          assert(!signAndZeroExtend || signExtend);

          bool zeroExtendOnly = opSize >= destSize && !destSigned;
          assert(!zeroExtendOnly || !signExtend);

          if (signExtend) {
            Value* signExtDest = (signAndZeroExtend
                                  ? new TmpInstruction(mcfi, destType, opVal)
                                  : destI);

            target.getInstrInfo().CreateSignExtensionInstructions
              (target, currentFunc,opVal,signExtDest,extSourceInBits,mvec,mcfi);

            if (signAndZeroExtend)
              target.getInstrInfo().CreateZeroExtensionInstructions
              (target, currentFunc, signExtDest, destI, 8*destSize, mvec, mcfi);
          }
          else if (zeroExtendOnly) {
            target.getInstrInfo().CreateZeroExtensionInstructions
              (target, currentFunc, opVal, destI, extSourceInBits, mvec, mcfi);
          }
          else
            forwardOperandNum = 0;          // forward first operand to user

        } else
          assert(0 && "Unrecognized operand type for convert-to-integer");

        break;
      }
      
      case  31:	// reg:   ToFloatTy(reg):
      case  32:	// reg:   ToDoubleTy(reg):
      case 232:	// reg:   ToDoubleTy(Constant):
      
        // If this instruction has a parent (a user) in the tree 
        // and the user is translated as an FsMULd instruction,
        // then the cast is unnecessary.  So check that first.
        // In the future, we'll want to do the same for the FdMULq instruction,
        // so do the check here instead of only for ToFloatTy(reg).
        // 
        if (subtreeRoot->parent() != NULL) {
          const MachineCodeForInstruction& mcfi =
            MachineCodeForInstruction::get(
                cast<InstructionNode>(subtreeRoot->parent())->getInstruction());
          if (mcfi.size() == 0 || mcfi.front()->getOpcode() == V9::FSMULD)
            forwardOperandNum = 0;    // forward first operand to user
        }

        if (forwardOperandNum != 0) {    // we do need the cast
          Value* leftVal = subtreeRoot->leftChild()->getValue();
          const Type* opType = leftVal->getType();
          MachineOpCode opCode=ChooseConvertToFloatInstr(target,
                                       subtreeRoot->getOpLabel(), opType);
          if (opCode == V9::NOP) {      // no conversion needed
            forwardOperandNum = 0;      // forward first operand to user
          } else {
            // If the source operand is a non-FP type it must be
            // first copied from int to float register via memory!
            Instruction *dest = subtreeRoot->getInstruction();
            Value* srcForCast;
            int n = 0;
            if (! opType->isFloatingPoint()) {
              // Create a temporary to represent the FP register
              // into which the integer will be copied via memory.
              // The type of this temporary will determine the FP
              // register used: single-prec for a 32-bit int or smaller,
              // double-prec for a 64-bit int.
              // 
              uint64_t srcSize =
                target.getTargetData().getTypeSize(leftVal->getType());
              Type* tmpTypeToUse =
                (srcSize <= 4)? Type::FloatTy : Type::DoubleTy;
              MachineCodeForInstruction &destMCFI = 
                MachineCodeForInstruction::get(dest);
              srcForCast = new TmpInstruction(destMCFI, tmpTypeToUse, dest);

              target.getInstrInfo().CreateCodeToCopyIntToFloat(target,
                         dest->getParent()->getParent(),
                         leftVal, cast<Instruction>(srcForCast),
                         mvec, destMCFI);
            } else
              srcForCast = leftVal;

            M = BuildMI(opCode, 2).addReg(srcForCast).addRegDef(dest);
            mvec.push_back(M);
          }
        }
        break;

      case 19:	// reg:   ToArrayTy(reg):
      case 20:	// reg:   ToPointerTy(reg):
        forwardOperandNum = 0;          // forward first operand to user
        break;

      case 233:	// reg:   Add(reg, Constant)
        maskUnsignedResult = true;
        M = CreateAddConstInstruction(subtreeRoot);
        if (M != NULL) {
          mvec.push_back(M);
          break;
        }
        // ELSE FALL THROUGH
        
      case 33:	// reg:   Add(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseAddInstruction(subtreeRoot), subtreeRoot, mvec);
        break;

      case 234:	// reg:   Sub(reg, Constant)
        maskUnsignedResult = true;
        M = CreateSubConstInstruction(subtreeRoot);
        if (M != NULL) {
          mvec.push_back(M);
          break;
        }
        // ELSE FALL THROUGH
        
      case 34:	// reg:   Sub(reg, reg)
        maskUnsignedResult = true;
        Add3OperandInstr(ChooseSubInstructionByType(
                                   subtreeRoot->getInstruction()->getType()),
                         subtreeRoot, mvec);
        break;

      case 135:	// reg:   Mul(todouble, todouble)
        checkCast = true;
        // FALL THROUGH 

      case 35:	// reg:   Mul(reg, reg)
      {
        maskUnsignedResult = true;
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? (MachineOpCode)V9::FSMULD
                                 : -1);
        Instruction* mulInstr = subtreeRoot->getInstruction();
        CreateMulInstruction(target, mulInstr->getParent()->getParent(),
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             mulInstr, mvec,
                             MachineCodeForInstruction::get(mulInstr),forceOp);
        break;
      }
      case 335:	// reg:   Mul(todouble, todoubleConst)
        checkCast = true;
        // FALL THROUGH 

      case 235:	// reg:   Mul(reg, Constant)
      {
        maskUnsignedResult = true;
        MachineOpCode forceOp = ((checkCast && BothFloatToDouble(subtreeRoot))
                                 ? (MachineOpCode)V9::FSMULD
                                 : -1);
        Instruction* mulInstr = subtreeRoot->getInstruction();
        CreateMulInstruction(target, mulInstr->getParent()->getParent(),
                             subtreeRoot->leftChild()->getValue(),
                             subtreeRoot->rightChild()->getValue(),
                             mulInstr, mvec,
                             MachineCodeForInstruction::get(mulInstr),
                             forceOp);
        break;
      }
      case 236:	// reg:   Div(reg, Constant)
        maskUnsignedResult = true;
        L = mvec.size();
        CreateDivConstInstruction(target, subtreeRoot, mvec);
        if (mvec.size() > L)
          break;
        // ELSE FALL THROUGH
      
      case 36:	// reg:   Div(reg, reg)
      {
        maskUnsignedResult = true;

        // If either operand of divide is smaller than 64 bits, we have
        // to make sure the unused top bits are correct because they affect
        // the result.  These bits are already correct for unsigned values.
        // They may be incorrect for signed values, so sign extend to fill in.
        Instruction* divI = subtreeRoot->getInstruction();
        Value* divOp1 = subtreeRoot->leftChild()->getValue();
        Value* divOp2 = subtreeRoot->rightChild()->getValue();
        Value* divOp1ToUse = divOp1;
        Value* divOp2ToUse = divOp2;
        if (divI->getType()->isSigned()) {
          unsigned opSize=target.getTargetData().getTypeSize(divI->getType());
          if (opSize < 8) {
            MachineCodeForInstruction& mcfi=MachineCodeForInstruction::get(divI);
            divOp1ToUse = new TmpInstruction(mcfi, divOp1);
            divOp2ToUse = new TmpInstruction(mcfi, divOp2);
            target.getInstrInfo().
              CreateSignExtensionInstructions(target,
                                              divI->getParent()->getParent(),
                                              divOp1, divOp1ToUse,
                                              8*opSize, mvec, mcfi);
            target.getInstrInfo().
              CreateSignExtensionInstructions(target,
                                              divI->getParent()->getParent(),
                                              divOp2, divOp2ToUse,
                                              8*opSize, mvec, mcfi);
          }
        }

        mvec.push_back(BuildMI(ChooseDivInstruction(target, subtreeRoot), 3)
                       .addReg(divOp1ToUse)
                       .addReg(divOp2ToUse)
                       .addRegDef(divI));

        break;
      }

      case  37:	// reg:   Rem(reg, reg)
      case 237:	// reg:   Rem(reg, Constant)
      {
        maskUnsignedResult = true;

        Instruction* remI   = subtreeRoot->getInstruction();
        Value* divOp1 = subtreeRoot->leftChild()->getValue();
        Value* divOp2 = subtreeRoot->rightChild()->getValue();

        MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(remI);
        
        // If second operand of divide is smaller than 64 bits, we have
        // to make sure the unused top bits are correct because they affect
        // the result.  These bits are already correct for unsigned values.
        // They may be incorrect for signed values, so sign extend to fill in.
        // 
        Value* divOpToUse = divOp2;
        if (divOp2->getType()->isSigned()) {
          unsigned opSize=target.getTargetData().getTypeSize(divOp2->getType());
          if (opSize < 8) {
            divOpToUse = new TmpInstruction(mcfi, divOp2);
            target.getInstrInfo().
              CreateSignExtensionInstructions(target,
                                              remI->getParent()->getParent(),
                                              divOp2, divOpToUse,
                                              8*opSize, mvec, mcfi);
          }
        }

        // Now compute: result = rem V1, V2 as:
        //      result = V1 - (V1 / signExtend(V2)) * signExtend(V2)
        // 
        TmpInstruction* quot = new TmpInstruction(mcfi, divOp1, divOpToUse);
        TmpInstruction* prod = new TmpInstruction(mcfi, quot, divOpToUse);

        mvec.push_back(BuildMI(ChooseDivInstruction(target, subtreeRoot), 3)
                       .addReg(divOp1).addReg(divOpToUse).addRegDef(quot));
        
        mvec.push_back(BuildMI(ChooseMulInstructionByType(remI->getType()), 3)
                       .addReg(quot).addReg(divOpToUse).addRegDef(prod));
        
        mvec.push_back(BuildMI(ChooseSubInstructionByType(remI->getType()), 3)
                       .addReg(divOp1).addReg(prod).addRegDef(remI));
        
        break;
      }
      
      case  38:	// bool:   And(bool, bool)
      case 138:	// bool:   And(bool, not)
      case 238:	// bool:   And(bool, boolconst)
      case 338:	// reg :   BAnd(reg, reg)
      case 538:	// reg :   BAnd(reg, Constant)
        Add3OperandInstr(V9::ANDr, subtreeRoot, mvec);
        break;

      case 438:	// bool:   BAnd(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        // If the type is boolean, set 1 or 0 in the result register.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *lhs = subtreeRoot->leftChild()->getValue();
        Value *dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::ANDNr, 3).addReg(lhs).addReg(notArg)
                                       .addReg(dest, MachineOperand::Def));

        if (notArg->getType() == Type::BoolTy) {
          // set 1 in result register if result of above is non-zero
          mvec.push_back(BuildMI(V9::MOVRNZi, 3).addReg(dest).addZImm(1)
                         .addReg(dest, MachineOperand::UseAndDef));
        }

        break;
      }

      case  39:	// bool:   Or(bool, bool)
      case 139:	// bool:   Or(bool, not)
      case 239:	// bool:   Or(bool, boolconst)
      case 339:	// reg :   BOr(reg, reg)
      case 539:	// reg :   BOr(reg, Constant)
        Add3OperandInstr(V9::ORr, subtreeRoot, mvec);
        break;

      case 439:	// bool:   BOr(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        // If the type is boolean, set 1 or 0 in the result register.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *lhs = subtreeRoot->leftChild()->getValue();
        Value *dest = subtreeRoot->getValue();

        mvec.push_back(BuildMI(V9::ORNr, 3).addReg(lhs).addReg(notArg)
                       .addReg(dest, MachineOperand::Def));

        if (notArg->getType() == Type::BoolTy) {
          // set 1 in result register if result of above is non-zero
          mvec.push_back(BuildMI(V9::MOVRNZi, 3).addReg(dest).addZImm(1)
                         .addReg(dest, MachineOperand::UseAndDef));
        }

        break;
      }

      case  40:	// bool:   Xor(bool, bool)
      case 140:	// bool:   Xor(bool, not)
      case 240:	// bool:   Xor(bool, boolconst)
      case 340:	// reg :   BXor(reg, reg)
      case 540:	// reg :   BXor(reg, Constant)
        Add3OperandInstr(V9::XORr, subtreeRoot, mvec);
        break;

      case 440:	// bool:   BXor(bool, bnot)
      { // Use the argument of NOT as the second argument!
        // Mark the NOT node so that no code is generated for it.
        // If the type is boolean, set 1 or 0 in the result register.
        InstructionNode* notNode = (InstructionNode*) subtreeRoot->rightChild();
        Value* notArg = BinaryOperator::getNotArgument(
                           cast<BinaryOperator>(notNode->getInstruction()));
        notNode->markFoldedIntoParent();
        Value *lhs = subtreeRoot->leftChild()->getValue();
        Value *dest = subtreeRoot->getValue();
        mvec.push_back(BuildMI(V9::XNORr, 3).addReg(lhs).addReg(notArg)
                       .addReg(dest, MachineOperand::Def));

        if (notArg->getType() == Type::BoolTy) {
          // set 1 in result register if result of above is non-zero
          mvec.push_back(BuildMI(V9::MOVRNZi, 3).addReg(dest).addZImm(1)
                         .addReg(dest, MachineOperand::UseAndDef));
        }
        break;
      }

      case 41:	// setCCconst:   SetCC(reg, Constant)
      { // Comparison is with a constant:
        // 
        // If the bool result must be computed into a register (see below),
        // and the constant is int ZERO, we can use the MOVR[op] instructions
        // and avoid the SUBcc instruction entirely.
        // Otherwise this is just the same as case 42, so just fall through.
        // 
        // The result of the SetCC must be computed and stored in a register if
        // it is used outside the current basic block (so it must be computed
        // as a boolreg) or it is used by anything other than a branch.
        // We will use a conditional move to do this.
        // 
        Instruction* setCCInstr = subtreeRoot->getInstruction();
        bool computeBoolVal = (subtreeRoot->parent() == NULL ||
                               ! AllUsesAreBranches(setCCInstr));

        if (computeBoolVal) {
          InstrTreeNode* constNode = subtreeRoot->rightChild();
          assert(constNode &&
                 constNode->getNodeType() ==InstrTreeNode::NTConstNode);
          Constant *constVal = cast<Constant>(constNode->getValue());
          bool isValidConst;
          
          if ((constVal->getType()->isInteger()
               || isa<PointerType>(constVal->getType()))
              && target.getInstrInfo().ConvertConstantToIntType(target,
                             constVal, constVal->getType(), isValidConst) == 0
              && isValidConst)
          {
            // That constant is an integer zero after all...
            // Use a MOVR[op] to compute the boolean result
            // Unconditionally set register to 0
            mvec.push_back(BuildMI(V9::SETHI, 2).addZImm(0)
                           .addRegDef(setCCInstr));
                
            // Now conditionally move 1 into the register.
            // Mark the register as a use (as well as a def) because the old
            // value will be retained if the condition is false.
            MachineOpCode movOpCode = ChooseMovpregiForSetCC(subtreeRoot);
            mvec.push_back(BuildMI(movOpCode, 3)
                           .addReg(subtreeRoot->leftChild()->getValue())
                           .addZImm(1)
                           .addReg(setCCInstr, MachineOperand::UseAndDef));
                
            break;
          }
        }
        // ELSE FALL THROUGH
      }

      case 42:	// bool:   SetCC(reg, reg):
      {
        // This generates a SUBCC instruction, putting the difference in a
        // result reg. if needed, and/or setting a condition code if needed.
        // 
        Instruction* setCCInstr = subtreeRoot->getInstruction();
        Value* leftVal  = subtreeRoot->leftChild()->getValue();
        Value* rightVal = subtreeRoot->rightChild()->getValue();
        const Type* opType = leftVal->getType();
        bool isFPCompare = opType->isFloatingPoint();
        
        // If the boolean result of the SetCC is used outside the current basic
        // block (so it must be computed as a boolreg) or is used by anything
        // other than a branch, the boolean must be computed and stored
        // in a result register.  We will use a conditional move to do this.
        // 
        bool computeBoolVal = (subtreeRoot->parent() == NULL ||
                               ! AllUsesAreBranches(setCCInstr));
        
        // A TmpInstruction is created to represent the CC "result".
        // Unlike other instances of TmpInstruction, this one is used
        // by machine code of multiple LLVM instructions, viz.,
        // the SetCC and the branch.  Make sure to get the same one!
        // Note that we do this even for FP CC registers even though they
        // are explicit operands, because the type of the operand
        // needs to be a floating point condition code, not an integer
        // condition code.  Think of this as casting the bool result to
        // a FP condition code register.
        // Later, we mark the 4th operand as being a CC register, and as a def.
        // 
        TmpInstruction* tmpForCC = GetTmpForCC(setCCInstr,
                                    setCCInstr->getParent()->getParent(),
                                    leftVal->getType(),
                                    MachineCodeForInstruction::get(setCCInstr));

        // If the operands are signed values smaller than 4 bytes, then they
        // must be sign-extended in order to do a valid 32-bit comparison
        // and get the right result in the 32-bit CC register (%icc).
        // 
        Value* leftOpToUse  = leftVal;
        Value* rightOpToUse = rightVal;
        if (opType->isIntegral() && opType->isSigned()) {
          unsigned opSize = target.getTargetData().getTypeSize(opType);
          if (opSize < 4) {
            MachineCodeForInstruction& mcfi =
              MachineCodeForInstruction::get(setCCInstr); 

            // create temporary virtual regs. to hold the sign-extensions
            leftOpToUse  = new TmpInstruction(mcfi, leftVal);
            rightOpToUse = new TmpInstruction(mcfi, rightVal);
            
            // sign-extend each operand and put the result in the temporary reg.
            target.getInstrInfo().CreateSignExtensionInstructions
              (target, setCCInstr->getParent()->getParent(),
               leftVal, leftOpToUse, 8*opSize, mvec, mcfi);
            target.getInstrInfo().CreateSignExtensionInstructions
              (target, setCCInstr->getParent()->getParent(),
               rightVal, rightOpToUse, 8*opSize, mvec, mcfi);
          }
        }

        if (! isFPCompare) {
          // Integer condition: set CC and discard result.
          mvec.push_back(BuildMI(V9::SUBccr, 4)
                         .addReg(leftOpToUse)
                         .addReg(rightOpToUse)
                         .addMReg(target.getRegInfo()
                                   .getZeroRegNum(), MachineOperand::Def)
                         .addCCReg(tmpForCC, MachineOperand::Def));
        } else {
          // FP condition: dest of FCMP should be some FCCn register
          mvec.push_back(BuildMI(ChooseFcmpInstruction(subtreeRoot), 3)
                         .addCCReg(tmpForCC, MachineOperand::Def)
                         .addReg(leftOpToUse)
                         .addReg(rightOpToUse));
        }
        
        if (computeBoolVal) {
          MachineOpCode movOpCode = (isFPCompare
                                     ? ChooseMovFpcciInstruction(subtreeRoot)
                                     : ChooseMovpcciForSetCC(subtreeRoot));

          // Unconditionally set register to 0
          M = BuildMI(V9::SETHI, 2).addZImm(0).addRegDef(setCCInstr);
          mvec.push_back(M);
          
          // Now conditionally move 1 into the register.
          // Mark the register as a use (as well as a def) because the old
          // value will be retained if the condition is false.
          M = (BuildMI(movOpCode, 3).addCCReg(tmpForCC).addZImm(1)
               .addReg(setCCInstr, MachineOperand::UseAndDef));
          mvec.push_back(M);
        }
        break;
      }    
      
      case 51:	// reg:   Load(reg)
      case 52:	// reg:   Load(ptrreg)
        SetOperandsForMemInstr(ChooseLoadInstruction(
                                   subtreeRoot->getValue()->getType()),
                               mvec, subtreeRoot, target);
        break;

      case 55:	// reg:   GetElemPtr(reg)
      case 56:	// reg:   GetElemPtrIdx(reg,reg)
        // If the GetElemPtr was folded into the user (parent), it will be
        // caught above.  For other cases, we have to compute the address.
        SetOperandsForMemInstr(V9::ADDr, mvec, subtreeRoot, target);
        break;

      case 57:	// reg:  Alloca: Implement as 1 instruction:
      {         //	    add %fp, offsetFromFP -> result
        AllocationInst* instr =
          cast<AllocationInst>(subtreeRoot->getInstruction());
        unsigned tsize =
          target.getTargetData().getTypeSize(instr->getAllocatedType());
        assert(tsize != 0);
        CreateCodeForFixedSizeAlloca(target, instr, tsize, 1, mvec);
        break;
      }

      case 58:	// reg:   Alloca(reg): Implement as 3 instructions:
                //	mul num, typeSz -> tmp
                //	sub %sp, tmp    -> %sp
      {         //	add %sp, frameSizeBelowDynamicArea -> result
        AllocationInst* instr =
          cast<AllocationInst>(subtreeRoot->getInstruction());
        const Type* eltType = instr->getAllocatedType();
        
        // If #elements is constant, use simpler code for fixed-size allocas
        int tsize = (int) target.getTargetData().getTypeSize(eltType);
        Value* numElementsVal = NULL;
        bool isArray = instr->isArrayAllocation();
        
        if (!isArray || isa<Constant>(numElementsVal = instr->getArraySize())) {
          // total size is constant: generate code for fixed-size alloca
          unsigned numElements = isArray? 
            cast<ConstantUInt>(numElementsVal)->getValue() : 1;
          CreateCodeForFixedSizeAlloca(target, instr, tsize,
                                       numElements, mvec);
        } else {
          // total size is not constant.
          CreateCodeForVariableSizeAlloca(target, instr, tsize,
                                          numElementsVal, mvec);
        }
        break;
      }

      case 61:	// reg:   Call
      {         // Generate a direct (CALL) or indirect (JMPL) call.
                // Mark the return-address register, the indirection
                // register (for indirect calls), the operands of the Call,
                // and the return value (if any) as implicit operands
                // of the machine instruction.
                // 
                // If this is a varargs function, floating point arguments
                // have to passed in integer registers so insert
                // copy-float-to-int instructions for each float operand.
                // 
        CallInst *callInstr = cast<CallInst>(subtreeRoot->getInstruction());
        Value *callee = callInstr->getCalledValue();
        Function* calledFunc = dyn_cast<Function>(callee);

        // Check if this is an intrinsic function that needs a special code
        // sequence (e.g., va_start).  Indirect calls cannot be special.
        // 
        bool specialIntrinsic = false;
        Intrinsic::ID iid;
        if (calledFunc && (iid=(Intrinsic::ID)calledFunc->getIntrinsicID()))
          specialIntrinsic = CodeGenIntrinsic(iid, *callInstr, target, mvec);

        // If not, generate the normal call sequence for the function.
        // This can also handle any intrinsics that are just function calls.
        // 
        if (! specialIntrinsic) {
          Function* currentFunc = callInstr->getParent()->getParent();
          MachineFunction& MF = MachineFunction::get(currentFunc);
          MachineCodeForInstruction& mcfi =
            MachineCodeForInstruction::get(callInstr); 
          const SparcV9RegInfo& regInfo =
            (SparcV9RegInfo&) target.getRegInfo();
          const TargetFrameInfo& frameInfo = target.getFrameInfo();

          // Create hidden virtual register for return address with type void*
          TmpInstruction* retAddrReg =
            new TmpInstruction(mcfi, PointerType::get(Type::VoidTy), callInstr);

          // Generate the machine instruction and its operands.
          // Use CALL for direct function calls; this optimistically assumes
          // the PC-relative address fits in the CALL address field (22 bits).
          // Use JMPL for indirect calls.
          // This will be added to mvec later, after operand copies.
          // 
          MachineInstr* callMI;
          if (calledFunc)             // direct function call
            callMI = BuildMI(V9::CALL, 1).addPCDisp(callee);
          else                        // indirect function call
            callMI = (BuildMI(V9::JMPLCALLi,3).addReg(callee)
                      .addSImm((int64_t)0).addRegDef(retAddrReg));

          const FunctionType* funcType =
            cast<FunctionType>(cast<PointerType>(callee->getType())
                               ->getElementType());
          bool isVarArgs = funcType->isVarArg();
          bool noPrototype = isVarArgs && funcType->getNumParams() == 0;
        
          // Use a descriptor to pass information about call arguments
          // to the register allocator.  This descriptor will be "owned"
          // and freed automatically when the MachineCodeForInstruction
          // object for the callInstr goes away.
          CallArgsDescriptor* argDesc =
            new CallArgsDescriptor(callInstr, retAddrReg,isVarArgs,noPrototype);
          assert(callInstr->getOperand(0) == callee
                 && "This is assumed in the loop below!");

          // Insert sign-extension instructions for small signed values,
          // if this is an unknown function (i.e., called via a funcptr)
          // or an external one (i.e., which may not be compiled by llc).
          // 
          if (calledFunc == NULL || calledFunc->isExternal()) {
            for (unsigned i=1, N=callInstr->getNumOperands(); i < N; ++i) {
              Value* argVal = callInstr->getOperand(i);
              const Type* argType = argVal->getType();
              if (argType->isIntegral() && argType->isSigned()) {
                unsigned argSize = target.getTargetData().getTypeSize(argType);
                if (argSize <= 4) {
                  // create a temporary virtual reg. to hold the sign-extension
                  TmpInstruction* argExtend = new TmpInstruction(mcfi, argVal);

                  // sign-extend argVal and put the result in the temporary reg.
                  target.getInstrInfo().CreateSignExtensionInstructions
                    (target, currentFunc, argVal, argExtend,
                     8*argSize, mvec, mcfi);

                  // replace argVal with argExtend in CallArgsDescriptor
                  argDesc->getArgInfo(i-1).replaceArgVal(argExtend);
                }
              }
            }
          }

          // Insert copy instructions to get all the arguments into
          // all the places that they need to be.
          // 
          for (unsigned i=1, N=callInstr->getNumOperands(); i < N; ++i) {
            int argNo = i-1;
            CallArgInfo& argInfo = argDesc->getArgInfo(argNo);
            Value* argVal = argInfo.getArgVal(); // don't use callInstr arg here
            const Type* argType = argVal->getType();
            unsigned regType = regInfo.getRegTypeForDataType(argType);
            unsigned argSize = target.getTargetData().getTypeSize(argType);
            int regNumForArg = TargetRegInfo::getInvalidRegNum();
            unsigned regClassIDOfArgReg;

            // Check for FP arguments to varargs functions.
            // Any such argument in the first $K$ args must be passed in an
            // integer register.  If there is no prototype, it must also
            // be passed as an FP register.
            // K = #integer argument registers.
            bool isFPArg = argVal->getType()->isFloatingPoint();
            if (isVarArgs && isFPArg) {

              if (noPrototype) {
                // It is a function with no prototype: pass value
                // as an FP value as well as a varargs value.  The FP value
                // may go in a register or on the stack.  The copy instruction
                // to the outgoing reg/stack is created by the normal argument
                // handling code since this is the "normal" passing mode.
                // 
                regNumForArg = regInfo.regNumForFPArg(regType,
                                                      false, false, argNo,
                                                      regClassIDOfArgReg);
                if (regNumForArg == regInfo.getInvalidRegNum())
                  argInfo.setUseStackSlot();
                else
                  argInfo.setUseFPArgReg();
              }
              
              // If this arg. is in the first $K$ regs, add special copy-
              // float-to-int instructions to pass the value as an int.
              // To check if it is in the first $K$, get the register
              // number for the arg #i.  These copy instructions are
              // generated here because they are extra cases and not needed
              // for the normal argument handling (some code reuse is
              // possible though -- later).
              // 
              int copyRegNum = regInfo.regNumForIntArg(false, false, argNo,
                                                       regClassIDOfArgReg);
              if (copyRegNum != regInfo.getInvalidRegNum()) {
                // Create a virtual register to represent copyReg. Mark
                // this vreg as being an implicit operand of the call MI
                const Type* loadTy = (argType == Type::FloatTy
                                      ? Type::IntTy : Type::LongTy);
                TmpInstruction* argVReg = new TmpInstruction(mcfi, loadTy,
                                                             argVal, NULL,
                                                             "argRegCopy");
                callMI->addImplicitRef(argVReg);
                
                // Get a temp stack location to use to copy
                // float-to-int via the stack.
                // 
                // FIXME: For now, we allocate permanent space because
                // the stack frame manager does not allow locals to be
                // allocated (e.g., for alloca) after a temp is
                // allocated!
                // 
                // int tmpOffset = MF.getInfo()->pushTempValue(argSize);
                int tmpOffset = MF.getInfo()->allocateLocalVar(argVReg);
                    
                // Generate the store from FP reg to stack
                unsigned StoreOpcode = ChooseStoreInstruction(argType);
                M = BuildMI(convertOpcodeFromRegToImm(StoreOpcode), 3)
                  .addReg(argVal).addMReg(regInfo.getFramePointer())
                  .addSImm(tmpOffset);
                mvec.push_back(M);
                        
                // Generate the load from stack to int arg reg
                unsigned LoadOpcode = ChooseLoadInstruction(loadTy);
                M = BuildMI(convertOpcodeFromRegToImm(LoadOpcode), 3)
                  .addMReg(regInfo.getFramePointer()).addSImm(tmpOffset)
                  .addReg(argVReg, MachineOperand::Def);

                // Mark operand with register it should be assigned
                // both for copy and for the callMI
                M->SetRegForOperand(M->getNumOperands()-1, copyRegNum);
                callMI->SetRegForImplicitRef(callMI->getNumImplicitRefs()-1,
                                             copyRegNum);
                mvec.push_back(M);

                // Add info about the argument to the CallArgsDescriptor
                argInfo.setUseIntArgReg();
                argInfo.setArgCopy(copyRegNum);
              } else {
                // Cannot fit in first $K$ regs so pass arg on stack
                argInfo.setUseStackSlot();
              }
            } else if (isFPArg) {
              // Get the outgoing arg reg to see if there is one.
              regNumForArg = regInfo.regNumForFPArg(regType, false, false,
                                                    argNo, regClassIDOfArgReg);
              if (regNumForArg == regInfo.getInvalidRegNum())
                argInfo.setUseStackSlot();
              else {
                argInfo.setUseFPArgReg();
                regNumForArg =regInfo.getUnifiedRegNum(regClassIDOfArgReg,
                                                       regNumForArg);
              }
            } else {
              // Get the outgoing arg reg to see if there is one.
              regNumForArg = regInfo.regNumForIntArg(false,false,
                                                     argNo, regClassIDOfArgReg);
              if (regNumForArg == regInfo.getInvalidRegNum())
                argInfo.setUseStackSlot();
              else {
                argInfo.setUseIntArgReg();
                regNumForArg =regInfo.getUnifiedRegNum(regClassIDOfArgReg,
                                                       regNumForArg);
              }
            }                

            // 
            // Now insert copy instructions to stack slot or arg. register
            // 
            if (argInfo.usesStackSlot()) {
              // Get the stack offset for this argument slot.
              // FP args on stack are right justified so adjust offset!
              // int arguments are also right justified but they are
              // always loaded as a full double-word so the offset does
              // not need to be adjusted.
              int argOffset = frameInfo.getOutgoingArgOffset(MF, argNo);
              if (argType->isFloatingPoint()) {
                unsigned slotSize = frameInfo.getSizeOfEachArgOnStack();
                assert(argSize <= slotSize && "Insufficient slot size!");
                argOffset += slotSize - argSize;
              }

              // Now generate instruction to copy argument to stack
              MachineOpCode storeOpCode =
                (argType->isFloatingPoint()
                 ? ((argSize == 4)? V9::STFi : V9::STDFi) : V9::STXi);

              M = BuildMI(storeOpCode, 3).addReg(argVal)
                .addMReg(regInfo.getStackPointer()).addSImm(argOffset);
              mvec.push_back(M);
            }
            else if (regNumForArg != regInfo.getInvalidRegNum()) {

              // Create a virtual register to represent the arg reg. Mark
              // this vreg as being an implicit operand of the call MI.
              TmpInstruction* argVReg = 
                new TmpInstruction(mcfi, argVal, NULL, "argReg");

              callMI->addImplicitRef(argVReg);
              
              // Generate the reg-to-reg copy into the outgoing arg reg.
              // -- For FP values, create a FMOVS or FMOVD instruction
              // -- For non-FP values, create an add-with-0 instruction
              if (argType->isFloatingPoint())
                M=(BuildMI(argType==Type::FloatTy? V9::FMOVS :V9::FMOVD,2)
                   .addReg(argVal).addReg(argVReg, MachineOperand::Def));
              else
                M = (BuildMI(ChooseAddInstructionByType(argType), 3)
                     .addReg(argVal).addSImm((int64_t) 0)
                     .addReg(argVReg, MachineOperand::Def));
              
              // Mark the operand with the register it should be assigned
              M->SetRegForOperand(M->getNumOperands()-1, regNumForArg);
              callMI->SetRegForImplicitRef(callMI->getNumImplicitRefs()-1,
                                           regNumForArg);

              mvec.push_back(M);
            }
            else
              assert(argInfo.getArgCopy() != regInfo.getInvalidRegNum() &&
                     "Arg. not in stack slot, primary or secondary register?");
          }

          // add call instruction and delay slot before copying return value
          mvec.push_back(callMI);
          mvec.push_back(BuildMI(V9::NOP, 0));

          // Add the return value as an implicit ref.  The call operands
          // were added above.  Also, add code to copy out the return value.
          // This is always register-to-register for int or FP return values.
          // 
          if (callInstr->getType() != Type::VoidTy) { 
            // Get the return value reg.
            const Type* retType = callInstr->getType();

            int regNum = (retType->isFloatingPoint()
                          ? (unsigned) SparcV9FloatRegClass::f0 
                          : (unsigned) SparcV9IntRegClass::o0);
            unsigned regClassID = regInfo.getRegClassIDOfType(retType);
            regNum = regInfo.getUnifiedRegNum(regClassID, regNum);

            // Create a virtual register to represent it and mark
            // this vreg as being an implicit operand of the call MI
            TmpInstruction* retVReg = 
              new TmpInstruction(mcfi, callInstr, NULL, "argReg");

            callMI->addImplicitRef(retVReg, /*isDef*/ true);

            // Generate the reg-to-reg copy from the return value reg.
            // -- For FP values, create a FMOVS or FMOVD instruction
            // -- For non-FP values, create an add-with-0 instruction
            if (retType->isFloatingPoint())
              M = (BuildMI(retType==Type::FloatTy? V9::FMOVS : V9::FMOVD, 2)
                   .addReg(retVReg).addReg(callInstr, MachineOperand::Def));
            else
              M = (BuildMI(ChooseAddInstructionByType(retType), 3)
                   .addReg(retVReg).addSImm((int64_t) 0)
                   .addReg(callInstr, MachineOperand::Def));

            // Mark the operand with the register it should be assigned
            // Also mark the implicit ref of the call defining this operand
            M->SetRegForOperand(0, regNum);
            callMI->SetRegForImplicitRef(callMI->getNumImplicitRefs()-1,regNum);

            mvec.push_back(M);
          }

          // For the CALL instruction, the ret. addr. reg. is also implicit
          if (isa<Function>(callee))
            callMI->addImplicitRef(retAddrReg, /*isDef*/ true);

          MF.getInfo()->popAllTempValues();  // free temps used for this inst
        }

        break;
      }
      
      case 62:	// reg:   Shl(reg, reg)
      {
        Value* argVal1 = subtreeRoot->leftChild()->getValue();
        Value* argVal2 = subtreeRoot->rightChild()->getValue();
        Instruction* shlInstr = subtreeRoot->getInstruction();
        
        const Type* opType = argVal1->getType();
        assert((opType->isInteger() || isa<PointerType>(opType)) &&
               "Shl unsupported for other types");
        unsigned opSize = target.getTargetData().getTypeSize(opType);
        
        CreateShiftInstructions(target, shlInstr->getParent()->getParent(),
                                (opSize > 4)? V9::SLLXr6:V9::SLLr5,
                                argVal1, argVal2, 0, shlInstr, mvec,
                                MachineCodeForInstruction::get(shlInstr));
        break;
      }
      
      case 63:	// reg:   Shr(reg, reg)
      { 
        const Type* opType = subtreeRoot->leftChild()->getValue()->getType();
        assert((opType->isInteger() || isa<PointerType>(opType)) &&
               "Shr unsupported for other types");
        unsigned opSize = target.getTargetData().getTypeSize(opType);
        Add3OperandInstr(opType->isSigned()
                         ? (opSize > 4? V9::SRAXr6 : V9::SRAr5)
                         : (opSize > 4? V9::SRLXr6 : V9::SRLr5),
                         subtreeRoot, mvec);
        break;
      }
      
      case 64:	// reg:   Phi(reg,reg)
        break;                          // don't forward the value

      case 65:	// reg:   VANext(reg):  the va_next(va_list, type) instruction
      { // Increment the va_list pointer register according to the type.
        // All LLVM argument types are <= 64 bits, so use one doubleword.
        Instruction* vaNextI = subtreeRoot->getInstruction();
        assert(target.getTargetData().getTypeSize(vaNextI->getType()) <= 8 &&
               "We assumed that all LLVM parameter types <= 8 bytes!");
        int argSize = target.getFrameInfo().getSizeOfEachArgOnStack();
        mvec.push_back(BuildMI(V9::ADDi, 3).addReg(vaNextI->getOperand(0)).
                       addSImm(argSize).addRegDef(vaNextI));
        break;
      }

      case 66:	// reg:   VAArg (reg): the va_arg instruction
      { // Load argument from stack using current va_list pointer value.
        // Use 64-bit load for all non-FP args, and LDDF or double for FP.
        Instruction* vaArgI = subtreeRoot->getInstruction();
        MachineOpCode loadOp = (vaArgI->getType()->isFloatingPoint()
                                ? (vaArgI->getType() == Type::FloatTy
                                   ? V9::LDFi : V9::LDDFi)
                                : V9::LDXi);
        mvec.push_back(BuildMI(loadOp, 3).addReg(vaArgI->getOperand(0)).
                       addSImm(0).addRegDef(vaArgI));
        break;
      }
      
      case 71:	// reg:     VReg
      case 72:	// reg:     Constant
        break;                          // don't forward the value

      default:
        assert(0 && "Unrecognized BURG rule");
        break;
      }
    }

  if (forwardOperandNum >= 0) {
    // We did not generate a machine instruction but need to use operand.
    // If user is in the same tree, replace Value in its machine operand.
    // If not, insert a copy instruction which should get coalesced away
    // by register allocation.
    if (subtreeRoot->parent() != NULL)
      ForwardOperand(subtreeRoot, subtreeRoot->parent(), forwardOperandNum);
    else {
      std::vector<MachineInstr*> minstrVec;
      Instruction* instr = subtreeRoot->getInstruction();
      target.getInstrInfo().
        CreateCopyInstructionsByType(target,
                                     instr->getParent()->getParent(),
                                     instr->getOperand(forwardOperandNum),
                                     instr, minstrVec,
                                     MachineCodeForInstruction::get(instr));
      assert(minstrVec.size() > 0);
      mvec.insert(mvec.end(), minstrVec.begin(), minstrVec.end());
    }
  }

  if (maskUnsignedResult) {
    // If result is unsigned and smaller than int reg size,
    // we need to clear high bits of result value.
    assert(forwardOperandNum < 0 && "Need mask but no instruction generated");
    Instruction* dest = subtreeRoot->getInstruction();
    if (dest->getType()->isUnsigned()) {
      unsigned destSize=target.getTargetData().getTypeSize(dest->getType());
      if (destSize <= 4) {
        // Mask high 64 - N bits, where N = 4*destSize.
        
        // Use a TmpInstruction to represent the
        // intermediate result before masking.  Since those instructions
        // have already been generated, go back and substitute tmpI
        // for dest in the result position of each one of them.
        // 
        MachineCodeForInstruction& mcfi = MachineCodeForInstruction::get(dest);
        TmpInstruction *tmpI = new TmpInstruction(mcfi, dest->getType(),
                                                  dest, NULL, "maskHi");
        Value* srlArgToUse = tmpI;

        unsigned numSubst = 0;
        for (unsigned i=0, N=mvec.size(); i < N; ++i) {

          // Make sure we substitute all occurrences of dest in these instrs.
          // Otherwise, we will have bogus code.
          bool someArgsWereIgnored = false;

          // Make sure not to substitute an upwards-exposed use -- that would
          // introduce a use of `tmpI' with no preceding def.  Therefore,
          // substitute a use or def-and-use operand only if a previous def
          // operand has already been substituted (i.e., numSusbt > 0).
          // 
          numSubst += mvec[i]->substituteValue(dest, tmpI,
                                               /*defsOnly*/ numSubst == 0,
                                               /*notDefsAndUses*/ numSubst > 0,
                                               someArgsWereIgnored);
          assert(!someArgsWereIgnored &&
                 "Operand `dest' exists but not replaced: probably bogus!");
        }
        assert(numSubst > 0 && "Operand `dest' not replaced: probably bogus!");

        // Left shift 32-N if size (N) is less than 32 bits.
        // Use another tmp. virtual register to represent this result.
        if (destSize < 4) {
          srlArgToUse = new TmpInstruction(mcfi, dest->getType(),
                                           tmpI, NULL, "maskHi2");
          mvec.push_back(BuildMI(V9::SLLXi6, 3).addReg(tmpI)
                         .addZImm(8*(4-destSize))
                         .addReg(srlArgToUse, MachineOperand::Def));
        }

        // Logical right shift 32-N to get zero extension in top 64-N bits.
        mvec.push_back(BuildMI(V9::SRLi5, 3).addReg(srlArgToUse)
                         .addZImm(8*(4-destSize))
                         .addReg(dest, MachineOperand::Def));

      } else if (destSize < 8) {
        assert(0 && "Unsupported type size: 32 < size < 64 bits");
      }
    }
  }
}

}
