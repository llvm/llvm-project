//===--- SPIRVUtils.h ---- SPIR-V Utility Functions -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains miscellaneous utility functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H
#define LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H

#include "MCTargetDesc/SPIRVBaseInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/TypedPointerType.h"
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace llvm {
class MCInst;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;
class MachineIRBuilder;
class MachineRegisterInfo;
class Register;
class StringRef;
class SPIRVInstrInfo;
class SPIRVSubtarget;
class SPIRVGlobalRegistry;

// This class implements a partial ordering visitor, which visits a cyclic graph
// in natural topological-like ordering. Topological ordering is not defined for
// directed graphs with cycles, so this assumes cycles are a single node, and
// ignores back-edges. The cycle is visited from the entry in the same
// topological-like ordering.
//
// Note: this visitor REQUIRES a reducible graph.
//
// This means once we visit a node, we know all the possible ancestors have been
// visited.
//
// clang-format off
//
// Given this graph:
//
//     ,-> B -\
// A -+        +---> D ----> E -> F -> G -> H
//     `-> C -/      ^                 |
//                   +-----------------+
//
// Visit order is:
//  A, [B, C in any order], D, E, F, G, H
//
// clang-format on
//
// Changing the function CFG between the construction of the visitor and
// visiting is undefined. The visitor can be reused, but if the CFG is updated,
// the visitor must be rebuilt.
class PartialOrderingVisitor {
  DomTreeBuilder::BBDomTree DT;
  LoopInfo LI;

  std::unordered_set<BasicBlock *> Queued = {};
  std::queue<BasicBlock *> ToVisit = {};

  struct OrderInfo {
    size_t Rank;
    size_t TraversalIndex;
  };

  using BlockToOrderInfoMap = std::unordered_map<BasicBlock *, OrderInfo>;
  BlockToOrderInfoMap BlockToOrder;
  std::vector<BasicBlock *> Order = {};

  // Get all basic-blocks reachable from Start.
  std::unordered_set<BasicBlock *> getReachableFrom(BasicBlock *Start);

  // Internal function used to determine the partial ordering.
  // Visits |BB| with the current rank being |Rank|.
  size_t visit(BasicBlock *BB, size_t Rank);

  bool CanBeVisited(BasicBlock *BB) const;

public:
  size_t GetNodeRank(BasicBlock *BB) const;

  // Build the visitor to operate on the function F.
  PartialOrderingVisitor(Function &F);

  // Returns true is |LHS| comes before |RHS| in the partial ordering.
  // If |LHS| and |RHS| have the same rank, the traversal order determines the
  // order (order is stable).
  bool compare(const BasicBlock *LHS, const BasicBlock *RHS) const;

  // Visit the function starting from the basic block |Start|, and calling |Op|
  // on each visited BB. This traversal ignores back-edges, meaning this won't
  // visit a node to which |Start| is not an ancestor.
  // If Op returns |true|, the visitor continues. If |Op| returns false, the
  // visitor will stop at that rank. This means if 2 nodes share the same rank,
  // and Op returns false when visiting the first, the second will be visited
  // afterwards. But none of their successors will.
  void partialOrderVisit(BasicBlock &Start,
                         std::function<bool(BasicBlock *)> Op);
};

// Add the given string as a series of integer operand, inserting null
// terminators and padding to make sure the operands all have 32-bit
// little-endian words.
void addStringImm(const StringRef &Str, MCInst &Inst);
void addStringImm(const StringRef &Str, MachineInstrBuilder &MIB);
void addStringImm(const StringRef &Str, IRBuilder<> &B,
                  std::vector<Value *> &Args);

// Read the series of integer operands back as a null-terminated string using
// the reverse of the logic in addStringImm.
std::string getStringImm(const MachineInstr &MI, unsigned StartIndex);

// Returns the string constant that the register refers to. It is assumed that
// Reg is a global value that contains a string.
std::string getStringValueFromReg(Register Reg, MachineRegisterInfo &MRI);

// Add the given numerical immediate to MIB.
void addNumImm(const APInt &Imm, MachineInstrBuilder &MIB);

// Add an OpName instruction for the given target register.
void buildOpName(Register Target, const StringRef &Name,
                 MachineIRBuilder &MIRBuilder);
void buildOpName(Register Target, const StringRef &Name, MachineInstr &I,
                 const SPIRVInstrInfo &TII);

// Add an OpDecorate instruction for the given Reg.
void buildOpDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                     SPIRV::Decoration::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs,
                     StringRef StrImm = "");
void buildOpDecorate(Register Reg, MachineInstr &I, const SPIRVInstrInfo &TII,
                     SPIRV::Decoration::Decoration Dec,
                     const std::vector<uint32_t> &DecArgs,
                     StringRef StrImm = "");

// Add an OpDecorate instruction for the given Reg.
void buildOpMemberDecorate(Register Reg, MachineIRBuilder &MIRBuilder,
                           SPIRV::Decoration::Decoration Dec, uint32_t Member,
                           const std::vector<uint32_t> &DecArgs,
                           StringRef StrImm = "");
void buildOpMemberDecorate(Register Reg, MachineInstr &I,
                           const SPIRVInstrInfo &TII,
                           SPIRV::Decoration::Decoration Dec, uint32_t Member,
                           const std::vector<uint32_t> &DecArgs,
                           StringRef StrImm = "");

// Add an OpDecorate instruction by "spirv.Decorations" metadata node.
void buildOpSpirvDecorations(Register Reg, MachineIRBuilder &MIRBuilder,
                             const MDNode *GVarMD);

// Return a valid position for the OpVariable instruction inside a function,
// i.e., at the beginning of the first block of the function.
MachineBasicBlock::iterator getOpVariableMBBIt(MachineInstr &I);

// Return a valid position for the instruction at the end of the block before
// terminators and debug instructions.
MachineBasicBlock::iterator getInsertPtValidEnd(MachineBasicBlock *MBB);

// Returns true if a pointer to the storage class can be casted to/from a
// pointer to the Generic storage class.
constexpr bool isGenericCastablePtr(SPIRV::StorageClass::StorageClass SC) {
  switch (SC) {
  case SPIRV::StorageClass::Workgroup:
  case SPIRV::StorageClass::CrossWorkgroup:
  case SPIRV::StorageClass::Function:
    return true;
  default:
    return false;
  }
}

// Convert a SPIR-V storage class to the corresponding LLVM IR address space.
// TODO: maybe the following two functions should be handled in the subtarget
// to allow for different OpenCL vs Vulkan handling.
constexpr unsigned
storageClassToAddressSpace(SPIRV::StorageClass::StorageClass SC) {
  switch (SC) {
  case SPIRV::StorageClass::Function:
    return 0;
  case SPIRV::StorageClass::CrossWorkgroup:
    return 1;
  case SPIRV::StorageClass::UniformConstant:
    return 2;
  case SPIRV::StorageClass::Workgroup:
    return 3;
  case SPIRV::StorageClass::Generic:
    return 4;
  case SPIRV::StorageClass::DeviceOnlyINTEL:
    return 5;
  case SPIRV::StorageClass::HostOnlyINTEL:
    return 6;
  case SPIRV::StorageClass::Input:
    return 7;
  case SPIRV::StorageClass::Output:
    return 8;
  case SPIRV::StorageClass::CodeSectionINTEL:
    return 9;
  case SPIRV::StorageClass::Private:
    return 10;
  case SPIRV::StorageClass::StorageBuffer:
    return 11;
  case SPIRV::StorageClass::Uniform:
    return 12;
  default:
    report_fatal_error("Unable to get address space id");
  }
}

// Convert an LLVM IR address space to a SPIR-V storage class.
SPIRV::StorageClass::StorageClass
addressSpaceToStorageClass(unsigned AddrSpace, const SPIRVSubtarget &STI);

SPIRV::MemorySemantics::MemorySemantics
getMemSemanticsForStorageClass(SPIRV::StorageClass::StorageClass SC);

SPIRV::MemorySemantics::MemorySemantics getMemSemantics(AtomicOrdering Ord);

SPIRV::Scope::Scope getMemScope(LLVMContext &Ctx, SyncScope::ID Id);

// Find def instruction for the given ConstReg, walking through
// spv_track_constant and ASSIGN_TYPE instructions. Updates ConstReg by def
// of OpConstant instruction.
MachineInstr *getDefInstrMaybeConstant(Register &ConstReg,
                                       const MachineRegisterInfo *MRI);

// Get constant integer value of the given ConstReg.
uint64_t getIConstVal(Register ConstReg, const MachineRegisterInfo *MRI);

// Check if MI is a SPIR-V specific intrinsic call.
bool isSpvIntrinsic(const MachineInstr &MI, Intrinsic::ID IntrinsicID);
// Check if it's a SPIR-V specific intrinsic call.
bool isSpvIntrinsic(const Value *Arg);

// Get type of i-th operand of the metadata node.
Type *getMDOperandAsType(const MDNode *N, unsigned I);

// If OpenCL or SPIR-V builtin function name is recognized, return a demangled
// name, otherwise return an empty string.
std::string getOclOrSpirvBuiltinDemangledName(StringRef Name);

// Check if a string contains a builtin prefix.
bool hasBuiltinTypePrefix(StringRef Name);

// Check if given LLVM type is a special opaque builtin type.
bool isSpecialOpaqueType(const Type *Ty);

// Check if the function is an SPIR-V entry point
bool isEntryPoint(const Function &F);

// Parse basic scalar type name, substring TypeName, and return LLVM type.
Type *parseBasicTypeName(StringRef &TypeName, LLVMContext &Ctx);

// Sort blocks in a partial ordering, so each block is after all its
// dominators. This should match both the SPIR-V and the MIR requirements.
// Returns true if the function was changed.
bool sortBlocks(Function &F);

inline bool hasInitializer(const GlobalVariable *GV) {
  return GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer());
}

// True if this is an instance of TypedPointerType.
inline bool isTypedPointerTy(const Type *T) {
  return T && T->getTypeID() == Type::TypedPointerTyID;
}

// True if this is an instance of PointerType.
inline bool isUntypedPointerTy(const Type *T) {
  return T && T->getTypeID() == Type::PointerTyID;
}

// True if this is an instance of PointerType or TypedPointerType.
inline bool isPointerTy(const Type *T) {
  return isUntypedPointerTy(T) || isTypedPointerTy(T);
}

// Get the address space of this pointer or pointer vector type for instances of
// PointerType or TypedPointerType.
inline unsigned getPointerAddressSpace(const Type *T) {
  Type *SubT = T->getScalarType();
  return SubT->getTypeID() == Type::PointerTyID
             ? cast<PointerType>(SubT)->getAddressSpace()
             : cast<TypedPointerType>(SubT)->getAddressSpace();
}

// Return true if the Argument is decorated with a pointee type
inline bool hasPointeeTypeAttr(Argument *Arg) {
  return Arg->hasByValAttr() || Arg->hasByRefAttr() || Arg->hasStructRetAttr();
}

// Return the pointee type of the argument or nullptr otherwise
inline Type *getPointeeTypeByAttr(Argument *Arg) {
  if (Arg->hasByValAttr())
    return Arg->getParamByValType();
  if (Arg->hasStructRetAttr())
    return Arg->getParamStructRetType();
  if (Arg->hasByRefAttr())
    return Arg->getParamByRefType();
  return nullptr;
}

inline Type *reconstructFunctionType(Function *F) {
  SmallVector<Type *> ArgTys;
  for (unsigned i = 0; i < F->arg_size(); ++i)
    ArgTys.push_back(F->getArg(i)->getType());
  return FunctionType::get(F->getReturnType(), ArgTys, F->isVarArg());
}

#define TYPED_PTR_TARGET_EXT_NAME "spirv.$TypedPointerType"
inline Type *getTypedPointerWrapper(Type *ElemTy, unsigned AS) {
  return TargetExtType::get(ElemTy->getContext(), TYPED_PTR_TARGET_EXT_NAME,
                            {ElemTy}, {AS});
}

inline bool isTypedPointerWrapper(const TargetExtType *ExtTy) {
  return ExtTy->getName() == TYPED_PTR_TARGET_EXT_NAME &&
         ExtTy->getNumIntParameters() == 1 &&
         ExtTy->getNumTypeParameters() == 1;
}

// True if this is an instance of PointerType or TypedPointerType.
inline bool isPointerTyOrWrapper(const Type *Ty) {
  if (auto *ExtTy = dyn_cast<TargetExtType>(Ty))
    return isTypedPointerWrapper(ExtTy);
  return isPointerTy(Ty);
}

inline Type *applyWrappers(Type *Ty) {
  if (auto *ExtTy = dyn_cast<TargetExtType>(Ty)) {
    if (isTypedPointerWrapper(ExtTy))
      return TypedPointerType::get(applyWrappers(ExtTy->getTypeParameter(0)),
                                   ExtTy->getIntParameter(0));
  } else if (auto *VecTy = dyn_cast<VectorType>(Ty)) {
    Type *ElemTy = VecTy->getElementType();
    Type *NewElemTy = ElemTy->isTargetExtTy() ? applyWrappers(ElemTy) : ElemTy;
    if (NewElemTy != ElemTy)
      return VectorType::get(NewElemTy, VecTy->getElementCount());
  }
  return Ty;
}

inline Type *getPointeeType(const Type *Ty) {
  if (Ty) {
    if (auto PType = dyn_cast<TypedPointerType>(Ty))
      return PType->getElementType();
    else if (auto *ExtTy = dyn_cast<TargetExtType>(Ty))
      if (isTypedPointerWrapper(ExtTy))
        return ExtTy->getTypeParameter(0);
  }
  return nullptr;
}

inline bool isUntypedEquivalentToTyExt(Type *Ty1, Type *Ty2) {
  if (!isUntypedPointerTy(Ty1) || !Ty2)
    return false;
  if (auto *ExtTy = dyn_cast<TargetExtType>(Ty2))
    if (isTypedPointerWrapper(ExtTy) &&
        ExtTy->getTypeParameter(0) ==
            IntegerType::getInt8Ty(Ty1->getContext()) &&
        ExtTy->getIntParameter(0) == cast<PointerType>(Ty1)->getAddressSpace())
      return true;
  return false;
}

inline bool isEquivalentTypes(Type *Ty1, Type *Ty2) {
  return isUntypedEquivalentToTyExt(Ty1, Ty2) ||
         isUntypedEquivalentToTyExt(Ty2, Ty1);
}

inline Type *toTypedPointer(Type *Ty) {
  if (Type *NewTy = applyWrappers(Ty); NewTy != Ty)
    return NewTy;
  return isUntypedPointerTy(Ty)
             ? TypedPointerType::get(IntegerType::getInt8Ty(Ty->getContext()),
                                     getPointerAddressSpace(Ty))
             : Ty;
}

inline Type *toTypedFunPointer(FunctionType *FTy) {
  Type *OrigRetTy = FTy->getReturnType();
  Type *RetTy = toTypedPointer(OrigRetTy);
  bool IsUntypedPtr = false;
  for (Type *PTy : FTy->params()) {
    if (isUntypedPointerTy(PTy)) {
      IsUntypedPtr = true;
      break;
    }
  }
  if (!IsUntypedPtr && RetTy == OrigRetTy)
    return FTy;
  SmallVector<Type *> ParamTys;
  for (Type *PTy : FTy->params())
    ParamTys.push_back(toTypedPointer(PTy));
  return FunctionType::get(RetTy, ParamTys, FTy->isVarArg());
}

inline const Type *unifyPtrType(const Type *Ty) {
  if (auto FTy = dyn_cast<FunctionType>(Ty))
    return toTypedFunPointer(const_cast<FunctionType *>(FTy));
  return toTypedPointer(const_cast<Type *>(Ty));
}

inline bool isVector1(Type *Ty) {
  auto *FVTy = dyn_cast<FixedVectorType>(Ty);
  return FVTy && FVTy->getNumElements() == 1;
}

// Modify an LLVM type to conform with future transformations in IRTranslator.
// At the moment use cases comprise only a <1 x Type> vector. To extend when/if
// needed.
inline Type *normalizeType(Type *Ty) {
  auto *FVTy = dyn_cast<FixedVectorType>(Ty);
  if (!FVTy || FVTy->getNumElements() != 1)
    return Ty;
  // If it's a <1 x Type> vector type, replace it by the element type, because
  // it's not a legal vector type in LLT and IRTranslator will represent it as
  // the scalar eventually.
  return normalizeType(FVTy->getElementType());
}

inline PoisonValue *getNormalizedPoisonValue(Type *Ty) {
  return PoisonValue::get(normalizeType(Ty));
}

inline MetadataAsValue *buildMD(Value *Arg) {
  LLVMContext &Ctx = Arg->getContext();
  return MetadataAsValue::get(
      Ctx, MDNode::get(Ctx, ValueAsMetadata::getConstant(Arg)));
}

CallInst *buildIntrWithMD(Intrinsic::ID IntrID, ArrayRef<Type *> Types,
                          Value *Arg, Value *Arg2, ArrayRef<Constant *> Imms,
                          IRBuilder<> &B);

MachineInstr *getVRegDef(MachineRegisterInfo &MRI, Register Reg);

#define SPIRV_BACKEND_SERVICE_FUN_NAME "__spirv_backend_service_fun"
bool getVacantFunctionName(Module &M, std::string &Name);

void setRegClassType(Register Reg, const Type *Ty, SPIRVGlobalRegistry *GR,
                     MachineIRBuilder &MIRBuilder,
                     SPIRV::AccessQualifier::AccessQualifier AccessQual,
                     bool EmitIR, bool Force = false);
void setRegClassType(Register Reg, const MachineInstr *SpvType,
                     SPIRVGlobalRegistry *GR, MachineRegisterInfo *MRI,
                     const MachineFunction &MF, bool Force = false);
Register createVirtualRegister(const MachineInstr *SpvType,
                               SPIRVGlobalRegistry *GR,
                               MachineRegisterInfo *MRI,
                               const MachineFunction &MF);
Register createVirtualRegister(const MachineInstr *SpvType,
                               SPIRVGlobalRegistry *GR,
                               MachineIRBuilder &MIRBuilder);
Register createVirtualRegister(
    const Type *Ty, SPIRVGlobalRegistry *GR, MachineIRBuilder &MIRBuilder,
    SPIRV::AccessQualifier::AccessQualifier AccessQual, bool EmitIR);

// Return true if there is an opaque pointer type nested in the argument.
bool isNestedPointer(const Type *Ty);

enum FPDecorationId { NONE, RTE, RTZ, RTP, RTN, SAT };

inline FPDecorationId demangledPostfixToDecorationId(const std::string &S) {
  static std::unordered_map<std::string, FPDecorationId> Mapping = {
      {"rte", FPDecorationId::RTE},
      {"rtz", FPDecorationId::RTZ},
      {"rtp", FPDecorationId::RTP},
      {"rtn", FPDecorationId::RTN},
      {"sat", FPDecorationId::SAT}};
  auto It = Mapping.find(S);
  return It == Mapping.end() ? FPDecorationId::NONE : It->second;
}

SmallVector<MachineInstr *, 4>
createContinuedInstructions(MachineIRBuilder &MIRBuilder, unsigned Opcode,
                            unsigned MinWC, unsigned ContinuedOpcode,
                            ArrayRef<Register> Args, Register ReturnRegister,
                            Register TypeID);

// Instruction selection directed by type folding.
const std::set<unsigned> &getTypeFoldingSupportedOpcodes();
bool isTypeFoldingSupported(unsigned Opcode);

// Get loop controls from llvm.loop. metadata.
SmallVector<unsigned, 1> getSpirvLoopControlOperandsFromLoopMetadata(Loop *L);

// Traversing [g]MIR accounting for pseudo-instructions.
MachineInstr *passCopy(MachineInstr *Def, const MachineRegisterInfo *MRI);
MachineInstr *getDef(const MachineOperand &MO, const MachineRegisterInfo *MRI);
MachineInstr *getImm(const MachineOperand &MO, const MachineRegisterInfo *MRI);
int64_t foldImm(const MachineOperand &MO, const MachineRegisterInfo *MRI);
unsigned getArrayComponentCount(const MachineRegisterInfo *MRI,
                                const MachineInstr *ResType);
MachineBasicBlock::iterator
getFirstValidInstructionInsertPoint(MachineBasicBlock &BB);

} // namespace llvm
#endif // LLVM_LIB_TARGET_SPIRV_SPIRVUTILS_H
