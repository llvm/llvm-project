//===- YkIR/YkIRWRiter.cpp -- Yk JIT IR Serialiaser---------------------===//
//
// Converts an LLVM module into Yk's on-disk AOT IR.
//
//===-------------------------------------------------------------------===//

#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace std;

namespace {

class SerialiseInstructionException {
private:
  string S;

public:
  SerialiseInstructionException(string S) : S(S) {}
  string &what() { return S; }
};

const char *SectionName = ".yk_ir";
const uint32_t Magic = 0xedd5f00d;
const uint32_t Version = 0;

enum OpCode {
  Nop = 0,
  Load,
  Store,
  Alloca,
  Call,
  GetElementPtr,
  Br,
  CondBr,
  ICmp,
  BinaryOperator,
  Ret,
  UnimplementedInstruction = 255, // YKFIXME: Will eventually be deleted.
};

enum OperandKind {
  Constant = 0,
  LocalVariable,
  Type,
  Function,
  Block,
  Arg,
  UnimplementedOperand = 255,
};

enum TypeKind {
  Void = 0,
  Integer,
  Ptr,
  FunctionTy,
  UnimplementedType = 255, // YKFIXME: Will eventually be deleted.
};

template <class T> string toString(T *X) {
  string S;
  raw_string_ostream SS(S);
  X->print(SS);
  return S;
}

// Get the index of an element in its parent container.
template <class C, class E> size_t getIndex(C *Container, E *FindElement) {
  bool Found = false;
  size_t Idx = 0;
  for (E &AnElement : *Container) {
    if (&AnElement == FindElement) {
      Found = true;
      break;
    }
    Idx++;
  }
  assert(Found);
  return Idx;
}

// A <BBIdx, InstrIdx> pair that Uniquely identifies an Yk IR instruction within
// a function.
using InstrLoc = std::tuple<size_t, size_t>;

// Maps an LLVM instruction that generates a value to the corresponding Yk IR
// instruction.
using ValueLoweringMap = map<Instruction *, InstrLoc>;

// The class responsible for serialising our IR into the interpreter binary.
//
// It walks over the LLVM IR, lowering each function, block, instruction, etc.
// into a Yk IR equivalent.
//
// As it does this there are some invariants that must be maintained:
//
//  - The current basic block index (BBIdx) is passed down the lowering process.
//    This must be incremented each time we finish a Yk IR basic block.
//
//  - Similarly for instructions. Each time we finish a Yk IR instruction,
//    we must increment the current instruction index (InstIdx).
//
//  - When we are done lowering an LLVM instruction that generates a value, we
//    must update the `VLMap` with an entry that maps the LLVM instruction to
//    the final Yk IR instruction in the lowering. If the LLVM instruction
//    doesn't generate a value, or the LLVM instruction lowered to exactly zero
//    Yk IR instructions, then there is no need to update the `VLMap`.
//
// These invariants are required so that when we encounter a local variable as
// an operand to an LLVM instruction, we can quickly find the corresponding Yk
// IR local variable.
class YkIRWriter {
private:
  Module &M;
  MCStreamer &OutStreamer;

  vector<llvm::Type *> Types;
  vector<llvm::Constant *> Constants;

  // Return the index of the LLVM type `Ty`, inserting a new entry if
  // necessary.
  size_t typeIndex(llvm::Type *Ty) {
    vector<llvm::Type *>::iterator Found =
        std::find(Types.begin(), Types.end(), Ty);
    if (Found != Types.end()) {
      return std::distance(Types.begin(), Found);
    }

    // Not found. Assign it a type index.
    size_t Idx = Types.size();
    Types.push_back(Ty);

    // If the newly-registered type is an aggregate type that contains other
    // types, then assign them type indices now too.
    for (llvm::Type *STy : Ty->subtypes()) {
      typeIndex(STy);
    }

    return Idx;
  }

  // Return the index of the LLVM constant `C`, inserting a new entry if
  // necessary.
  size_t constantIndex(class Constant *C) {
    vector<class Constant *>::iterator Found =
        std::find(Constants.begin(), Constants.end(), C);
    if (Found != Constants.end()) {
      return std::distance(Constants.begin(), Found);
    }
    size_t Idx = Constants.size();
    Constants.push_back(C);
    return Idx;
  }

  size_t functionIndex(llvm::Function *F) {
    // FIXME: For now we assume that function indicies in LLVM IR and our IR
    // are the same.
    return getIndex(&M, F);
  }

public:
  YkIRWriter(Module &M, MCStreamer &OutStreamer)
      : M(M), OutStreamer(OutStreamer) {}

  // Serialises a null-terminated string.
  void serialiseString(StringRef S) {
    OutStreamer.emitBinaryData(S);
    OutStreamer.emitInt8(0); // null terminator.
  }

  void serialiseOpcode(OpCode Code) { OutStreamer.emitInt8(Code); }

  void serialiseConstantOperand(Instruction *Parent, llvm::Constant *C) {
    OutStreamer.emitInt8(OperandKind::Constant);
    OutStreamer.emitSizeT(constantIndex(C));
  }

  void serialiseLocalVariableOperand(Instruction *I, ValueLoweringMap &VLMap) {
    auto [BBIdx, InstIdx] = VLMap.at(I);
    OutStreamer.emitInt8(OperandKind::LocalVariable);
    OutStreamer.emitSizeT(BBIdx);
    OutStreamer.emitSizeT(InstIdx);
  }

  void serialiseStringOperand(const char *S) {
    OutStreamer.emitInt8(OperandKind::UnimplementedOperand);
    serialiseString(S);
  }

  void serialiseFunctionOperand(llvm::Function *F) {
    OutStreamer.emitInt8(OperandKind::Function);
    OutStreamer.emitSizeT(functionIndex(F));
  }

  void serialiseBlockOperand(BasicBlock *BB, ValueLoweringMap &VLMap) {
    OutStreamer.emitInt8(OperandKind::Block);
    // FIXME: For now we assume that basic block indices are the same in LLVM
    // IR and our IR.
    OutStreamer.emitSizeT(getIndex(BB->getParent(), BB));
  }

  // YKFIXME: This allows programs which we haven't yet defined a
  // lowering for to compile. For now We just emit a string operand containing
  // the unhandled LLVM operand in textual form.
  void serialiseUnimplementedOperand(Value *V) {
    OutStreamer.emitInt8(OperandKind::UnimplementedOperand);
    serialiseString(toString(V));
  }

  void serialiseArgOperand(ValueLoweringMap &VLMap, Argument *A) {
    // This assumes that the argument indices match in both IRs.
    OutStreamer.emitInt8(OperandKind::Arg);
    OutStreamer.emitSizeT(A->getArgNo());
  }

  void serialiseOperand(Instruction *Parent, ValueLoweringMap &VLMap,
                        Value *V) {
    if (llvm::Function *F = dyn_cast<llvm::Function>(V)) {
      serialiseFunctionOperand(F);
    } else if (llvm::Constant *C = dyn_cast<llvm::Constant>(V)) {
      serialiseConstantOperand(Parent, C);
    } else if (llvm::Argument *A = dyn_cast<llvm::Argument>(V)) {
      serialiseArgOperand(VLMap, A);
    } else if (Instruction *I = dyn_cast<Instruction>(V)) {
      // If an instruction defines the operand, it's a local variable.
      serialiseLocalVariableOperand(I, VLMap);
    } else if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
      serialiseBlockOperand(BB, VLMap);
    } else {
      serialiseUnimplementedOperand(V);
    }
  }

  /// Does a naiave serialisation of an LLVM instruction by iterating over its
  /// operands and serialising them in turn.
  void serialiseInstGeneric(Instruction *I, ValueLoweringMap &VLMap,
                            unsigned BBIdx, unsigned &InstIdx, OpCode Opc) {
    OutStreamer.emitSizeT(typeIndex(I->getType()));
    serialiseOpcode(Opc);
    OutStreamer.emitInt32(I->getNumOperands());
    for (Value *O : I->operands()) {
      serialiseOperand(I, VLMap, O);
    }
    if (!I->getType()->isVoidTy()) {
      VLMap[I] = {BBIdx, InstIdx};
    }
    InstIdx++;
  }

  void serialiseAllocaInst(AllocaInst *I, ValueLoweringMap &VLMap,
                           unsigned BBIdx, unsigned &InstIdx) {
    // type_index:
    OutStreamer.emitSizeT(typeIndex(I->getType()));
    // opcode:
    serialiseOpcode(OpCode::Alloca);
    // num_operands:
    OutStreamer.emitInt32(2);

    // OPERAND 0: allocated type
    // Needs custom serialisation: not stored in the instruction's operand list.
    //
    // operand_kind:
    OutStreamer.emitInt8(OperandKind::Type);
    // type_index
    OutStreamer.emitSizeT(typeIndex(I->getAllocatedType()));

    // OPERAND 1: number of objects to allocate
    Value *Op0 = I->getOperand(0);
    assert(isa<ConstantInt>(Op0));
    serialiseOperand(I, VLMap, Op0);

    VLMap[I] = {BBIdx, InstIdx};
    InstIdx++;
  }

  void serialiseCallInst(CallInst *I, ValueLoweringMap &VLMap, unsigned BBIdx,
                         unsigned &InstIdx) {
    // type_index:
    OutStreamer.emitSizeT(typeIndex(I->getType()));
    // opcode:
    serialiseOpcode(OpCode::Call);
    // num_operands:
    unsigned NumOpers = I->getNumOperands();
    OutStreamer.emitInt32(NumOpers);

    // OPERAND 0: What to call.
    //
    // In LLVM IR this is the final operand, which is a cause of confusion.
    serialiseOperand(I, VLMap, I->getOperand(NumOpers - 1));

    // Now the rest of the operands.
    for (unsigned OI = 0; OI < NumOpers - 1; OI++) {
      serialiseOperand(I, VLMap, I->getOperand(OI));
    }

    VLMap[I] = {BBIdx, InstIdx};
    InstIdx++;
  }

  void serialiseBranchInst(BranchInst *I, ValueLoweringMap &VLMap,
                           unsigned BBIdx, unsigned &InstIdx) {
    // We split LLVM's `br` into two Yk IR instructions: one for unconditional
    // branching, another for conidtional branching.
    if (!I->isConditional()) {
      // type_index:
      OutStreamer.emitSizeT(typeIndex(I->getType()));
      // opcode:
      serialiseOpcode(OpCode::Br);
      // num_operands:
      // We don't serialise any operands, because traces will guide us.
      OutStreamer.emitInt32(0);

      VLMap[I] = {BBIdx, InstIdx};
      InstIdx++;
    } else {
      // We DO need operands for conditional branches, so that we can build
      // guards.
      serialiseInstGeneric(I, VLMap, BBIdx, InstIdx, OpCode::CondBr);
    }
  }

  void serialiseInst(Instruction *I, ValueLoweringMap &VLMap, unsigned BBIdx,
                     unsigned &InstIdx) {
// Macros to help dispatch to serialisers.
//
// Note that this is unhygenic so as to make the call-sites readable.
#define GENERIC_INST_SERIALISE(LLVM_INST, LLVM_INST_TYPE, YKIR_OPCODE)         \
  if (isa<LLVM_INST_TYPE>(LLVM_INST)) {                                        \
    serialiseInstGeneric(LLVM_INST, VLMap, BBIdx, InstIdx, YKIR_OPCODE);       \
    return;                                                                    \
  }
#define CUSTOM_INST_SERIALISE(LLVM_INST, LLVM_INST_TYPE, SERIALISER)           \
  if (LLVM_INST_TYPE *II = dyn_cast<LLVM_INST_TYPE>(LLVM_INST)) {              \
    SERIALISER(II, VLMap, BBIdx, InstIdx);                                     \
    return;                                                                    \
  }

    GENERIC_INST_SERIALISE(I, LoadInst, Load)
    GENERIC_INST_SERIALISE(I, StoreInst, Store)
    GENERIC_INST_SERIALISE(I, GetElementPtrInst, GetElementPtr)
    GENERIC_INST_SERIALISE(I, ICmpInst, ICmp)
    GENERIC_INST_SERIALISE(I, llvm::BinaryOperator, BinaryOperator)
    GENERIC_INST_SERIALISE(I, ReturnInst, Ret)

    CUSTOM_INST_SERIALISE(I, AllocaInst, serialiseAllocaInst)
    CUSTOM_INST_SERIALISE(I, CallInst, serialiseCallInst)
    CUSTOM_INST_SERIALISE(I, BranchInst, serialiseBranchInst)

    // GENERIC_INST_SERIALISE and CUSTOM_INST_SERIALISE do an early return upon
    // a match, so if we get here then the instruction wasn't handled.
    serialiseUnimplementedInstruction(I, VLMap, BBIdx, InstIdx);
  }

  // An unimplemented instruction is lowered to an instruction with one
  // unimplemented operand containing the textual LLVM IR we couldn't handle.
  void serialiseUnimplementedInstruction(Instruction *I,
                                         ValueLoweringMap &VLMap,
                                         unsigned BBIdx, unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(UnimplementedInstruction);
    // num_operands:
    OutStreamer.emitInt32(1);
    // problem instruction:
    serialiseUnimplementedOperand(I);

    if (!I->getType()->isVoidTy()) {
      VLMap[I] = {BBIdx, InstIdx};
    }
    InstIdx++;
  }

  void serialiseBlock(BasicBlock &BB, ValueLoweringMap &VLMap,
                      unsigned &BBIdx) {
    // num_instrs:
    OutStreamer.emitSizeT(BB.size());
    // instrs:
    unsigned InstIdx = 0;
    for (Instruction &I : BB) {
      serialiseInst(&I, VLMap, BBIdx, InstIdx);
    }
    BBIdx++;
  }

  void serialiseArg(Argument *A) {
    // type_index:
    OutStreamer.emitSizeT(typeIndex(A->getType()));
  }

  void serialiseFunc(llvm::Function &F) {
    // name:
    serialiseString(F.getName());
    // type_idx:
    OutStreamer.emitSizeT(typeIndex(F.getFunctionType()));
    F.getType()->dump();
    // num_blocks:
    OutStreamer.emitSizeT(F.size());
    // blocks:
    unsigned BBIdx = 0;
    ValueLoweringMap VLMap;
    for (BasicBlock &BB : F) {
      serialiseBlock(BB, VLMap, BBIdx);
    }
  }

  void serialiseFunctionType(FunctionType *Ty) {
    OutStreamer.emitInt8(TypeKind::FunctionTy);
    // num_args:
    OutStreamer.emitSizeT(Ty->getNumParams());
    // arg_tys:
    for (llvm::Type *SubTy : Ty->params()) {
      OutStreamer.emitSizeT(typeIndex(SubTy));
    }
    // ret_ty:
    OutStreamer.emitSizeT(typeIndex(Ty->getReturnType()));
    // is_vararg:
    OutStreamer.emitInt8(Ty->isVarArg());
  }

  void serialiseType(llvm::Type *Ty) {
    if (Ty->isVoidTy()) {
      OutStreamer.emitInt8(TypeKind::Void);
    } else if (Ty->isPointerTy()) {
      OutStreamer.emitInt8(TypeKind::Ptr);
    } else if (IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
      OutStreamer.emitInt8(TypeKind::Integer);
      OutStreamer.emitInt32(ITy->getBitWidth());
    } else if (FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
      serialiseFunctionType(FTy);
    } else {
      OutStreamer.emitInt8(TypeKind::UnimplementedType);
      serialiseString(toString(Ty));
    }
  }

  void serialiseConstantInt(ConstantInt *CI) {
    OutStreamer.emitSizeT(typeIndex(CI->getType()));
    OutStreamer.emitSizeT(CI->getBitWidth() / 8);
    for (size_t I = 0; I < CI->getBitWidth(); I += 8) {
      uint64_t Byte = CI->getValue().extractBitsAsZExtValue(8, I);
      OutStreamer.emitInt8(Byte);
    }
  }

  void serialiseUnimplementedConstant(class Constant *C) {
    // type_index:
    OutStreamer.emitSizeT(typeIndex(C->getType()));
    // num_bytes:
    // Just report zero for now.
    OutStreamer.emitSizeT(0);
  }

  void serialiseConstant(class Constant *C) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
      serialiseConstantInt(CI);
    } else {
      serialiseUnimplementedConstant(C);
    }
  }

  void serialise() {
    // header:
    OutStreamer.emitInt32(Magic);
    OutStreamer.emitInt32(Version);

    // num_funcs:
    OutStreamer.emitSizeT(M.size());
    // funcs:
    for (llvm::Function &F : M) {
      serialiseFunc(F);
    }

    // num_constants:
    OutStreamer.emitSizeT(Constants.size());
    // constants:
    for (class Constant *&C : Constants) {
      serialiseConstant(C);
    }

    // num_types:
    OutStreamer.emitSizeT(Types.size());
    // types:
    for (llvm::Type *&Ty : Types) {
      serialiseType(Ty);
    }
  }
};
} // anonymous namespace

// Create an ELF section for storing Yk IR into.
MCSection *createYkIRSection(MCContext &Ctx, const MCSection *TextSec) {
  if (Ctx.getObjectFileType() != MCContext::IsELF)
    return nullptr;

  const MCSectionELF *ElfSec = static_cast<const MCSectionELF *>(TextSec);
  unsigned Flags = ELF::SHF_LINK_ORDER;
  StringRef GroupName;

  // Ensure the loader loads it.
  Flags |= ELF::SHF_ALLOC;

  return Ctx.getELFSection(SectionName, ELF::SHT_LLVM_BB_ADDR_MAP, Flags, 0,
                           GroupName, true, ElfSec->getUniqueID(),
                           cast<MCSymbolELF>(TextSec->getBeginSymbol()));
}

// Emit a start/end IR marker.
//
// The JIT uses a start and end marker to make a Rust slice of the IR.
void emitStartOrEndSymbol(MCContext &MCtxt, MCStreamer &OutStreamer,
                          bool Start) {
  std::string SymName("ykllvm.yk_ir.");
  if (Start)
    SymName.append("start");
  else
    SymName.append("stop");

  MCSymbol *Sym = MCtxt.getOrCreateSymbol(SymName);
  OutStreamer.emitSymbolAttribute(Sym, llvm::MCSA_Global);
  OutStreamer.emitLabel(Sym);
}

namespace llvm {

// Emit Yk IR into the resulting ELF binary.
void embedYkIR(MCContext &Ctx, MCStreamer &OutStreamer, Module &M) {
  MCSection *YkIRSec =
      createYkIRSection(Ctx, std::get<0>(OutStreamer.getCurrentSection()));

  OutStreamer.pushSection();
  OutStreamer.switchSection(YkIRSec);
  emitStartOrEndSymbol(Ctx, OutStreamer, true);
  YkIRWriter(M, OutStreamer).serialise();
  emitStartOrEndSymbol(Ctx, OutStreamer, false);
  OutStreamer.popSection();
}
} // namespace llvm
