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
  Branch,
  ICmp,
  BinaryOperator,
  Ret,
  UnimplementedInstruction = 255, // YKFIXME: Will eventually be deleted.
};

enum OperandKind {
  Constant = 0,
  LocalVariable,
  String,
};

enum TypeKind {
  Integer = 0,
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

#define GENERIC_INST_SERIALISE(LLVM_INST, LLVM_INST_TYPE, YKIR_OPCODE)         \
  if (isa<LLVM_INST_TYPE>(LLVM_INST)) {                                        \
    serialiseInstGeneric(LLVM_INST, YKIR_OPCODE);                              \
    return;                                                                    \
  }

class YkIRWriter {
private:
  Module &M;
  MCStreamer &OutStreamer;

  vector<llvm::Type *> Types;
  vector<llvm::Constant *> Constants;

  // Return the index of the LLVM type `Ty`, inserting a new entry if
  // necessary.
  size_t typeIndex(Type *Ty) {
    vector<Type *>::iterator Found = std::find(Types.begin(), Types.end(), Ty);
    if (Found != Types.end()) {
      return std::distance(Types.begin(), Found);
    }
    size_t Idx = Types.size();
    Types.push_back(Ty);
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

  void serialiseLocalVariableOperand(Instruction *I) {
    // For now we assume that there is a one to one relationship between LLVM
    // instructions and Yk IR instructions, and that the instruction
    // (and block) indices are the same in both IRs.
    BasicBlock *ParentBlock = I->getParent();
    Function *ParentFunc = ParentBlock->getParent();

    size_t BlockIdx = getIndex(ParentFunc, ParentBlock);
    size_t InstIdx = getIndex(ParentBlock, I);

    OutStreamer.emitInt8(OperandKind::LocalVariable);
    OutStreamer.emitSizeT(BlockIdx);
    OutStreamer.emitSizeT(InstIdx);
  }

  void serialiseStringOperand(const char *S) {
    OutStreamer.emitInt8(OperandKind::String);
    serialiseString(S);
  }

  // YKFIXME: This allows programs which we haven't yet defined a
  // lowering for to compile. For now We just emit a string operand containing
  // the unhandled LLVM operand in textual form.
  void serialiseUnimplementedOperand(Value *V) {
    OutStreamer.emitInt8(OperandKind::String);
    serialiseString(toString(V));
  }

  void serialiseOperand(Instruction *Parent, Value *V) {
    if (llvm::Constant *C = dyn_cast<llvm::Constant>(V)) {
      serialiseConstantOperand(Parent, C);
    } else if (Instruction *I = dyn_cast<Instruction>(V)) {
      // If an instruction defines the operand, it's a local variable.
      serialiseLocalVariableOperand(I);
    } else {
      serialiseUnimplementedOperand(V);
    }
  }

  /// Does a naiave serialisation of an LLVM instruction by iterating over its
  /// operands and serialising them in turn.
  void serialiseInstGeneric(Instruction *I, OpCode Opc) {
    OutStreamer.emitSizeT(typeIndex(I->getType()));
    serialiseOpcode(Opc);
    OutStreamer.emitInt32(I->getNumOperands());
    for (Value *O : I->operands()) {
      serialiseOperand(I, O);
    }
  }

  void serialiseInst(Instruction *I) {
    GENERIC_INST_SERIALISE(I, LoadInst, Load)
    GENERIC_INST_SERIALISE(I, StoreInst, Store)
    GENERIC_INST_SERIALISE(I, AllocaInst, Alloca)
    GENERIC_INST_SERIALISE(I, CallInst, Call)
    GENERIC_INST_SERIALISE(I, GetElementPtrInst, GetElementPtr)
    GENERIC_INST_SERIALISE(I, BranchInst, Branch)
    GENERIC_INST_SERIALISE(I, ICmpInst, ICmp)
    GENERIC_INST_SERIALISE(I, llvm::BinaryOperator, BinaryOperator)
    GENERIC_INST_SERIALISE(I, ReturnInst, Ret)

    // GENERIC_INST_SERIALISE does an early return upon a match, so if we get
    // here then the instruction wasn't handled.
    serialiseUnimplementedInstruction(I);
  }

  void serialiseUnimplementedInstruction(Instruction *I) {
    // opcode:
    serialiseOpcode(UnimplementedInstruction);
    // num_operands:
    OutStreamer.emitInt32(1);
    // problem instruction:
    serialiseStringOperand(toString(I).data());
  }

  void serialiseBlock(BasicBlock &BB) {
    // num_instrs:
    OutStreamer.emitSizeT(BB.size());
    // instrs:
    for (Instruction &I : BB) {
      serialiseInst(&I);
    }
  }

  void serialiseFunc(Function &F) {
    // name:
    serialiseString(F.getName());
    // num_blocks:
    OutStreamer.emitSizeT(F.size());
    // blocks:
    for (BasicBlock &BB : F) {
      serialiseBlock(BB);
    }
  }

  void serialiseType(Type *Ty) {
    if (IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
      OutStreamer.emitInt8(TypeKind::Integer);
      OutStreamer.emitInt32(ITy->getBitWidth());
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
    for (Function &F : M) {
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
    for (Type *&Ty : Types) {
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
