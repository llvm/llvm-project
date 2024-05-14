//===- YkIR/YkIRWRiter.cpp -- Yk JIT IR Serialiaser---------------------===//
//
// Converts an LLVM module into Yk's on-disk AOT IR.
//
//===-------------------------------------------------------------------===//

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
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

#define YK_OUTLINE_FNATTR "yk_outline"
const char *SectionName = ".yk_ir";
const uint32_t Magic = 0xedd5f00d;
const uint32_t Version = 0;

enum OpCode {
  OpCodeNop = 0,
  OpCodeLoad,
  OpCodeStore,
  OpCodeAlloca,
  OpCodeCall,
  OpCodeBr,
  OpCodeCondBr,
  OpCodeICmp,
  OpCodeRet,
  OpCodeInsertValue,
  OpCodePtrAdd,
  OpCodeBinOp,
  OpCodeCast,
  OpCodeSwitch,
  OpCodePHI,
  OpCodeIndirectCall,
  OpCodeUnimplemented = 255, // YKFIXME: Will eventually be deleted.
};

enum OperandKind {
  OperandKindConstant = 0,
  OperandKindLocal,
  OperandKindGlobal,
  OperandKindFunction,
  OperandKindArg,
};

enum TypeKind {
  TypeKindVoid = 0,
  TypeKindInteger,
  TypeKindPtr,
  TypeKindFunction,
  TypeKindStruct,
  TypeKindUnimplemented = 255, // YKFIXME: Will eventually be deleted.
};

enum CastKind {
  CastKindSignExt = 0,
  CastKindZeroExt = 1,
};

// A predicate used in a numeric comparison.
enum CmpPredicate {
  PredEqual = 0,
  PredNotEqual,
  PredUnsignedGreater,
  PredUnsignedGreaterEqual,
  PredUnsignedLess,
  PredUnsignedLessEqual,
  PredSignedGreater,
  PredSignedGreaterEqual,
  PredSignedLess,
  PredSignedLessEqual,
};

// A binary operator.
enum BinOp {
  BinOpAdd,
  BinOpSub,
  BinOpMul,
  BinOpOr,
  BinOpAnd,
  BinOpXor,
  BinOpShl,
  BinOpAShr,
  BinOpFAdd,
  BinOpFDiv,
  BinOpFMul,
  BinOpFRem,
  BinOpFSub,
  BinOpLShr,
  BinOpSDiv,
  BinOpSRem,
  BinOpUDiv,
  BinOpURem,
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

// An instruction index that uniquely identifies an Yk instruction within
// a basic block.
//
// FIXME: At some point it may be worth making type-safe index types (for
// instruction, block and function indices) and using them throughout
// the serialiser.
using InstIdx = size_t;

// Maps an LLVM local (the instruction that creates it) to the correspoinding Yk
// instruction index in its parent basic block.
//
// Note: The Yk basic block index is not stored because it's the same as
// the LLVM IR block index, which can be found elsewhere (see `getIndex()`).
using ValueLoweringMap = map<Instruction *, InstIdx>;

// Function lowering context.
//
// This groups together some per-function lowering bits so that they can be
// passed down through the serialiser together (and so that they can die
// together).
class FuncLowerCtxt {
  // The local variable mapping for one function.
  ValueLoweringMap VLMap;
  // Local variable indices that require patching once we have finished
  // lowwering the function.
  vector<tuple<Instruction *, MCSymbol *>> InstIdxPatchUps;

public:
  // Create an empty function lowering context.
  FuncLowerCtxt() : VLMap(ValueLoweringMap()), InstIdxPatchUps({}){};

  // Defer (patch up later) the use-site of the (as-yet unknown) instruction
  // index of the instruction  `I`, which has the symbol `Sym`.
  void deferInstIdx(Instruction *I, MCSymbol *Sym) {
    InstIdxPatchUps.push_back({I, Sym});
  }

  // Fill in instruction indices that had to be deferred.
  void patchUpInstIdxs(MCStreamer &OutStreamer) {
    MCContext &MCtxt = OutStreamer.getContext();
    for (auto &[Inst, Sym] : InstIdxPatchUps) {
      InstIdx InstIdx = VLMap.at(Inst);
      OutStreamer.emitAssignment(Sym, MCConstantExpr::create(InstIdx, MCtxt));
    }
  }

  // Add/update an entry in the value lowering map.
  void updateVLMap(Instruction *I, InstIdx L) { VLMap[I] = L; }

  // Get the entry for `I` in the value lowering map.
  //
  // Raises `std::out_of_range` if not present.
  InstIdx lookupInVLMap(Instruction *I) { return VLMap.at(I); }

  // Determines if there's an entry for `I` in the value lowering map.
  bool vlMapContains(Instruction *I) { return VLMap.count(I) == 1; }
};

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
//    must update the `VLMap` (found inside the `FuncLowerCtxt`) with an entry
//    that maps the LLVM instruction to the final Yk IR instruction in the
//    lowering. If the LLVM instruction doesn't generate a value, or the LLVM
//    instruction lowered to exactly zero Yk IR instructions, then there is no
//    need to update the `VLMap`.
//
// These invariants are required so that when we encounter a local variable as
// an operand to an LLVM instruction, we can quickly find the corresponding Yk
// IR local variable.
class YkIRWriter {
private:
  Module &M;
  MCStreamer &OutStreamer;
  DataLayout DL;

  vector<llvm::Type *> Types;
  vector<llvm::Constant *> Constants;
  vector<llvm::GlobalVariable *> Globals;

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
  size_t constantIndex(Constant *C) {
    vector<Constant *>::iterator Found =
        std::find(Constants.begin(), Constants.end(), C);
    if (Found != Constants.end()) {
      return std::distance(Constants.begin(), Found);
    }
    size_t Idx = Constants.size();
    Constants.push_back(C);
    return Idx;
  }

  // Return the index of the LLVM global `G`, inserting a new entry if
  // necessary.
  size_t globalIndex(GlobalVariable *G) {
    vector<GlobalVariable *>::iterator Found =
        std::find(Globals.begin(), Globals.end(), G);
    if (Found != Globals.end()) {
      return std::distance(Globals.begin(), Found);
    }
    size_t Idx = Globals.size();
    Globals.push_back(G);
    return Idx;
  }

  size_t functionIndex(llvm::Function *F) {
    // FIXME: For now we assume that function indicies in LLVM IR and our IR
    // are the same.
    return getIndex(&M, F);
  }

  // Serialises a null-terminated string.
  void serialiseString(StringRef S) {
    OutStreamer.emitBinaryData(S);
    OutStreamer.emitInt8(0); // null terminator.
  }

  void serialiseOpcode(OpCode Code) { OutStreamer.emitInt8(Code); }
  void serialiseOperandKind(OperandKind Kind) { OutStreamer.emitInt8(Kind); }
  void serialiseTypeKind(TypeKind Kind) { OutStreamer.emitInt8(Kind); }

  void serialiseConstantOperand(Instruction *Parent, llvm::Constant *C) {
    serialiseOperandKind(OperandKindConstant);
    OutStreamer.emitSizeT(constantIndex(C));
  }

  void serialiseLocalVariableOperand(Instruction *I, FuncLowerCtxt &FLCtxt) {
    // operand kind:
    serialiseOperandKind(OperandKindLocal);
    // func_idx:
    OutStreamer.emitSizeT(getIndex(&M, I->getFunction()));
    // bb_idx:
    OutStreamer.emitSizeT(getIndex(I->getFunction(), I->getParent()));

    // inst_idx:
    if (FLCtxt.vlMapContains(I)) {
      InstIdx InstIdx = FLCtxt.lookupInVLMap(I);
      OutStreamer.emitSizeT(InstIdx);
    } else {
      // It's a local variable generated by an instruction that we haven't
      // serialised yet. This can happen in loop bodies where a PHI node merges
      // in a variable from the end of the loop body.
      //
      // To work around this, we emit a dummy instruction index
      // and patch it up later once it becomes known.
      MCSymbol *PatchUpSym = OutStreamer.getContext().createTempSymbol();
      OutStreamer.emitSymbolValue(PatchUpSym, sizeof(size_t));
      FLCtxt.deferInstIdx(I, PatchUpSym);
    }
  }

  void serialiseFunctionOperand(llvm::Function *F) {
    serialiseOperandKind(OperandKindFunction);
    OutStreamer.emitSizeT(functionIndex(F));
  }

  void serialiseBlockLabel(BasicBlock *BB) {
    // Basic block indices are the same in both LLVM IR and our IR.
    OutStreamer.emitSizeT(getIndex(BB->getParent(), BB));
  }

  void serialiseArgOperand(Argument *A) {
    // This assumes that the argument indices match in both IRs.

    // opcode:
    serialiseOperandKind(OperandKindArg);
    // parent function index:
    OutStreamer.emitSizeT(getIndex(&M, A->getParent()));
    // arg index
    OutStreamer.emitSizeT(A->getArgNo());
  }

  void serialiseGlobalOperand(GlobalVariable *G) {
    serialiseOperandKind(OperandKindGlobal);
    OutStreamer.emitSizeT(globalIndex(G));
  }

  void serialiseOperand(Instruction *Parent, FuncLowerCtxt &FLCtxt, Value *V) {
    if (llvm::GlobalVariable *G = dyn_cast<llvm::GlobalVariable>(V)) {
      serialiseGlobalOperand(G);
    } else if (llvm::Function *F = dyn_cast<llvm::Function>(V)) {
      serialiseFunctionOperand(F);
    } else if (llvm::Constant *C = dyn_cast<llvm::Constant>(V)) {
      serialiseConstantOperand(Parent, C);
    } else if (llvm::Argument *A = dyn_cast<llvm::Argument>(V)) {
      serialiseArgOperand(A);
    } else if (Instruction *I = dyn_cast<Instruction>(V)) {
      serialiseLocalVariableOperand(I, FLCtxt);
    } else {
      llvm::report_fatal_error(
          StringRef("attempt to serialise non-yk-operand: " + toString(V)));
    }
  }

  void serialiseBinaryOperatorInst(llvm::BinaryOperator *I,
                                   FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                                   unsigned &InstIdx) {
    assert(I->getNumOperands() == 2);

    // opcode:
    serialiseOpcode(OpCodeBinOp);
    // left-hand side:
    serialiseOperand(I, FLCtxt, I->getOperand(0));
    // binary operator:
    serialiseBinOperator(I->getOpcode());
    // right-hand side:
    serialiseOperand(I, FLCtxt, I->getOperand(1));

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  // Serialise a binary operator.
  void serialiseBinOperator(Instruction::BinaryOps BO) {
    // operand kind:
    // OutStreamer.emitInt8(OperandKind::OpKindBinOp);
    // the operator:
    switch (BO) {
    case Instruction::BinaryOps::Add:
      OutStreamer.emitInt8(BinOp::BinOpAdd);
      break;
    case Instruction::BinaryOps::Sub:
      OutStreamer.emitInt8(BinOp::BinOpSub);
      break;
    case Instruction::BinaryOps::Mul:
      OutStreamer.emitInt8(BinOp::BinOpMul);
      break;
    case Instruction::BinaryOps::Or:
      OutStreamer.emitInt8(BinOp::BinOpOr);
      break;
    case Instruction::BinaryOps::And:
      OutStreamer.emitInt8(BinOp::BinOpAnd);
      break;
    case Instruction::BinaryOps::Xor:
      OutStreamer.emitInt8(BinOp::BinOpXor);
      break;
    case Instruction::BinaryOps::Shl:
      OutStreamer.emitInt8(BinOp::BinOpShl);
      break;
    case Instruction::BinaryOps::AShr:
      OutStreamer.emitInt8(BinOp::BinOpAShr);
      break;
    case Instruction::BinaryOps::FAdd:
      OutStreamer.emitInt8(BinOp::BinOpFAdd);
      break;
    case Instruction::BinaryOps::FDiv:
      OutStreamer.emitInt8(BinOp::BinOpFDiv);
      break;
    case Instruction::BinaryOps::FMul:
      OutStreamer.emitInt8(BinOp::BinOpFMul);
      break;
    case Instruction::BinaryOps::FRem:
      OutStreamer.emitInt8(BinOp::BinOpFRem);
      break;
    case Instruction::BinaryOps::FSub:
      OutStreamer.emitInt8(BinOp::BinOpFSub);
      break;
    case Instruction::BinaryOps::LShr:
      OutStreamer.emitInt8(BinOp::BinOpLShr);
      break;
    case Instruction::BinaryOps::SDiv:
      OutStreamer.emitInt8(BinOp::BinOpSDiv);
      break;
    case Instruction::BinaryOps::SRem:
      OutStreamer.emitInt8(BinOp::BinOpSRem);
      break;
    case Instruction::BinaryOps::UDiv:
      OutStreamer.emitInt8(BinOp::BinOpUDiv);
      break;
    case Instruction::BinaryOps::URem:
      OutStreamer.emitInt8(BinOp::BinOpURem);
      break;
    default:
      llvm::report_fatal_error("unknown binary operator");
    }
  }

  void serialiseAllocaInst(AllocaInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                           unsigned &InstIdx) {
    // We don't yet support:
    //  - the `inalloca` keyword.
    //  - non-zero address spaces.
    //  - dynamic alloca (because stackmaps can't handle them).
    //  - allocating an array with more than SIZE_MAX elements.
    if ((I->isUsedWithInAlloca()) || (I->getAddressSpace() != 0) ||
        (!isa<Constant>(I->getArraySize())) ||
        cast<ConstantInt>(I->getArraySize())->getValue().ugt(SIZE_MAX)) {
      serialiseUnimplementedInstruction(I, FLCtxt, BBIdx, InstIdx);
      return;
    }

    // opcode:
    serialiseOpcode(OpCodeAlloca);

    // type to be allocated:
    OutStreamer.emitSizeT(typeIndex(I->getAllocatedType()));

    // number of objects to allocate
    ConstantInt *CI = cast<ConstantInt>(I->getArraySize());
    static_assert(sizeof(size_t) <= sizeof(uint64_t));
    OutStreamer.emitSizeT(CI->getZExtValue());

    // align:
    OutStreamer.emitInt64(I->getAlign().value());

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  void serialiseStackmapCall(CallInst *I, FuncLowerCtxt &FLCtxt) {
    assert(I);
    assert(I->getCalledFunction()->isIntrinsic());
    assert(I->getIntrinsicID() == Intrinsic::experimental_stackmap);
    // stackmap ID:
    serialiseOperand(I, FLCtxt, I->getOperand(0));

    // num_lives:
    OutStreamer.emitInt32(I->arg_size() - 2);

    // lives:
    for (unsigned OI = 2; OI < I->arg_size(); OI++) {
      serialiseOperand(I, FLCtxt, I->getOperand(OI));
    }
  }

  void serialiseIndirectCallInst(CallInst *I, FuncLowerCtxt &FLCtxt,
                                 unsigned BBIdx, unsigned &InstIdx) {

    serialiseOpcode(OpCodeIndirectCall);
    // function type:
    OutStreamer.emitSizeT(typeIndex(I->getFunctionType()));
    // callee (operand):
    serialiseOperand(I, FLCtxt, I->getCalledOperand());
    // num_args:
    // (this includes static and varargs arguments)
    OutStreamer.emitInt32(I->arg_size());
    // args:
    for (unsigned OI = 0; OI < I->arg_size(); OI++) {
      serialiseOperand(I, FLCtxt, I->getOperand(OI));
    }

    // If the return type is non-void, then this defines a local.
    if (!I->getType()->isVoidTy()) {
      FLCtxt.updateVLMap(I, InstIdx);
    }
    InstIdx++;
  }

  void serialiseCallInst(CallInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                         unsigned &InstIdx) {
    if (I->isInlineAsm()) {
      // For now we omit calls to empty inline asm blocks.
      //
      // These are pretty much always present in yk unit tests to block
      // optimisations.
      // if (!(cast<InlineAsm>(Callee)->getAsmString().empty())) {
      if (!(cast<InlineAsm>(I->getCalledOperand())->getAsmString().empty())) {
        // Non-empty asm block. We can't ignore it.
        serialiseUnimplementedInstruction(I, FLCtxt, BBIdx, InstIdx);
      }
      return;
    }

    if (I->isIndirectCall()) {
      serialiseIndirectCallInst(I, FLCtxt, BBIdx, InstIdx);
      return;
    }

    // Stackmap calls are serialised on-demand by folding them into the `call`
    // or `condbr` instruction which they belong to.
    if (I->getCalledFunction()->isIntrinsic() &&
        I->getIntrinsicID() == Intrinsic::experimental_stackmap) {
      return;
    }

    // FIXME: Note that this assertion can fail if you do a direct call without
    // the correct type annotation at the call site.
    //
    // e.g. for a functiion:
    //
    //   define i32 @f(i32, ...)
    //
    // if you do:
    //
    //   call i32 @f(1i32, 2i32);
    //
    // instead of:
    //
    //   call i32 (i32, ...) @f(1i32, 2i32);
    assert(I->getCalledFunction());

    serialiseOpcode(OpCodeCall);
    // callee:
    OutStreamer.emitSizeT(functionIndex(I->getCalledFunction()));
    // num_args:
    // (this includes static and varargs arguments)
    OutStreamer.emitInt32(I->arg_size());
    // args:
    for (unsigned OI = 0; OI < I->arg_size(); OI++) {
      serialiseOperand(I, FLCtxt, I->getOperand(OI));
    }
    if (!I->getCalledFunction()->isDeclaration()) {
      // The next instruction will be the stackmap entry
      // has_safepoint = 1:
      OutStreamer.emitInt8(1);
      CallInst *SMI = dyn_cast<CallInst>(I->getNextNonDebugInstruction());
      serialiseStackmapCall(SMI, FLCtxt);
    } else {
      // has_safepoint = 0:
      OutStreamer.emitInt8(0);
    }

    // If the return type is non-void, then this defines a local.
    if (!I->getType()->isVoidTy()) {
      FLCtxt.updateVLMap(I, InstIdx);
    }
    InstIdx++;
  }

  void serialiseBranchInst(BranchInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                           unsigned &InstIdx) {
    // We split LLVM's `br` into two Yk IR instructions: one for unconditional
    // branching, another for conidtional branching.
    if (!I->isConditional()) {
      // We don't serialise the branch target for unconditional branches because
      // traces will guide us.
      //
      // opcode:
      serialiseOpcode(OpCodeBr);
      // successor:
      serialiseBlockLabel(I->getSuccessor(0));
    } else {
      // opcode:
      serialiseOpcode(OpCodeCondBr);
      // We DO need operands for conditional branches, so that we can build
      // guards.
      //
      // cond:
      serialiseOperand(I, FLCtxt, I->getCondition());
      // true_bb:
      serialiseBlockLabel(I->getSuccessor(0));
      // false_bb:
      serialiseBlockLabel(I->getSuccessor(1));

      CallInst *SMI = dyn_cast<CallInst>(I->getPrevNonDebugInstruction());
      serialiseStackmapCall(SMI, FLCtxt);
    }
    InstIdx++;
  }

  void serialiseLoadInst(LoadInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                         unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeLoad);
    // ptr:
    serialiseOperand(I, FLCtxt, I->getPointerOperand());
    // type_idx:
    OutStreamer.emitSizeT(typeIndex(I->getType()));

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  void serialiseStoreInst(StoreInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                          unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeStore);
    // value:
    serialiseOperand(I, FLCtxt, I->getValueOperand());
    // ptr:
    serialiseOperand(I, FLCtxt, I->getPointerOperand());

    InstIdx++;
  }

  void serialiseGetElementPtrInst(GetElementPtrInst *I, FuncLowerCtxt &FLCtxt,
                                  unsigned BBIdx, unsigned &InstIdx) {
    // We yet don't support:
    //  - the `inrange` keyword.
    //  - the vector variant.
    //  - exotic (non-zero) address spaces.
    //
    // It appears that `inrange` can't appear in a GEP *instruction* (only a
    // GEP expression, inline in another instruction), but we check for it
    // anyway.
    if ((cast<GEPOperator>(I)->getInRangeIndex() != nullopt) ||
        (I->getPointerOperand()->getType()->isVectorTy()) ||
        (I->getPointerAddressSpace() != 0)) {
      serialiseUnimplementedInstruction(I, FLCtxt, BBIdx, InstIdx);
      return;
    }

    // opcode:
    serialiseOpcode(OpCodePtrAdd);
    // type_idx:
    OutStreamer.emitSizeT(typeIndex(I->getType()));
    // pointer:
    serialiseOperand(I, FLCtxt, I->getPointerOperand());

    // GetElementPtrInst::collectOffset() reduces the GEP to:
    //  - a static byte offset and
    //  - zero or more dynamic byte offsets of the form `elem_count *
    //    elem_size`, where `elem_count` is not known statically.
    //
    // We encode this information into our Yk AOT `PtrAdd` instruction, but
    // there are some semantics of LLVM's GEP that we have to be very careful to
    // preserve. At the time of writing, these are the things that it's
    // important to know:
    //
    //  1. LLVM integer types are neither signed nor unsigned. They are a bit
    //     pattern that can be interpreted (by LLVM instructions) as a signed
    //     or unsigned integer.
    //
    //  2. a dynamic index cannot be applied to a struct, because struct fields
    //     can have different types and you wouldn't be able to statically
    //     determine the type of the field being selected.
    //
    //  3. When indexing a struct, the index is interpreted as unsigned
    //     (because a negative field index makes no sense).
    //
    //  4. When indexing anything but a struct, the index is interpreted as
    //     signed, to allow (e.g.) negative array indices, or negative
    //     offsetting pointers.
    //
    //  5. Index operands to `getelementptr` can have arbitrary bit-width
    //     (although struct indexing must use i32). Index types with a different
    //     bit-width than the "pointer indexing type" for the address space in
    //     question must be extended or truncated (and if it's a signed index,
    //     then that's a sign extend!). To get the indexing type, you use
    //     `DataLayout:getIndexSizeInBits()`.
    //
    //  6. We can ignore the `inbounds` keyword on GEPs. When an `inbounds` GEP
    //     is out of bounds, a poison value is generated. Since a poison value
    //     represents (deferred) undefined behaviour (UB), we are free to
    //     compute any value we want, including the out of bounds offset.
    //
    // To simplify things a bit, we assume (as is that case for "regular"
    // hardware/software platforms) that the LLVM pointer indexing type is the
    // same size as a pointer. Just in case, let's assert it though:
    unsigned IdxBitWidth = DL.getIndexSizeInBits(I->getPointerAddressSpace());
    assert(sizeof(void *) * 8 == IdxBitWidth);
    //// And since we are going to use `get{S,Z}ExtValue()`, which return
    //// `uint64_t` and `int64_t`, we should also check:
    static_assert(sizeof(size_t) <= sizeof(uint64_t));

    APInt ConstOff(IdxBitWidth, 0);
    MapVector<Value *, APInt> DynOffs;
    // Note: the width of the collected constant offset must be the same as the
    // index type bit-width.
    bool CollectRes = I->collectOffset(DL, IdxBitWidth, DynOffs, ConstOff);
    assert(CollectRes);

    // const_off:
    //
    // This is always signed and we can statically sign-extend now.
    //
    // FIXME: We can't deal with static offsets that don't fit in a ssize_t.
    assert(ConstOff.sle(APInt(sizeof(size_t) * 8, SSIZE_MAX)));
    OutStreamer.emitSizeT(ConstOff.getSExtValue());
    // num_dyn_offs:
    size_t NumDyn = DynOffs.size();
    OutStreamer.emitSizeT(NumDyn);
    // dyn_elem_counts:
    //
    // These are interpreted as signed.
    for (auto &KV : DynOffs) {
      serialiseOperand(I, FLCtxt, std::get<0>(KV));
    }
    // dyn_elem_sizes:
    //
    // These are unsigned and we can statically zero-extend now.
    for (auto &KV : DynOffs) {
      APInt DS = std::get<1>(KV);
      // FIXME: We can't deal with element sizes that don't fit in a size_t.
      assert(DS.ule(APInt(sizeof(size_t) * 8, SIZE_MAX)));
      OutStreamer.emitSizeT(DS.getZExtValue());
    }

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  // Serialise an LLVM predicate.
  void serialisePredicate(llvm::CmpInst::Predicate P) {
    std::optional<CmpPredicate> LP = std::nullopt;
    switch (P) {
    case llvm::CmpInst::ICMP_EQ:
      LP = PredEqual;
      break;
    case llvm::CmpInst::ICMP_NE:
      LP = PredNotEqual;
      break;
    case llvm::CmpInst::ICMP_UGT:
      LP = PredUnsignedGreater;
      break;
    case llvm::CmpInst::ICMP_UGE:
      LP = PredUnsignedGreaterEqual;
      break;
    case llvm::CmpInst::ICMP_ULT:
      LP = PredUnsignedLess;
      break;
    case llvm::CmpInst::ICMP_ULE:
      LP = PredUnsignedLessEqual;
      break;
    case llvm::CmpInst::ICMP_SGT:
      LP = PredSignedGreater;
      break;
    case llvm::CmpInst::ICMP_SGE:
      LP = PredSignedGreaterEqual;
      break;
    case llvm::CmpInst::ICMP_SLT:
      LP = PredSignedLess;
      break;
    case llvm::CmpInst::ICMP_SLE:
      LP = PredSignedLessEqual;
      break;
    default:
      abort(); // TODO: floating point predicates.
    }
    OutStreamer.emitInt8(LP.value());
  }

  void serialiseICmpInst(ICmpInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                         unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeICmp);
    // type_idx:
    OutStreamer.emitSizeT(typeIndex(I->getType()));
    // lhs:
    serialiseOperand(I, FLCtxt, I->getOperand(0));
    // predicate:
    serialisePredicate(I->getPredicate());
    // rhs:
    serialiseOperand(I, FLCtxt, I->getOperand(1));

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  void serialiseReturnInst(ReturnInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                           unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeRet);

    Value *RV = I->getReturnValue();
    if (RV == nullptr) {
      // has_val = 0:
      OutStreamer.emitInt8(0);
    } else {
      // has_val = 1:
      OutStreamer.emitInt8(1);
      // value:
      serialiseOperand(I, FLCtxt, RV);
    }

    InstIdx++;
  }

  void serialiseInsertValueInst(InsertValueInst *I, FuncLowerCtxt &FLCtxt,
                                unsigned BBIdx, unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeInsertValue);
    // agg:
    serialiseOperand(I, FLCtxt, I->getAggregateOperand());
    // elem:
    serialiseOperand(I, FLCtxt, I->getInsertedValueOperand());

    InstIdx++;
  }

  void serialiseCastKind(enum CastKind Cast) { OutStreamer.emitInt8(Cast); }

  /// Serialise a cast-like instruction.
  void serialiseSExtInst(SExtInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                         unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeCast);
    // cast_kind:
    serialiseCastKind(CastKindSignExt);
    // val:
    serialiseOperand(I, FLCtxt, I->getOperand(0));
    // dest_type_idx:
    OutStreamer.emitSizeT(typeIndex(I->getDestTy()));

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  /// Serialise a cast-like instruction.
  void serialiseZExtInst(ZExtInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                         unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeCast);
    // cast_kind:
    serialiseCastKind(CastKindZeroExt);
    // val:
    serialiseOperand(I, FLCtxt, I->getOperand(0));
    // dest_type_idx:
    OutStreamer.emitSizeT(typeIndex(I->getDestTy()));

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  void serialiseSwitchInst(SwitchInst *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                           unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeSwitch);
    // test_val:
    serialiseOperand(I, FLCtxt, I->getCondition());
    // default_dest:
    serialiseBlockLabel(I->getDefaultDest());
    // num_cases:
    assert(I->getNumCases() == std::distance(I->case_begin(), I->case_end()));
    OutStreamer.emitSizeT(I->getNumCases());
    // case_values:
    for (auto &Case : I->cases()) {
      // We can't handle case values larger than uint64_t for now.
      assert(Case.getCaseValue()->getBitWidth() <= 64);
      OutStreamer.emitInt64(Case.getCaseValue()->getZExtValue());
    }
    // case_dests:
    for (auto &Case : I->cases()) {
      serialiseBlockLabel(Case.getCaseSuccessor());
    }
    // safepoint:
    CallInst *SMI = dyn_cast<CallInst>(I->getPrevNonDebugInstruction());
    serialiseStackmapCall(SMI, FLCtxt);
    InstIdx++;
  }

  void serialisePhiInst(PHINode *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                        unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodePHI);
    // num_incoming:
    size_t NumIncoming = I->getNumIncomingValues();
    OutStreamer.emitSizeT(NumIncoming);
    // incoming_bbs:
    for (size_t J = 0; J < NumIncoming; J++) {
      serialiseBlockLabel(I->getIncomingBlock(J));
    }
    // incoming_vals:
    for (size_t J = 0; J < NumIncoming; J++) {
      serialiseOperand(I, FLCtxt, I->getIncomingValue(J));
    }

    FLCtxt.updateVLMap(I, InstIdx);
    InstIdx++;
  }

  void serialiseInst(Instruction *I, FuncLowerCtxt &FLCtxt, unsigned BBIdx,
                     unsigned &InstIdx) {
    // Macro to make the dispatch below easier to read/sort.
#define INST_SERIALISE(LLVM_INST, LLVM_INST_TYPE, SERIALISER)                  \
  if (LLVM_INST_TYPE *II = dyn_cast<LLVM_INST_TYPE>(LLVM_INST)) {              \
    SERIALISER(II, FLCtxt, BBIdx, InstIdx);                                    \
    return;                                                                    \
  }

    INST_SERIALISE(I, AllocaInst, serialiseAllocaInst);
    INST_SERIALISE(I, BinaryOperator, serialiseBinaryOperatorInst);
    INST_SERIALISE(I, BranchInst, serialiseBranchInst);
    INST_SERIALISE(I, CallInst, serialiseCallInst);
    INST_SERIALISE(I, GetElementPtrInst, serialiseGetElementPtrInst);
    INST_SERIALISE(I, ICmpInst, serialiseICmpInst);
    INST_SERIALISE(I, InsertValueInst, serialiseInsertValueInst);
    INST_SERIALISE(I, LoadInst, serialiseLoadInst);
    INST_SERIALISE(I, PHINode, serialisePhiInst);
    INST_SERIALISE(I, ReturnInst, serialiseReturnInst);
    INST_SERIALISE(I, ZExtInst, serialiseZExtInst);
    INST_SERIALISE(I, SExtInst, serialiseSExtInst);
    INST_SERIALISE(I, StoreInst, serialiseStoreInst);
    INST_SERIALISE(I, SwitchInst, serialiseSwitchInst);

    // INST_SERIALISE does an early return upon a match, so if we get here then
    // the instruction wasn't handled.
    serialiseUnimplementedInstruction(I, FLCtxt, BBIdx, InstIdx);
  }

  void serialiseUnimplementedInstruction(Instruction *I, FuncLowerCtxt &FLCtxt,
                                         unsigned BBIdx, unsigned &InstIdx) {
    // opcode:
    serialiseOpcode(OpCodeUnimplemented);
    // stringified problem instruction
    serialiseString(toString(I));

    if (!I->getType()->isVoidTy()) {
      FLCtxt.updateVLMap(I, InstIdx);
    }
    InstIdx++;
  }

  void serialiseBlock(BasicBlock &BB, FuncLowerCtxt &FLCtxt, unsigned &BBIdx) {
    auto ShouldSkipInstr = [](Instruction *I) {
      // Skip non-semantic instrucitons for now.
      //
      // We may come back to them later if we need better debugging
      // facilities, but for now they just clutter up our AOT module.
      if (I->isDebugOrPseudoInst()) {
        return true;
      }

      // See serialiseCallInst() for details.
      if (CallInst *CI = dyn_cast<CallInst>(I)) {
        if (InlineAsm *IA = dyn_cast<InlineAsm>(CI->getCalledOperand())) {
          return IA->getAsmString().empty();
        }
      }

      return false;
    };

    // num_instrs:
    //
    // We don't know how many instructions there will be in advance, so what we
    // do is emit a placeholder field (in the form of a symbol value) which is
    // patched up (assigned) later.
    MCContext &MCtxt = OutStreamer.getContext();
    MCSymbol *NumInstrsSym = MCtxt.createTempSymbol();
    OutStreamer.emitSymbolValue(NumInstrsSym, sizeof(size_t));

    // instrs:
    unsigned InstIdx = 0;
    for (Instruction &I : BB) {
      if (ShouldSkipInstr(&I)) {
        continue;
      }
      serialiseInst(&I, FLCtxt, BBIdx, InstIdx);
    }

    // Now that we have finished serialising instructions, we know how many
    // there are and we can patch up the "number of instructions" field.
    OutStreamer.emitAssignment(NumInstrsSym,
                               MCConstantExpr::create(InstIdx, MCtxt));

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
    // outline:
    OutStreamer.emitInt8(F.hasFnAttribute(YK_OUTLINE_FNATTR));
    // num_blocks:
    OutStreamer.emitSizeT(F.size());
    // blocks:
    unsigned BBIdx = 0;
    FuncLowerCtxt FLCtxt;
    for (BasicBlock &BB : F) {
      serialiseBlock(BB, FLCtxt, BBIdx);
    }
    FLCtxt.patchUpInstIdxs(OutStreamer);
  }

  void serialiseFunctionType(FunctionType *Ty) {
    serialiseTypeKind(TypeKindFunction);
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

  void serialiseStructType(StructType *STy) {
    serialiseTypeKind(TypeKindStruct);
    unsigned NumFields = STy->getNumElements();
    DataLayout DL(&M);
    const StructLayout *SL = DL.getStructLayout(STy);
    // num_fields:
    OutStreamer.emitSizeT(NumFields);
    // field_tys:
    for (unsigned I = 0; I < NumFields; I++) {
      OutStreamer.emitSizeT(typeIndex(STy->getElementType(I)));
    }
    // field_bit_offs:
    for (unsigned I = 0; I < NumFields; I++) {
      OutStreamer.emitSizeT(SL->getElementOffsetInBits(I));
    }
  }

  void serialiseType(llvm::Type *Ty) {
    if (Ty->isVoidTy()) {
      serialiseTypeKind(TypeKindVoid);
    } else if (PointerType *PT = dyn_cast<PointerType>(Ty)) {
      // FIXME: The Yk runtime assumes all pointers are void-ptr-sized.
      assert(DL.getPointerSize(PT->getAddressSpace()) == sizeof(void *));
      serialiseTypeKind(TypeKindPtr);
    } else if (IntegerType *ITy = dyn_cast<IntegerType>(Ty)) {
      serialiseTypeKind(TypeKindInteger);
      OutStreamer.emitInt32(ITy->getBitWidth());
    } else if (FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
      serialiseFunctionType(FTy);
    } else if (StructType *STy = dyn_cast<StructType>(Ty)) {
      serialiseStructType(STy);
    } else {
      serialiseTypeKind(TypeKindUnimplemented);
      serialiseString(toString(Ty));
    }
  }

  void serialiseConstantInt(ConstantInt *CI) {
    OutStreamer.emitSizeT(typeIndex(CI->getType()));
    unsigned BitWidth = CI->getBitWidth();

    // Figure out how many bytes it'd take to store that many bits.
    unsigned ByteWidth = BitWidth / 8;
    if (BitWidth % 8 != 0) {
      ByteWidth++;
    }
    OutStreamer.emitSizeT(ByteWidth);

    unsigned BitsRemain = BitWidth;
    while (BitsRemain > 0) {
      // We have to be careful not to ask extractBitsAsZExtValue() for more
      // bits than there are left, or an internal assertion will fail.
      unsigned NumBitsThisIter = std::min({BitsRemain, unsigned(8)});
      uint64_t Byte = CI->getValue().extractBitsAsZExtValue(
          NumBitsThisIter, BitWidth - BitsRemain);
      // This could be optimised by writing larger chunks thank a byte, but
      // beware endianess!
      OutStreamer.emitInt8(Byte);
      BitsRemain -= NumBitsThisIter;
    }
  }

  void serialiseUnimplementedConstant(Constant *C) {
    // type_index:
    OutStreamer.emitSizeT(typeIndex(C->getType()));
    // num_bytes:
    // Just report zero for now.
    OutStreamer.emitSizeT(0);
  }

  void serialiseConstant(Constant *C) {
    if (ConstantInt *CI = dyn_cast<ConstantInt>(C)) {
      serialiseConstantInt(CI);
    } else {
      serialiseUnimplementedConstant(C);
    }
  }

  void serialiseGlobal(GlobalVariable *G) {
    OutStreamer.emitInt8(G->isThreadLocal());
    serialiseString(G->getName());
  }

public:
  YkIRWriter(Module &M, MCStreamer &OutStreamer)
      : M(M), OutStreamer(OutStreamer), DL(&M) {}

  // Entry point for IR serialisation.
  //
  // The order of serialisation matters.
  //
  // - Serialising functions can introduce new types and constants.
  // - Serialising constants can introduce new types.
  //
  // So we must serialise functions, then constants, then types.
  void serialise() {
    // header:
    OutStreamer.emitInt32(Magic);
    OutStreamer.emitInt32(Version);

    // ptr_off_bitsize:
    unsigned IdxBitWidth = DL.getIndexSizeInBits(0);
    assert(IdxBitWidth <= 0xff);
    OutStreamer.emitInt8(IdxBitWidth);

    // num_funcs:
    OutStreamer.emitSizeT(M.size());
    // funcs:
    for (llvm::Function &F : M) {
      serialiseFunc(F);
    }

    // num_constants:
    OutStreamer.emitSizeT(Constants.size());
    // constants:
    for (Constant *&C : Constants) {
      serialiseConstant(C);
    }

    // num_globals:
    OutStreamer.emitSizeT(Globals.size());
    // globals:
    for (GlobalVariable *&G : Globals) {
      serialiseGlobal(G);
    }

    // Now that we've finished serialising globals, add a global (immutable, for
    // now) array to the LLVM module containing pointers to all the global
    // variables. We will use this to find the addresses of globals at runtime.
    // The indices of the array correspond with `GlobalDeclIdx`s in the AOT IR.
    vector<llvm::Constant *> GlobalsAsConsts;
    for (llvm::GlobalVariable *G : Globals) {
      GlobalsAsConsts.push_back(cast<llvm::Constant>(G));
    }
    ArrayType *GlobalsArrayTy =
        ArrayType::get(PointerType::get(M.getContext(), 0), Globals.size());
    GlobalVariable *GlobalsArray = new GlobalVariable(
        M, GlobalsArrayTy, true, GlobalValue::LinkageTypes::ExternalLinkage,
        ConstantArray::get(GlobalsArrayTy, GlobalsAsConsts));
    GlobalsArray->setName("__yk_globalvar_ptrs");

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
