//===- BitcodeReader.cpp - Internal BitcodeReader implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header defines the BitcodeReader class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "BitcodeReader.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/Operator.h"
#include "llvm/AutoUpgrade.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/OperandTraits.h"
using namespace llvm;

void BitcodeReader::FreeState() {
  if (BufferOwned)
    delete Buffer;
  Buffer = 0;
  std::vector<PATypeHolder>().swap(TypeList);
  ValueList.clear();
  MDValueList.clear();

  std::vector<AttrListPtr>().swap(MAttributes);
  std::vector<BasicBlock*>().swap(FunctionBBs);
  std::vector<Function*>().swap(FunctionsWithBodies);
  DeferredFunctionInfo.clear();
  MDKindMap.clear();
}

//===----------------------------------------------------------------------===//
//  Helper functions to implement forward reference resolution, etc.
//===----------------------------------------------------------------------===//

/// ConvertToString - Convert a string from a record into an std::string, return
/// true on failure.
template<typename StrTy>
static bool ConvertToString(SmallVector<uint64_t, 64> &Record, unsigned Idx,
                            StrTy &Result) {
  if (Idx > Record.size())
    return true;

  for (unsigned i = Idx, e = Record.size(); i != e; ++i)
    Result += (char)Record[i];
  return false;
}

static GlobalValue::LinkageTypes GetDecodedLinkage(unsigned Val) {
  switch (Val) {
  default: // Map unknown/new linkages to external
  case 0:  return GlobalValue::ExternalLinkage;
  case 1:  return GlobalValue::WeakAnyLinkage;
  case 2:  return GlobalValue::AppendingLinkage;
  case 3:  return GlobalValue::InternalLinkage;
  case 4:  return GlobalValue::LinkOnceAnyLinkage;
  case 5:  return GlobalValue::DLLImportLinkage;
  case 6:  return GlobalValue::DLLExportLinkage;
  case 7:  return GlobalValue::ExternalWeakLinkage;
  case 8:  return GlobalValue::CommonLinkage;
  case 9:  return GlobalValue::PrivateLinkage;
  case 10: return GlobalValue::WeakODRLinkage;
  case 11: return GlobalValue::LinkOnceODRLinkage;
  case 12: return GlobalValue::AvailableExternallyLinkage;
  case 13: return GlobalValue::LinkerPrivateLinkage;
  case 14: return GlobalValue::LinkerPrivateWeakLinkage;
  case 15: return GlobalValue::LinkerPrivateWeakDefAutoLinkage;
  }
}

static GlobalValue::VisibilityTypes GetDecodedVisibility(unsigned Val) {
  switch (Val) {
  default: // Map unknown visibilities to default.
  case 0: return GlobalValue::DefaultVisibility;
  case 1: return GlobalValue::HiddenVisibility;
  case 2: return GlobalValue::ProtectedVisibility;
  }
}

static int GetDecodedCastOpcode(unsigned Val) {
  switch (Val) {
  default: return -1;
  case bitc::CAST_TRUNC   : return Instruction::Trunc;
  case bitc::CAST_ZEXT    : return Instruction::ZExt;
  case bitc::CAST_SEXT    : return Instruction::SExt;
  case bitc::CAST_FPTOUI  : return Instruction::FPToUI;
  case bitc::CAST_FPTOSI  : return Instruction::FPToSI;
  case bitc::CAST_UITOFP  : return Instruction::UIToFP;
  case bitc::CAST_SITOFP  : return Instruction::SIToFP;
  case bitc::CAST_FPTRUNC : return Instruction::FPTrunc;
  case bitc::CAST_FPEXT   : return Instruction::FPExt;
  case bitc::CAST_PTRTOINT: return Instruction::PtrToInt;
  case bitc::CAST_INTTOPTR: return Instruction::IntToPtr;
  case bitc::CAST_BITCAST : return Instruction::BitCast;
  }
}
static int GetDecodedBinaryOpcode(unsigned Val, const Type *Ty) {
  switch (Val) {
  default: return -1;
  case bitc::BINOP_ADD:
    return Ty->isFPOrFPVectorTy() ? Instruction::FAdd : Instruction::Add;
  case bitc::BINOP_SUB:
    return Ty->isFPOrFPVectorTy() ? Instruction::FSub : Instruction::Sub;
  case bitc::BINOP_MUL:
    return Ty->isFPOrFPVectorTy() ? Instruction::FMul : Instruction::Mul;
  case bitc::BINOP_UDIV: return Instruction::UDiv;
  case bitc::BINOP_SDIV:
    return Ty->isFPOrFPVectorTy() ? Instruction::FDiv : Instruction::SDiv;
  case bitc::BINOP_UREM: return Instruction::URem;
  case bitc::BINOP_SREM:
    return Ty->isFPOrFPVectorTy() ? Instruction::FRem : Instruction::SRem;
  case bitc::BINOP_SHL:  return Instruction::Shl;
  case bitc::BINOP_LSHR: return Instruction::LShr;
  case bitc::BINOP_ASHR: return Instruction::AShr;
  case bitc::BINOP_AND:  return Instruction::And;
  case bitc::BINOP_OR:   return Instruction::Or;
  case bitc::BINOP_XOR:  return Instruction::Xor;
  }
}

namespace llvm {
namespace {
  /// @brief A class for maintaining the slot number definition
  /// as a placeholder for the actual definition for forward constants defs.
  class ConstantPlaceHolder : public ConstantExpr {
    ConstantPlaceHolder();                       // DO NOT IMPLEMENT
    void operator=(const ConstantPlaceHolder &); // DO NOT IMPLEMENT
  public:
    // allocate space for exactly one operand
    void *operator new(size_t s) {
      return User::operator new(s, 1);
    }
    explicit ConstantPlaceHolder(const Type *Ty, LLVMContext& Context)
      : ConstantExpr(Ty, Instruction::UserOp1, &Op<0>(), 1) {
      Op<0>() = UndefValue::get(Type::getInt32Ty(Context));
    }

    /// @brief Methods to support type inquiry through isa, cast, and dyn_cast.
    static inline bool classof(const ConstantPlaceHolder *) { return true; }
    static bool classof(const Value *V) {
      return isa<ConstantExpr>(V) &&
             cast<ConstantExpr>(V)->getOpcode() == Instruction::UserOp1;
    }


    /// Provide fast operand accessors
    //DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
  };
}

// FIXME: can we inherit this from ConstantExpr?
template <>
struct OperandTraits<ConstantPlaceHolder> : public FixedNumOperandTraits<1> {
};
}


void BitcodeReaderValueList::AssignValue(Value *V, unsigned Idx) {
  if (Idx == size()) {
    push_back(V);
    return;
  }

  if (Idx >= size())
    resize(Idx+1);

  WeakVH &OldV = ValuePtrs[Idx];
  if (OldV == 0) {
    OldV = V;
    return;
  }

  // Handle constants and non-constants (e.g. instrs) differently for
  // efficiency.
  if (Constant *PHC = dyn_cast<Constant>(&*OldV)) {
    ResolveConstants.push_back(std::make_pair(PHC, Idx));
    OldV = V;
  } else {
    // If there was a forward reference to this value, replace it.
    Value *PrevVal = OldV;
    OldV->replaceAllUsesWith(V);
    delete PrevVal;
  }
}


Constant *BitcodeReaderValueList::getConstantFwdRef(unsigned Idx,
                                                    const Type *Ty) {
  if (Idx >= size())
    resize(Idx + 1);

  if (Value *V = ValuePtrs[Idx]) {
    assert(Ty == V->getType() && "Type mismatch in constant table!");
    return cast<Constant>(V);
  }

  // Create and return a placeholder, which will later be RAUW'd.
  Constant *C = new ConstantPlaceHolder(Ty, Context);
  ValuePtrs[Idx] = C;
  return C;
}

Value *BitcodeReaderValueList::getValueFwdRef(unsigned Idx, const Type *Ty) {
  if (Idx >= size())
    resize(Idx + 1);

  if (Value *V = ValuePtrs[Idx]) {
    assert((Ty == 0 || Ty == V->getType()) && "Type mismatch in value table!");
    return V;
  }

  // No type specified, must be invalid reference.
  if (Ty == 0) return 0;

  // Create and return a placeholder, which will later be RAUW'd.
  Value *V = new Argument(Ty);
  ValuePtrs[Idx] = V;
  return V;
}

/// ResolveConstantForwardRefs - Once all constants are read, this method bulk
/// resolves any forward references.  The idea behind this is that we sometimes
/// get constants (such as large arrays) which reference *many* forward ref
/// constants.  Replacing each of these causes a lot of thrashing when
/// building/reuniquing the constant.  Instead of doing this, we look at all the
/// uses and rewrite all the place holders at once for any constant that uses
/// a placeholder.
void BitcodeReaderValueList::ResolveConstantForwardRefs() {
  // Sort the values by-pointer so that they are efficient to look up with a
  // binary search.
  std::sort(ResolveConstants.begin(), ResolveConstants.end());

  SmallVector<Constant*, 64> NewOps;

  while (!ResolveConstants.empty()) {
    Value *RealVal = operator[](ResolveConstants.back().second);
    Constant *Placeholder = ResolveConstants.back().first;
    ResolveConstants.pop_back();

    // Loop over all users of the placeholder, updating them to reference the
    // new value.  If they reference more than one placeholder, update them all
    // at once.
    while (!Placeholder->use_empty()) {
      Value::use_iterator UI = Placeholder->use_begin();
      User *U = *UI;

      // If the using object isn't uniqued, just update the operands.  This
      // handles instructions and initializers for global variables.
      if (!isa<Constant>(U) || isa<GlobalValue>(U)) {
        UI.getUse().set(RealVal);
        continue;
      }

      // Otherwise, we have a constant that uses the placeholder.  Replace that
      // constant with a new constant that has *all* placeholder uses updated.
      Constant *UserC = cast<Constant>(U);
      for (User::op_iterator I = UserC->op_begin(), E = UserC->op_end();
           I != E; ++I) {
        Value *NewOp;
        if (!isa<ConstantPlaceHolder>(*I)) {
          // Not a placeholder reference.
          NewOp = *I;
        } else if (*I == Placeholder) {
          // Common case is that it just references this one placeholder.
          NewOp = RealVal;
        } else {
          // Otherwise, look up the placeholder in ResolveConstants.
          ResolveConstantsTy::iterator It =
            std::lower_bound(ResolveConstants.begin(), ResolveConstants.end(),
                             std::pair<Constant*, unsigned>(cast<Constant>(*I),
                                                            0));
          assert(It != ResolveConstants.end() && It->first == *I);
          NewOp = operator[](It->second);
        }

        NewOps.push_back(cast<Constant>(NewOp));
      }

      // Make the new constant.
      Constant *NewC;
      if (ConstantArray *UserCA = dyn_cast<ConstantArray>(UserC)) {
        NewC = ConstantArray::get(UserCA->getType(), &NewOps[0],
                                        NewOps.size());
      } else if (ConstantStruct *UserCS = dyn_cast<ConstantStruct>(UserC)) {
        NewC = ConstantStruct::get(Context, &NewOps[0], NewOps.size(),
                                         UserCS->getType()->isPacked());
      } else if (isa<ConstantVector>(UserC)) {
        NewC = ConstantVector::get(&NewOps[0], NewOps.size());
      } else {
        assert(isa<ConstantExpr>(UserC) && "Must be a ConstantExpr.");
        NewC = cast<ConstantExpr>(UserC)->getWithOperands(&NewOps[0],
                                                          NewOps.size());
      }

      UserC->replaceAllUsesWith(NewC);
      UserC->destroyConstant();
      NewOps.clear();
    }

    // Update all ValueHandles, they should be the only users at this point.
    Placeholder->replaceAllUsesWith(RealVal);
    delete Placeholder;
  }
}

void BitcodeReaderMDValueList::AssignValue(Value *V, unsigned Idx) {
  if (Idx == size()) {
    push_back(V);
    return;
  }

  if (Idx >= size())
    resize(Idx+1);

  WeakVH &OldV = MDValuePtrs[Idx];
  if (OldV == 0) {
    OldV = V;
    return;
  }

  // If there was a forward reference to this value, replace it.
  MDNode *PrevVal = cast<MDNode>(OldV);
  OldV->replaceAllUsesWith(V);
  MDNode::deleteTemporary(PrevVal);
  // Deleting PrevVal sets Idx value in MDValuePtrs to null. Set new
  // value for Idx.
  MDValuePtrs[Idx] = V;
}

Value *BitcodeReaderMDValueList::getValueFwdRef(unsigned Idx) {
  if (Idx >= size())
    resize(Idx + 1);

  if (Value *V = MDValuePtrs[Idx]) {
    assert(V->getType()->isMetadataTy() && "Type mismatch in value table!");
    return V;
  }

  // Create and return a placeholder, which will later be RAUW'd.
  Value *V = MDNode::getTemporary(Context, 0, 0);
  MDValuePtrs[Idx] = V;
  return V;
}

const Type *BitcodeReader::getTypeByID(unsigned ID, bool isTypeTable) {
  // If the TypeID is in range, return it.
  if (ID < TypeList.size())
    return TypeList[ID].get();
  if (!isTypeTable) return 0;

  // The type table allows forward references.  Push as many Opaque types as
  // needed to get up to ID.
  while (TypeList.size() <= ID)
    TypeList.push_back(OpaqueType::get(Context));
  return TypeList.back().get();
}

//===----------------------------------------------------------------------===//
//  Functions for parsing blocks from the bitcode file
//===----------------------------------------------------------------------===//

bool BitcodeReader::ParseAttributeBlock() {
  if (Stream.EnterSubBlock(bitc::PARAMATTR_BLOCK_ID))
    return Error("Malformed block record");

  if (!MAttributes.empty())
    return Error("Multiple PARAMATTR blocks found!");

  SmallVector<uint64_t, 64> Record;

  SmallVector<AttributeWithIndex, 8> Attrs;

  // Read all the records.
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of PARAMATTR block");
      return false;
    }

    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: ignore.
      break;
    case bitc::PARAMATTR_CODE_ENTRY: { // ENTRY: [paramidx0, attr0, ...]
      if (Record.size() & 1)
        return Error("Invalid ENTRY record");

      // FIXME : Remove this autoupgrade code in LLVM 3.0.
      // If Function attributes are using index 0 then transfer them
      // to index ~0. Index 0 is used for return value attributes but used to be
      // used for function attributes.
      Attributes RetAttribute = Attribute::None;
      Attributes FnAttribute = Attribute::None;
      for (unsigned i = 0, e = Record.size(); i != e; i += 2) {
        // FIXME: remove in LLVM 3.0
        // The alignment is stored as a 16-bit raw value from bits 31--16.
        // We shift the bits above 31 down by 11 bits.

        unsigned Alignment = (Record[i+1] & (0xffffull << 16)) >> 16;
        if (Alignment && !isPowerOf2_32(Alignment))
          return Error("Alignment is not a power of two.");

        Attributes ReconstitutedAttr = Record[i+1] & 0xffff;
        if (Alignment)
          ReconstitutedAttr |= Attribute::constructAlignmentFromInt(Alignment);
        ReconstitutedAttr |= (Record[i+1] & (0xffffull << 32)) >> 11;
        Record[i+1] = ReconstitutedAttr;

        if (Record[i] == 0)
          RetAttribute = Record[i+1];
        else if (Record[i] == ~0U)
          FnAttribute = Record[i+1];
      }

      unsigned OldRetAttrs = (Attribute::NoUnwind|Attribute::NoReturn|
                              Attribute::ReadOnly|Attribute::ReadNone);

      if (FnAttribute == Attribute::None && RetAttribute != Attribute::None &&
          (RetAttribute & OldRetAttrs) != 0) {
        if (FnAttribute == Attribute::None) { // add a slot so they get added.
          Record.push_back(~0U);
          Record.push_back(0);
        }

        FnAttribute  |= RetAttribute & OldRetAttrs;
        RetAttribute &= ~OldRetAttrs;
      }

      for (unsigned i = 0, e = Record.size(); i != e; i += 2) {
        if (Record[i] == 0) {
          if (RetAttribute != Attribute::None)
            Attrs.push_back(AttributeWithIndex::get(0, RetAttribute));
        } else if (Record[i] == ~0U) {
          if (FnAttribute != Attribute::None)
            Attrs.push_back(AttributeWithIndex::get(~0U, FnAttribute));
        } else if (Record[i+1] != Attribute::None)
          Attrs.push_back(AttributeWithIndex::get(Record[i], Record[i+1]));
      }

      MAttributes.push_back(AttrListPtr::get(Attrs.begin(), Attrs.end()));
      Attrs.clear();
      break;
    }
    }
  }
}


bool BitcodeReader::ParseTypeTable() {
  if (Stream.EnterSubBlock(bitc::TYPE_BLOCK_ID))
    return Error("Malformed block record");

  if (!TypeList.empty())
    return Error("Multiple TYPE_BLOCKs found!");

  SmallVector<uint64_t, 64> Record;
  unsigned NumRecords = 0;

  // Read all the records for this type table.
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (NumRecords != TypeList.size())
        return Error("Invalid type forward reference in TYPE_BLOCK");
      if (Stream.ReadBlockEnd())
        return Error("Error at end of type table block");
      return false;
    }

    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    Record.clear();
    const Type *ResultTy = 0;
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown type.
      ResultTy = 0;
      break;
    case bitc::TYPE_CODE_NUMENTRY: // TYPE_CODE_NUMENTRY: [numentries]
      // TYPE_CODE_NUMENTRY contains a count of the number of types in the
      // type list.  This allows us to reserve space.
      if (Record.size() < 1)
        return Error("Invalid TYPE_CODE_NUMENTRY record");
      TypeList.reserve(Record[0]);
      continue;
    case bitc::TYPE_CODE_VOID:      // VOID
      ResultTy = Type::getVoidTy(Context);
      break;
    case bitc::TYPE_CODE_FLOAT:     // FLOAT
      ResultTy = Type::getFloatTy(Context);
      break;
    case bitc::TYPE_CODE_DOUBLE:    // DOUBLE
      ResultTy = Type::getDoubleTy(Context);
      break;
    case bitc::TYPE_CODE_X86_FP80:  // X86_FP80
      ResultTy = Type::getX86_FP80Ty(Context);
      break;
    case bitc::TYPE_CODE_FP128:     // FP128
      ResultTy = Type::getFP128Ty(Context);
      break;
    case bitc::TYPE_CODE_PPC_FP128: // PPC_FP128
      ResultTy = Type::getPPC_FP128Ty(Context);
      break;
    case bitc::TYPE_CODE_LABEL:     // LABEL
      ResultTy = Type::getLabelTy(Context);
      break;
    case bitc::TYPE_CODE_OPAQUE:    // OPAQUE
      ResultTy = 0;
      break;
    case bitc::TYPE_CODE_METADATA:  // METADATA
      ResultTy = Type::getMetadataTy(Context);
      break;
    case bitc::TYPE_CODE_INTEGER:   // INTEGER: [width]
      if (Record.size() < 1)
        return Error("Invalid Integer type record");

      ResultTy = IntegerType::get(Context, Record[0]);
      break;
    case bitc::TYPE_CODE_POINTER: { // POINTER: [pointee type] or
                                    //          [pointee type, address space]
      if (Record.size() < 1)
        return Error("Invalid POINTER type record");
      unsigned AddressSpace = 0;
      if (Record.size() == 2)
        AddressSpace = Record[1];
      ResultTy = PointerType::get(getTypeByID(Record[0], true),
                                        AddressSpace);
      break;
    }
    case bitc::TYPE_CODE_FUNCTION: {
      // FIXME: attrid is dead, remove it in LLVM 3.0
      // FUNCTION: [vararg, attrid, retty, paramty x N]
      if (Record.size() < 3)
        return Error("Invalid FUNCTION type record");
      std::vector<const Type*> ArgTys;
      for (unsigned i = 3, e = Record.size(); i != e; ++i)
        ArgTys.push_back(getTypeByID(Record[i], true));

      ResultTy = FunctionType::get(getTypeByID(Record[2], true), ArgTys,
                                   Record[0]);
      break;
    }
    case bitc::TYPE_CODE_STRUCT: {  // STRUCT: [ispacked, eltty x N]
      if (Record.size() < 1)
        return Error("Invalid STRUCT type record");
      std::vector<const Type*> EltTys;
      for (unsigned i = 1, e = Record.size(); i != e; ++i)
        EltTys.push_back(getTypeByID(Record[i], true));
      ResultTy = StructType::get(Context, EltTys, Record[0]);
      break;
    }
    case bitc::TYPE_CODE_ARRAY:     // ARRAY: [numelts, eltty]
      if (Record.size() < 2)
        return Error("Invalid ARRAY type record");
      ResultTy = ArrayType::get(getTypeByID(Record[1], true), Record[0]);
      break;
    case bitc::TYPE_CODE_VECTOR:    // VECTOR: [numelts, eltty]
      if (Record.size() < 2)
        return Error("Invalid VECTOR type record");
      ResultTy = VectorType::get(getTypeByID(Record[1], true), Record[0]);
      break;
    }

    if (NumRecords == TypeList.size()) {
      // If this is a new type slot, just append it.
      TypeList.push_back(ResultTy ? ResultTy : OpaqueType::get(Context));
      ++NumRecords;
    } else if (ResultTy == 0) {
      // Otherwise, this was forward referenced, so an opaque type was created,
      // but the result type is actually just an opaque.  Leave the one we
      // created previously.
      ++NumRecords;
    } else {
      // Otherwise, this was forward referenced, so an opaque type was created.
      // Resolve the opaque type to the real type now.
      assert(NumRecords < TypeList.size() && "Typelist imbalance");
      const OpaqueType *OldTy = cast<OpaqueType>(TypeList[NumRecords++].get());

      // Don't directly push the new type on the Tab. Instead we want to replace
      // the opaque type we previously inserted with the new concrete value. The
      // refinement from the abstract (opaque) type to the new type causes all
      // uses of the abstract type to use the concrete type (NewTy). This will
      // also cause the opaque type to be deleted.
      const_cast<OpaqueType*>(OldTy)->refineAbstractTypeTo(ResultTy);

      // This should have replaced the old opaque type with the new type in the
      // value table... or with a preexisting type that was already in the
      // system.  Let's just make sure it did.
      assert(TypeList[NumRecords-1].get() != OldTy &&
             "refineAbstractType didn't work!");
    }
  }
}


bool BitcodeReader::ParseTypeSymbolTable() {
  if (Stream.EnterSubBlock(bitc::TYPE_SYMTAB_BLOCK_ID))
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this type table.
  std::string TypeName;
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of type symbol table block");
      return false;
    }

    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown type.
      break;
    case bitc::TST_CODE_ENTRY:    // TST_ENTRY: [typeid, namechar x N]
      if (ConvertToString(Record, 1, TypeName))
        return Error("Invalid TST_ENTRY record");
      unsigned TypeID = Record[0];
      if (TypeID >= TypeList.size())
        return Error("Invalid Type ID in TST_ENTRY record");

      TheModule->addTypeName(TypeName, TypeList[TypeID].get());
      TypeName.clear();
      break;
    }
  }
}

bool BitcodeReader::ParseValueSymbolTable() {
  if (Stream.EnterSubBlock(bitc::VALUE_SYMTAB_BLOCK_ID))
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this value table.
  SmallString<128> ValueName;
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of value symbol table block");
      return false;
    }
    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: unknown type.
      break;
    case bitc::VST_CODE_ENTRY: {  // VST_ENTRY: [valueid, namechar x N]
      if (ConvertToString(Record, 1, ValueName))
        return Error("Invalid VST_ENTRY record");
      unsigned ValueID = Record[0];
      if (ValueID >= ValueList.size())
        return Error("Invalid Value ID in VST_ENTRY record");
      Value *V = ValueList[ValueID];

      V->setName(StringRef(ValueName.data(), ValueName.size()));
      ValueName.clear();
      break;
    }
    case bitc::VST_CODE_BBENTRY: {
      if (ConvertToString(Record, 1, ValueName))
        return Error("Invalid VST_BBENTRY record");
      BasicBlock *BB = getBasicBlock(Record[0]);
      if (BB == 0)
        return Error("Invalid BB ID in VST_BBENTRY record");

      BB->setName(StringRef(ValueName.data(), ValueName.size()));
      ValueName.clear();
      break;
    }
    }
  }
}

bool BitcodeReader::ParseMetadata() {
  unsigned NextMDValueNo = MDValueList.size();

  if (Stream.EnterSubBlock(bitc::METADATA_BLOCK_ID))
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records.
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of PARAMATTR block");
      return false;
    }

    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    bool IsFunctionLocal = false;
    // Read a record.
    Record.clear();
    Code = Stream.ReadRecord(Code, Record);
    switch (Code) {
    default:  // Default behavior: ignore.
      break;
    case bitc::METADATA_NAME: {
      // Read named of the named metadata.
      unsigned NameLength = Record.size();
      SmallString<8> Name;
      Name.resize(NameLength);
      for (unsigned i = 0; i != NameLength; ++i)
        Name[i] = Record[i];
      Record.clear();
      Code = Stream.ReadCode();

      // METADATA_NAME is always followed by METADATA_NAMED_NODE2.
      // Or METADATA_NAMED_NODE in LLVM 2.7. FIXME: Remove this in LLVM 3.0.
      unsigned NextBitCode = Stream.ReadRecord(Code, Record);
      if (NextBitCode == bitc::METADATA_NAMED_NODE) {
        LLVM2_7MetadataDetected = true;
      } else if (NextBitCode != bitc::METADATA_NAMED_NODE2)
        assert ( 0 && "Inavlid Named Metadata record");

      // Read named metadata elements.
      unsigned Size = Record.size();
      NamedMDNode *NMD = TheModule->getOrInsertNamedMetadata(Name);
      for (unsigned i = 0; i != Size; ++i) {
        MDNode *MD = dyn_cast<MDNode>(MDValueList.getValueFwdRef(Record[i]));
        if (MD == 0)
          return Error("Malformed metadata record");
        NMD->addOperand(MD);
      }
      // Backwards compatibility hack: NamedMDValues used to be Values,
      // and they got their own slots in the value numbering. They are no
      // longer Values, however we still need to account for them in the
      // numbering in order to be able to read old bitcode files.
      // FIXME: Remove this in LLVM 3.0.
      if (LLVM2_7MetadataDetected)
        MDValueList.AssignValue(0, NextMDValueNo++);
      break;
    }
    case bitc::METADATA_FN_NODE: // FIXME: Remove in LLVM 3.0.
    case bitc::METADATA_FN_NODE2:
      IsFunctionLocal = true;
      // fall-through
    case bitc::METADATA_NODE:    // FIXME: Remove in LLVM 3.0.
    case bitc::METADATA_NODE2: {

      // Detect 2.7-era metadata.
      // FIXME: Remove in LLVM 3.0.
      if (Code == bitc::METADATA_FN_NODE || Code == bitc::METADATA_NODE)
        LLVM2_7MetadataDetected = true;

      if (Record.size() % 2 == 1)
        return Error("Invalid METADATA_NODE2 record");

      unsigned Size = Record.size();
      SmallVector<Value*, 8> Elts;
      for (unsigned i = 0; i != Size; i += 2) {
        const Type *Ty = getTypeByID(Record[i], false);
        if (Ty->isMetadataTy())
          Elts.push_back(MDValueList.getValueFwdRef(Record[i+1]));
        else if (!Ty->isVoidTy())
          Elts.push_back(ValueList.getValueFwdRef(Record[i+1], Ty));
        else
          Elts.push_back(NULL);
      }
      Value *V = MDNode::getWhenValsUnresolved(Context,
                                               Elts.data(), Elts.size(),
                                               IsFunctionLocal);
      IsFunctionLocal = false;
      MDValueList.AssignValue(V, NextMDValueNo++);
      break;
    }
    case bitc::METADATA_STRING: {
      unsigned MDStringLength = Record.size();
      SmallString<8> String;
      String.resize(MDStringLength);
      for (unsigned i = 0; i != MDStringLength; ++i)
        String[i] = Record[i];
      Value *V = MDString::get(Context,
                               StringRef(String.data(), String.size()));
      MDValueList.AssignValue(V, NextMDValueNo++);
      break;
    }
    case bitc::METADATA_KIND: {
      unsigned RecordLength = Record.size();
      if (Record.empty() || RecordLength < 2)
        return Error("Invalid METADATA_KIND record");
      SmallString<8> Name;
      Name.resize(RecordLength-1);
      unsigned Kind = Record[0];
      for (unsigned i = 1; i != RecordLength; ++i)
        Name[i-1] = Record[i];
      
      unsigned NewKind = TheModule->getMDKindID(Name.str());
      if (!MDKindMap.insert(std::make_pair(Kind, NewKind)).second)
        return Error("Conflicting METADATA_KIND records");
      break;
    }
    }
  }
}

/// DecodeSignRotatedValue - Decode a signed value stored with the sign bit in
/// the LSB for dense VBR encoding.
static uint64_t DecodeSignRotatedValue(uint64_t V) {
  if ((V & 1) == 0)
    return V >> 1;
  if (V != 1)
    return -(V >> 1);
  // There is no such thing as -0 with integers.  "-0" really means MININT.
  return 1ULL << 63;
}

/// ResolveGlobalAndAliasInits - Resolve all of the initializers for global
/// values and aliases that we can.
bool BitcodeReader::ResolveGlobalAndAliasInits() {
  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInitWorklist;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInitWorklist;

  GlobalInitWorklist.swap(GlobalInits);
  AliasInitWorklist.swap(AliasInits);

  while (!GlobalInitWorklist.empty()) {
    unsigned ValID = GlobalInitWorklist.back().second;
    if (ValID >= ValueList.size()) {
      // Not ready to resolve this yet, it requires something later in the file.
      GlobalInits.push_back(GlobalInitWorklist.back());
    } else {
      if (Constant *C = dyn_cast<Constant>(ValueList[ValID]))
        GlobalInitWorklist.back().first->setInitializer(C);
      else
        return Error("Global variable initializer is not a constant!");
    }
    GlobalInitWorklist.pop_back();
  }

  while (!AliasInitWorklist.empty()) {
    unsigned ValID = AliasInitWorklist.back().second;
    if (ValID >= ValueList.size()) {
      AliasInits.push_back(AliasInitWorklist.back());
    } else {
      if (Constant *C = dyn_cast<Constant>(ValueList[ValID]))
        AliasInitWorklist.back().first->setAliasee(C);
      else
        return Error("Alias initializer is not a constant!");
    }
    AliasInitWorklist.pop_back();
  }
  return false;
}

bool BitcodeReader::ParseConstants() {
  if (Stream.EnterSubBlock(bitc::CONSTANTS_BLOCK_ID))
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this value table.
  const Type *CurTy = Type::getInt32Ty(Context);
  unsigned NextCstNo = ValueList.size();
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK)
      break;

    if (Code == bitc::ENTER_SUBBLOCK) {
      // No known subblocks, always skip them.
      Stream.ReadSubBlockID();
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    Record.clear();
    Value *V = 0;
    unsigned BitCode = Stream.ReadRecord(Code, Record);
    switch (BitCode) {
    default:  // Default behavior: unknown constant
    case bitc::CST_CODE_UNDEF:     // UNDEF
      V = UndefValue::get(CurTy);
      break;
    case bitc::CST_CODE_SETTYPE:   // SETTYPE: [typeid]
      if (Record.empty())
        return Error("Malformed CST_SETTYPE record");
      if (Record[0] >= TypeList.size())
        return Error("Invalid Type ID in CST_SETTYPE record");
      CurTy = TypeList[Record[0]];
      continue;  // Skip the ValueList manipulation.
    case bitc::CST_CODE_NULL:      // NULL
      V = Constant::getNullValue(CurTy);
      break;
    case bitc::CST_CODE_INTEGER:   // INTEGER: [intval]
      if (!CurTy->isIntegerTy() || Record.empty())
        return Error("Invalid CST_INTEGER record");
      V = ConstantInt::get(CurTy, DecodeSignRotatedValue(Record[0]));
      break;
    case bitc::CST_CODE_WIDE_INTEGER: {// WIDE_INTEGER: [n x intval]
      if (!CurTy->isIntegerTy() || Record.empty())
        return Error("Invalid WIDE_INTEGER record");

      unsigned NumWords = Record.size();
      SmallVector<uint64_t, 8> Words;
      Words.resize(NumWords);
      for (unsigned i = 0; i != NumWords; ++i)
        Words[i] = DecodeSignRotatedValue(Record[i]);
      V = ConstantInt::get(Context,
                           APInt(cast<IntegerType>(CurTy)->getBitWidth(),
                           NumWords, &Words[0]));
      break;
    }
    case bitc::CST_CODE_FLOAT: {    // FLOAT: [fpval]
      if (Record.empty())
        return Error("Invalid FLOAT record");
      if (CurTy->isFloatTy())
        V = ConstantFP::get(Context, APFloat(APInt(32, (uint32_t)Record[0])));
      else if (CurTy->isDoubleTy())
        V = ConstantFP::get(Context, APFloat(APInt(64, Record[0])));
      else if (CurTy->isX86_FP80Ty()) {
        // Bits are not stored the same way as a normal i80 APInt, compensate.
        uint64_t Rearrange[2];
        Rearrange[0] = (Record[1] & 0xffffLL) | (Record[0] << 16);
        Rearrange[1] = Record[0] >> 48;
        V = ConstantFP::get(Context, APFloat(APInt(80, 2, Rearrange)));
      } else if (CurTy->isFP128Ty())
        V = ConstantFP::get(Context, APFloat(APInt(128, 2, &Record[0]), true));
      else if (CurTy->isPPC_FP128Ty())
        V = ConstantFP::get(Context, APFloat(APInt(128, 2, &Record[0])));
      else
        V = UndefValue::get(CurTy);
      break;
    }

    case bitc::CST_CODE_AGGREGATE: {// AGGREGATE: [n x value number]
      if (Record.empty())
        return Error("Invalid CST_AGGREGATE record");

      unsigned Size = Record.size();
      std::vector<Constant*> Elts;

      if (const StructType *STy = dyn_cast<StructType>(CurTy)) {
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i],
                                                     STy->getElementType(i)));
        V = ConstantStruct::get(STy, Elts);
      } else if (const ArrayType *ATy = dyn_cast<ArrayType>(CurTy)) {
        const Type *EltTy = ATy->getElementType();
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i], EltTy));
        V = ConstantArray::get(ATy, Elts);
      } else if (const VectorType *VTy = dyn_cast<VectorType>(CurTy)) {
        const Type *EltTy = VTy->getElementType();
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i], EltTy));
        V = ConstantVector::get(Elts);
      } else {
        V = UndefValue::get(CurTy);
      }
      break;
    }
    case bitc::CST_CODE_STRING: { // STRING: [values]
      if (Record.empty())
        return Error("Invalid CST_AGGREGATE record");

      const ArrayType *ATy = cast<ArrayType>(CurTy);
      const Type *EltTy = ATy->getElementType();

      unsigned Size = Record.size();
      std::vector<Constant*> Elts;
      for (unsigned i = 0; i != Size; ++i)
        Elts.push_back(ConstantInt::get(EltTy, Record[i]));
      V = ConstantArray::get(ATy, Elts);
      break;
    }
    case bitc::CST_CODE_CSTRING: { // CSTRING: [values]
      if (Record.empty())
        return Error("Invalid CST_AGGREGATE record");

      const ArrayType *ATy = cast<ArrayType>(CurTy);
      const Type *EltTy = ATy->getElementType();

      unsigned Size = Record.size();
      std::vector<Constant*> Elts;
      for (unsigned i = 0; i != Size; ++i)
        Elts.push_back(ConstantInt::get(EltTy, Record[i]));
      Elts.push_back(Constant::getNullValue(EltTy));
      V = ConstantArray::get(ATy, Elts);
      break;
    }
    case bitc::CST_CODE_CE_BINOP: {  // CE_BINOP: [opcode, opval, opval]
      if (Record.size() < 3) return Error("Invalid CE_BINOP record");
      int Opc = GetDecodedBinaryOpcode(Record[0], CurTy);
      if (Opc < 0) {
        V = UndefValue::get(CurTy);  // Unknown binop.
      } else {
        Constant *LHS = ValueList.getConstantFwdRef(Record[1], CurTy);
        Constant *RHS = ValueList.getConstantFwdRef(Record[2], CurTy);
        unsigned Flags = 0;
        if (Record.size() >= 4) {
          if (Opc == Instruction::Add ||
              Opc == Instruction::Sub ||
              Opc == Instruction::Mul) {
            if (Record[3] & (1 << bitc::OBO_NO_SIGNED_WRAP))
              Flags |= OverflowingBinaryOperator::NoSignedWrap;
            if (Record[3] & (1 << bitc::OBO_NO_UNSIGNED_WRAP))
              Flags |= OverflowingBinaryOperator::NoUnsignedWrap;
          } else if (Opc == Instruction::SDiv) {
            if (Record[3] & (1 << bitc::SDIV_EXACT))
              Flags |= SDivOperator::IsExact;
          }
        }
        V = ConstantExpr::get(Opc, LHS, RHS, Flags);
      }
      break;
    }
    case bitc::CST_CODE_CE_CAST: {  // CE_CAST: [opcode, opty, opval]
      if (Record.size() < 3) return Error("Invalid CE_CAST record");
      int Opc = GetDecodedCastOpcode(Record[0]);
      if (Opc < 0) {
        V = UndefValue::get(CurTy);  // Unknown cast.
      } else {
        const Type *OpTy = getTypeByID(Record[1]);
        if (!OpTy) return Error("Invalid CE_CAST record");
        Constant *Op = ValueList.getConstantFwdRef(Record[2], OpTy);
        V = ConstantExpr::getCast(Opc, Op, CurTy);
      }
      break;
    }
    case bitc::CST_CODE_CE_INBOUNDS_GEP:
    case bitc::CST_CODE_CE_GEP: {  // CE_GEP:        [n x operands]
      if (Record.size() & 1) return Error("Invalid CE_GEP record");
      SmallVector<Constant*, 16> Elts;
      for (unsigned i = 0, e = Record.size(); i != e; i += 2) {
        const Type *ElTy = getTypeByID(Record[i]);
        if (!ElTy) return Error("Invalid CE_GEP record");
        Elts.push_back(ValueList.getConstantFwdRef(Record[i+1], ElTy));
      }
      if (BitCode == bitc::CST_CODE_CE_INBOUNDS_GEP)
        V = ConstantExpr::getInBoundsGetElementPtr(Elts[0], &Elts[1],
                                                   Elts.size()-1);
      else
        V = ConstantExpr::getGetElementPtr(Elts[0], &Elts[1],
                                           Elts.size()-1);
      break;
    }
    case bitc::CST_CODE_CE_SELECT:  // CE_SELECT: [opval#, opval#, opval#]
      if (Record.size() < 3) return Error("Invalid CE_SELECT record");
      V = ConstantExpr::getSelect(ValueList.getConstantFwdRef(Record[0],
                                                              Type::getInt1Ty(Context)),
                                  ValueList.getConstantFwdRef(Record[1],CurTy),
                                  ValueList.getConstantFwdRef(Record[2],CurTy));
      break;
    case bitc::CST_CODE_CE_EXTRACTELT: { // CE_EXTRACTELT: [opty, opval, opval]
      if (Record.size() < 3) return Error("Invalid CE_EXTRACTELT record");
      const VectorType *OpTy =
        dyn_cast_or_null<VectorType>(getTypeByID(Record[0]));
      if (OpTy == 0) return Error("Invalid CE_EXTRACTELT record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2], Type::getInt32Ty(Context));
      V = ConstantExpr::getExtractElement(Op0, Op1);
      break;
    }
    case bitc::CST_CODE_CE_INSERTELT: { // CE_INSERTELT: [opval, opval, opval]
      const VectorType *OpTy = dyn_cast<VectorType>(CurTy);
      if (Record.size() < 3 || OpTy == 0)
        return Error("Invalid CE_INSERTELT record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[0], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[1],
                                                  OpTy->getElementType());
      Constant *Op2 = ValueList.getConstantFwdRef(Record[2], Type::getInt32Ty(Context));
      V = ConstantExpr::getInsertElement(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_SHUFFLEVEC: { // CE_SHUFFLEVEC: [opval, opval, opval]
      const VectorType *OpTy = dyn_cast<VectorType>(CurTy);
      if (Record.size() < 3 || OpTy == 0)
        return Error("Invalid CE_SHUFFLEVEC record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[0], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[1], OpTy);
      const Type *ShufTy = VectorType::get(Type::getInt32Ty(Context),
                                                 OpTy->getNumElements());
      Constant *Op2 = ValueList.getConstantFwdRef(Record[2], ShufTy);
      V = ConstantExpr::getShuffleVector(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_SHUFVEC_EX: { // [opty, opval, opval, opval]
      const VectorType *RTy = dyn_cast<VectorType>(CurTy);
      const VectorType *OpTy = dyn_cast<VectorType>(getTypeByID(Record[0]));
      if (Record.size() < 4 || RTy == 0 || OpTy == 0)
        return Error("Invalid CE_SHUFVEC_EX record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2], OpTy);
      const Type *ShufTy = VectorType::get(Type::getInt32Ty(Context),
                                                 RTy->getNumElements());
      Constant *Op2 = ValueList.getConstantFwdRef(Record[3], ShufTy);
      V = ConstantExpr::getShuffleVector(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_CMP: {     // CE_CMP: [opty, opval, opval, pred]
      if (Record.size() < 4) return Error("Invalid CE_CMP record");
      const Type *OpTy = getTypeByID(Record[0]);
      if (OpTy == 0) return Error("Invalid CE_CMP record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2], OpTy);

      if (OpTy->isFPOrFPVectorTy())
        V = ConstantExpr::getFCmp(Record[3], Op0, Op1);
      else
        V = ConstantExpr::getICmp(Record[3], Op0, Op1);
      break;
    }
    case bitc::CST_CODE_INLINEASM: {
      if (Record.size() < 2) return Error("Invalid INLINEASM record");
      std::string AsmStr, ConstrStr;
      bool HasSideEffects = Record[0] & 1;
      bool IsAlignStack = Record[0] >> 1;
      unsigned AsmStrSize = Record[1];
      if (2+AsmStrSize >= Record.size())
        return Error("Invalid INLINEASM record");
      unsigned ConstStrSize = Record[2+AsmStrSize];
      if (3+AsmStrSize+ConstStrSize > Record.size())
        return Error("Invalid INLINEASM record");

      for (unsigned i = 0; i != AsmStrSize; ++i)
        AsmStr += (char)Record[2+i];
      for (unsigned i = 0; i != ConstStrSize; ++i)
        ConstrStr += (char)Record[3+AsmStrSize+i];
      const PointerType *PTy = cast<PointerType>(CurTy);
      V = InlineAsm::get(cast<FunctionType>(PTy->getElementType()),
                         AsmStr, ConstrStr, HasSideEffects, IsAlignStack);
      break;
    }
    case bitc::CST_CODE_BLOCKADDRESS:{
      if (Record.size() < 3) return Error("Invalid CE_BLOCKADDRESS record");
      const Type *FnTy = getTypeByID(Record[0]);
      if (FnTy == 0) return Error("Invalid CE_BLOCKADDRESS record");
      Function *Fn =
        dyn_cast_or_null<Function>(ValueList.getConstantFwdRef(Record[1],FnTy));
      if (Fn == 0) return Error("Invalid CE_BLOCKADDRESS record");
      
      GlobalVariable *FwdRef = new GlobalVariable(*Fn->getParent(),
                                                  Type::getInt8Ty(Context),
                                            false, GlobalValue::InternalLinkage,
                                                  0, "");
      BlockAddrFwdRefs[Fn].push_back(std::make_pair(Record[2], FwdRef));
      V = FwdRef;
      break;
    }  
    }

    ValueList.AssignValue(V, NextCstNo);
    ++NextCstNo;
  }

  if (NextCstNo != ValueList.size())
    return Error("Invalid constant reference!");

  if (Stream.ReadBlockEnd())
    return Error("Error at end of constants block");

  // Once all the constants have been read, go through and resolve forward
  // references.
  ValueList.ResolveConstantForwardRefs();
  return false;
}

/// RememberAndSkipFunctionBody - When we see the block for a function body,
/// remember where it is and then skip it.  This lets us lazily deserialize the
/// functions.
bool BitcodeReader::RememberAndSkipFunctionBody() {
  // Get the function we are talking about.
  if (FunctionsWithBodies.empty())
    return Error("Insufficient function protos");

  Function *Fn = FunctionsWithBodies.back();
  FunctionsWithBodies.pop_back();

  // Save the current stream state.
  uint64_t CurBit = Stream.GetCurrentBitNo();
  DeferredFunctionInfo[Fn] = CurBit;

  // Skip over the function block for now.
  if (Stream.SkipBlock())
    return Error("Malformed block record");
  return false;
}

bool BitcodeReader::ParseModule() {
  if (Stream.EnterSubBlock(bitc::MODULE_BLOCK_ID))
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;
  std::vector<std::string> SectionTable;
  std::vector<std::string> GCTable;

  // Read all the records for this module.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of module block");

      // Patch the initializers for globals and aliases up.
      ResolveGlobalAndAliasInits();
      if (!GlobalInits.empty() || !AliasInits.empty())
        return Error("Malformed global initializer set");
      if (!FunctionsWithBodies.empty())
        return Error("Too few function bodies found");

      // Look for intrinsic functions which need to be upgraded at some point
      for (Module::iterator FI = TheModule->begin(), FE = TheModule->end();
           FI != FE; ++FI) {
        Function* NewFn;
        if (UpgradeIntrinsicFunction(FI, NewFn))
          UpgradedIntrinsics.push_back(std::make_pair(FI, NewFn));
      }

      // Look for global variables which need to be renamed.
      for (Module::global_iterator
             GI = TheModule->global_begin(), GE = TheModule->global_end();
           GI != GE; ++GI)
        UpgradeGlobalVariable(GI);

      // Force deallocation of memory for these vectors to favor the client that
      // want lazy deserialization.
      std::vector<std::pair<GlobalVariable*, unsigned> >().swap(GlobalInits);
      std::vector<std::pair<GlobalAlias*, unsigned> >().swap(AliasInits);
      std::vector<Function*>().swap(FunctionsWithBodies);
      return false;
    }

    if (Code == bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      default:  // Skip unknown content.
        if (Stream.SkipBlock())
          return Error("Malformed block record");
        break;
      case bitc::BLOCKINFO_BLOCK_ID:
        if (Stream.ReadBlockInfoBlock())
          return Error("Malformed BlockInfoBlock");
        break;
      case bitc::PARAMATTR_BLOCK_ID:
        if (ParseAttributeBlock())
          return true;
        break;
      case bitc::TYPE_BLOCK_ID:
        if (ParseTypeTable())
          return true;
        break;
      case bitc::TYPE_SYMTAB_BLOCK_ID:
        if (ParseTypeSymbolTable())
          return true;
        break;
      case bitc::VALUE_SYMTAB_BLOCK_ID:
        if (ParseValueSymbolTable())
          return true;
        break;
      case bitc::CONSTANTS_BLOCK_ID:
        if (ParseConstants() || ResolveGlobalAndAliasInits())
          return true;
        break;
      case bitc::METADATA_BLOCK_ID:
        if (ParseMetadata())
          return true;
        break;
      case bitc::FUNCTION_BLOCK_ID:
        // If this is the first function body we've seen, reverse the
        // FunctionsWithBodies list.
        if (!HasReversedFunctionsWithBodies) {
          std::reverse(FunctionsWithBodies.begin(), FunctionsWithBodies.end());
          HasReversedFunctionsWithBodies = true;
        }

        if (RememberAndSkipFunctionBody())
          return true;
        break;
      }
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    switch (Stream.ReadRecord(Code, Record)) {
    default: break;  // Default behavior, ignore unknown content.
    case bitc::MODULE_CODE_VERSION:  // VERSION: [version#]
      if (Record.size() < 1)
        return Error("Malformed MODULE_CODE_VERSION");
      // Only version #0 is supported so far.
      if (Record[0] != 0)
        return Error("Unknown bitstream version!");
      break;
    case bitc::MODULE_CODE_TRIPLE: {  // TRIPLE: [strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_TRIPLE record");
      TheModule->setTargetTriple(S);
      break;
    }
    case bitc::MODULE_CODE_DATALAYOUT: {  // DATALAYOUT: [strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_DATALAYOUT record");
      TheModule->setDataLayout(S);
      break;
    }
    case bitc::MODULE_CODE_ASM: {  // ASM: [strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_ASM record");
      TheModule->setModuleInlineAsm(S);
      break;
    }
    case bitc::MODULE_CODE_DEPLIB: {  // DEPLIB: [strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_DEPLIB record");
      TheModule->addLibrary(S);
      break;
    }
    case bitc::MODULE_CODE_SECTIONNAME: {  // SECTIONNAME: [strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_SECTIONNAME record");
      SectionTable.push_back(S);
      break;
    }
    case bitc::MODULE_CODE_GCNAME: {  // SECTIONNAME: [strchr x N]
      std::string S;
      if (ConvertToString(Record, 0, S))
        return Error("Invalid MODULE_CODE_GCNAME record");
      GCTable.push_back(S);
      break;
    }
    // GLOBALVAR: [pointer type, isconst, initid,
    //             linkage, alignment, section, visibility, threadlocal]
    case bitc::MODULE_CODE_GLOBALVAR: {
      if (Record.size() < 6)
        return Error("Invalid MODULE_CODE_GLOBALVAR record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!Ty->isPointerTy())
        return Error("Global not a pointer type!");
      unsigned AddressSpace = cast<PointerType>(Ty)->getAddressSpace();
      Ty = cast<PointerType>(Ty)->getElementType();

      bool isConstant = Record[1];
      GlobalValue::LinkageTypes Linkage = GetDecodedLinkage(Record[3]);
      unsigned Alignment = (1 << Record[4]) >> 1;
      std::string Section;
      if (Record[5]) {
        if (Record[5]-1 >= SectionTable.size())
          return Error("Invalid section ID");
        Section = SectionTable[Record[5]-1];
      }
      GlobalValue::VisibilityTypes Visibility = GlobalValue::DefaultVisibility;
      if (Record.size() > 6)
        Visibility = GetDecodedVisibility(Record[6]);
      bool isThreadLocal = false;
      if (Record.size() > 7)
        isThreadLocal = Record[7];

      GlobalVariable *NewGV =
        new GlobalVariable(*TheModule, Ty, isConstant, Linkage, 0, "", 0,
                           isThreadLocal, AddressSpace);
      NewGV->setAlignment(Alignment);
      if (!Section.empty())
        NewGV->setSection(Section);
      NewGV->setVisibility(Visibility);
      NewGV->setThreadLocal(isThreadLocal);

      ValueList.push_back(NewGV);

      // Remember which value to use for the global initializer.
      if (unsigned InitID = Record[2])
        GlobalInits.push_back(std::make_pair(NewGV, InitID-1));
      break;
    }
    // FUNCTION:  [type, callingconv, isproto, linkage, paramattr,
    //             alignment, section, visibility, gc]
    case bitc::MODULE_CODE_FUNCTION: {
      if (Record.size() < 8)
        return Error("Invalid MODULE_CODE_FUNCTION record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!Ty->isPointerTy())
        return Error("Function not a pointer type!");
      const FunctionType *FTy =
        dyn_cast<FunctionType>(cast<PointerType>(Ty)->getElementType());
      if (!FTy)
        return Error("Function not a pointer to function type!");

      Function *Func = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                        "", TheModule);

      Func->setCallingConv(static_cast<CallingConv::ID>(Record[1]));
      bool isProto = Record[2];
      Func->setLinkage(GetDecodedLinkage(Record[3]));
      Func->setAttributes(getAttributes(Record[4]));

      Func->setAlignment((1 << Record[5]) >> 1);
      if (Record[6]) {
        if (Record[6]-1 >= SectionTable.size())
          return Error("Invalid section ID");
        Func->setSection(SectionTable[Record[6]-1]);
      }
      Func->setVisibility(GetDecodedVisibility(Record[7]));
      if (Record.size() > 8 && Record[8]) {
        if (Record[8]-1 > GCTable.size())
          return Error("Invalid GC ID");
        Func->setGC(GCTable[Record[8]-1].c_str());
      }
      ValueList.push_back(Func);

      // If this is a function with a body, remember the prototype we are
      // creating now, so that we can match up the body with them later.
      if (!isProto)
        FunctionsWithBodies.push_back(Func);
      break;
    }
    // ALIAS: [alias type, aliasee val#, linkage]
    // ALIAS: [alias type, aliasee val#, linkage, visibility]
    case bitc::MODULE_CODE_ALIAS: {
      if (Record.size() < 3)
        return Error("Invalid MODULE_ALIAS record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!Ty->isPointerTy())
        return Error("Function not a pointer type!");

      GlobalAlias *NewGA = new GlobalAlias(Ty, GetDecodedLinkage(Record[2]),
                                           "", 0, TheModule);
      // Old bitcode files didn't have visibility field.
      if (Record.size() > 3)
        NewGA->setVisibility(GetDecodedVisibility(Record[3]));
      ValueList.push_back(NewGA);
      AliasInits.push_back(std::make_pair(NewGA, Record[1]));
      break;
    }
    /// MODULE_CODE_PURGEVALS: [numvals]
    case bitc::MODULE_CODE_PURGEVALS:
      // Trim down the value list to the specified size.
      if (Record.size() < 1 || Record[0] > ValueList.size())
        return Error("Invalid MODULE_PURGEVALS record");
      ValueList.shrinkTo(Record[0]);
      break;
    }
    Record.clear();
  }

  return Error("Premature end of bitstream");
}

bool BitcodeReader::ParseBitcodeInto(Module *M) {
  TheModule = 0;

  unsigned char *BufPtr = (unsigned char *)Buffer->getBufferStart();
  unsigned char *BufEnd = BufPtr+Buffer->getBufferSize();

  if (Buffer->getBufferSize() & 3) {
    if (!isRawBitcode(BufPtr, BufEnd) && !isBitcodeWrapper(BufPtr, BufEnd))
      return Error("Invalid bitcode signature");
    else
      return Error("Bitcode stream should be a multiple of 4 bytes in length");
  }

  // If we have a wrapper header, parse it and ignore the non-bc file contents.
  // The magic number is 0x0B17C0DE stored in little endian.
  if (isBitcodeWrapper(BufPtr, BufEnd))
    if (SkipBitcodeWrapperHeader(BufPtr, BufEnd))
      return Error("Invalid bitcode wrapper header");

  StreamFile.init(BufPtr, BufEnd);
  Stream.init(StreamFile);

  // Sniff for the signature.
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0x0 ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0xD)
    return Error("Invalid bitcode signature");

  // We expect a number of well-defined blocks, though we don't necessarily
  // need to understand them all.
  while (!Stream.AtEndOfStream()) {
    unsigned Code = Stream.ReadCode();

    if (Code != bitc::ENTER_SUBBLOCK)
      return Error("Invalid record at top-level");

    unsigned BlockID = Stream.ReadSubBlockID();

    // We only know the MODULE subblock ID.
    switch (BlockID) {
    case bitc::BLOCKINFO_BLOCK_ID:
      if (Stream.ReadBlockInfoBlock())
        return Error("Malformed BlockInfoBlock");
      break;
    case bitc::MODULE_BLOCK_ID:
      // Reject multiple MODULE_BLOCK's in a single bitstream.
      if (TheModule)
        return Error("Multiple MODULE_BLOCKs in same stream");
      TheModule = M;
      if (ParseModule())
        return true;
      break;
    default:
      if (Stream.SkipBlock())
        return Error("Malformed block record");
      break;
    }
  }

  return false;
}

/// ParseMetadataAttachment - Parse metadata attachments.
bool BitcodeReader::ParseMetadataAttachment() {
  if (Stream.EnterSubBlock(bitc::METADATA_ATTACHMENT_ID))
    return Error("Malformed block record");

  SmallVector<uint64_t, 64> Record;
  while(1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of PARAMATTR block");
      break;
    }
    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }
    // Read a metadata attachment record.
    Record.clear();
    switch (Stream.ReadRecord(Code, Record)) {
    default:  // Default behavior: ignore.
      break;
    // FIXME: Remove in LLVM 3.0.
    case bitc::METADATA_ATTACHMENT:
      LLVM2_7MetadataDetected = true;
    case bitc::METADATA_ATTACHMENT2: {
      unsigned RecordLength = Record.size();
      if (Record.empty() || (RecordLength - 1) % 2 == 1)
        return Error ("Invalid METADATA_ATTACHMENT reader!");
      Instruction *Inst = InstructionList[Record[0]];
      for (unsigned i = 1; i != RecordLength; i = i+2) {
        unsigned Kind = Record[i];
        DenseMap<unsigned, unsigned>::iterator I =
          MDKindMap.find(Kind);
        if (I == MDKindMap.end())
          return Error("Invalid metadata kind ID");
        Value *Node = MDValueList.getValueFwdRef(Record[i+1]);
        Inst->setMetadata(I->second, cast<MDNode>(Node));
      }
      break;
    }
    }
  }
  return false;
}

/// ParseFunctionBody - Lazily parse the specified function body block.
bool BitcodeReader::ParseFunctionBody(Function *F) {
  if (Stream.EnterSubBlock(bitc::FUNCTION_BLOCK_ID))
    return Error("Malformed block record");

  InstructionList.clear();
  unsigned ModuleValueListSize = ValueList.size();
  unsigned ModuleMDValueListSize = MDValueList.size();

  // Add all the function arguments to the value table.
  for(Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    ValueList.push_back(I);

  unsigned NextValueNo = ValueList.size();
  BasicBlock *CurBB = 0;
  unsigned CurBBNo = 0;

  DebugLoc LastLoc;
  
  // Read all the records.
  SmallVector<uint64_t, 64> Record;
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (Stream.ReadBlockEnd())
        return Error("Error at end of function block");
      break;
    }

    if (Code == bitc::ENTER_SUBBLOCK) {
      switch (Stream.ReadSubBlockID()) {
      default:  // Skip unknown content.
        if (Stream.SkipBlock())
          return Error("Malformed block record");
        break;
      case bitc::CONSTANTS_BLOCK_ID:
        if (ParseConstants()) return true;
        NextValueNo = ValueList.size();
        break;
      case bitc::VALUE_SYMTAB_BLOCK_ID:
        if (ParseValueSymbolTable()) return true;
        break;
      case bitc::METADATA_ATTACHMENT_ID:
        if (ParseMetadataAttachment()) return true;
        break;
      case bitc::METADATA_BLOCK_ID:
        if (ParseMetadata()) return true;
        break;
      }
      continue;
    }

    if (Code == bitc::DEFINE_ABBREV) {
      Stream.ReadAbbrevRecord();
      continue;
    }

    // Read a record.
    Record.clear();
    Instruction *I = 0;
    unsigned BitCode = Stream.ReadRecord(Code, Record);
    switch (BitCode) {
    default: // Default behavior: reject
      return Error("Unknown instruction");
    case bitc::FUNC_CODE_DECLAREBLOCKS:     // DECLAREBLOCKS: [nblocks]
      if (Record.size() < 1 || Record[0] == 0)
        return Error("Invalid DECLAREBLOCKS record");
      // Create all the basic blocks for the function.
      FunctionBBs.resize(Record[0]);
      for (unsigned i = 0, e = FunctionBBs.size(); i != e; ++i)
        FunctionBBs[i] = BasicBlock::Create(Context, "", F);
      CurBB = FunctionBBs[0];
      continue;

        
    case bitc::FUNC_CODE_DEBUG_LOC_AGAIN:  // DEBUG_LOC_AGAIN
      // This record indicates that the last instruction is at the same
      // location as the previous instruction with a location.
      I = 0;
        
      // Get the last instruction emitted.
      if (CurBB && !CurBB->empty())
        I = &CurBB->back();
      else if (CurBBNo && FunctionBBs[CurBBNo-1] &&
               !FunctionBBs[CurBBNo-1]->empty())
        I = &FunctionBBs[CurBBNo-1]->back();
        
      if (I == 0) return Error("Invalid DEBUG_LOC_AGAIN record");
      I->setDebugLoc(LastLoc);
      I = 0;
      continue;
        
    // FIXME: Remove this in LLVM 3.0.
    case bitc::FUNC_CODE_DEBUG_LOC:
      LLVM2_7MetadataDetected = true;
    case bitc::FUNC_CODE_DEBUG_LOC2: {      // DEBUG_LOC: [line, col, scope, ia]
      I = 0;     // Get the last instruction emitted.
      if (CurBB && !CurBB->empty())
        I = &CurBB->back();
      else if (CurBBNo && FunctionBBs[CurBBNo-1] &&
               !FunctionBBs[CurBBNo-1]->empty())
        I = &FunctionBBs[CurBBNo-1]->back();
      if (I == 0 || Record.size() < 4)
        return Error("Invalid FUNC_CODE_DEBUG_LOC record");
      
      unsigned Line = Record[0], Col = Record[1];
      unsigned ScopeID = Record[2], IAID = Record[3];
      
      MDNode *Scope = 0, *IA = 0;
      if (ScopeID) Scope = cast<MDNode>(MDValueList.getValueFwdRef(ScopeID-1));
      if (IAID)    IA = cast<MDNode>(MDValueList.getValueFwdRef(IAID-1));
      LastLoc = DebugLoc::get(Line, Col, Scope, IA);
      I->setDebugLoc(LastLoc);
      I = 0;
      continue;
    }

    case bitc::FUNC_CODE_INST_BINOP: {    // BINOP: [opval, ty, opval, opcode]
      unsigned OpNum = 0;
      Value *LHS, *RHS;
      if (getValueTypePair(Record, OpNum, NextValueNo, LHS) ||
          getValue(Record, OpNum, LHS->getType(), RHS) ||
          OpNum+1 > Record.size())
        return Error("Invalid BINOP record");

      int Opc = GetDecodedBinaryOpcode(Record[OpNum++], LHS->getType());
      if (Opc == -1) return Error("Invalid BINOP record");
      I = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
      InstructionList.push_back(I);
      if (OpNum < Record.size()) {
        if (Opc == Instruction::Add ||
            Opc == Instruction::Sub ||
            Opc == Instruction::Mul) {
          if (Record[OpNum] & (1 << bitc::OBO_NO_SIGNED_WRAP))
            cast<BinaryOperator>(I)->setHasNoSignedWrap(true);
          if (Record[OpNum] & (1 << bitc::OBO_NO_UNSIGNED_WRAP))
            cast<BinaryOperator>(I)->setHasNoUnsignedWrap(true);
        } else if (Opc == Instruction::SDiv) {
          if (Record[OpNum] & (1 << bitc::SDIV_EXACT))
            cast<BinaryOperator>(I)->setIsExact(true);
        }
      }
      break;
    }
    case bitc::FUNC_CODE_INST_CAST: {    // CAST: [opval, opty, destty, castopc]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          OpNum+2 != Record.size())
        return Error("Invalid CAST record");

      const Type *ResTy = getTypeByID(Record[OpNum]);
      int Opc = GetDecodedCastOpcode(Record[OpNum+1]);
      if (Opc == -1 || ResTy == 0)
        return Error("Invalid CAST record");
      I = CastInst::Create((Instruction::CastOps)Opc, Op, ResTy);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_INBOUNDS_GEP:
    case bitc::FUNC_CODE_INST_GEP: { // GEP: [n x operands]
      unsigned OpNum = 0;
      Value *BasePtr;
      if (getValueTypePair(Record, OpNum, NextValueNo, BasePtr))
        return Error("Invalid GEP record");

      SmallVector<Value*, 16> GEPIdx;
      while (OpNum != Record.size()) {
        Value *Op;
        if (getValueTypePair(Record, OpNum, NextValueNo, Op))
          return Error("Invalid GEP record");
        GEPIdx.push_back(Op);
      }

      I = GetElementPtrInst::Create(BasePtr, GEPIdx.begin(), GEPIdx.end());
      InstructionList.push_back(I);
      if (BitCode == bitc::FUNC_CODE_INST_INBOUNDS_GEP)
        cast<GetElementPtrInst>(I)->setIsInBounds(true);
      break;
    }

    case bitc::FUNC_CODE_INST_EXTRACTVAL: {
                                       // EXTRACTVAL: [opty, opval, n x indices]
      unsigned OpNum = 0;
      Value *Agg;
      if (getValueTypePair(Record, OpNum, NextValueNo, Agg))
        return Error("Invalid EXTRACTVAL record");

      SmallVector<unsigned, 4> EXTRACTVALIdx;
      for (unsigned RecSize = Record.size();
           OpNum != RecSize; ++OpNum) {
        uint64_t Index = Record[OpNum];
        if ((unsigned)Index != Index)
          return Error("Invalid EXTRACTVAL index");
        EXTRACTVALIdx.push_back((unsigned)Index);
      }

      I = ExtractValueInst::Create(Agg,
                                   EXTRACTVALIdx.begin(), EXTRACTVALIdx.end());
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_INSERTVAL: {
                           // INSERTVAL: [opty, opval, opty, opval, n x indices]
      unsigned OpNum = 0;
      Value *Agg;
      if (getValueTypePair(Record, OpNum, NextValueNo, Agg))
        return Error("Invalid INSERTVAL record");
      Value *Val;
      if (getValueTypePair(Record, OpNum, NextValueNo, Val))
        return Error("Invalid INSERTVAL record");

      SmallVector<unsigned, 4> INSERTVALIdx;
      for (unsigned RecSize = Record.size();
           OpNum != RecSize; ++OpNum) {
        uint64_t Index = Record[OpNum];
        if ((unsigned)Index != Index)
          return Error("Invalid INSERTVAL index");
        INSERTVALIdx.push_back((unsigned)Index);
      }

      I = InsertValueInst::Create(Agg, Val,
                                  INSERTVALIdx.begin(), INSERTVALIdx.end());
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_SELECT: { // SELECT: [opval, ty, opval, opval]
      // obsolete form of select
      // handles select i1 ... in old bitcode
      unsigned OpNum = 0;
      Value *TrueVal, *FalseVal, *Cond;
      if (getValueTypePair(Record, OpNum, NextValueNo, TrueVal) ||
          getValue(Record, OpNum, TrueVal->getType(), FalseVal) ||
          getValue(Record, OpNum, Type::getInt1Ty(Context), Cond))
        return Error("Invalid SELECT record");

      I = SelectInst::Create(Cond, TrueVal, FalseVal);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_VSELECT: {// VSELECT: [ty,opval,opval,predty,pred]
      // new form of select
      // handles select i1 or select [N x i1]
      unsigned OpNum = 0;
      Value *TrueVal, *FalseVal, *Cond;
      if (getValueTypePair(Record, OpNum, NextValueNo, TrueVal) ||
          getValue(Record, OpNum, TrueVal->getType(), FalseVal) ||
          getValueTypePair(Record, OpNum, NextValueNo, Cond))
        return Error("Invalid SELECT record");

      // select condition can be either i1 or [N x i1]
      if (const VectorType* vector_type =
          dyn_cast<const VectorType>(Cond->getType())) {
        // expect <n x i1>
        if (vector_type->getElementType() != Type::getInt1Ty(Context))
          return Error("Invalid SELECT condition type");
      } else {
        // expect i1
        if (Cond->getType() != Type::getInt1Ty(Context))
          return Error("Invalid SELECT condition type");
      }

      I = SelectInst::Create(Cond, TrueVal, FalseVal);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_EXTRACTELT: { // EXTRACTELT: [opty, opval, opval]
      unsigned OpNum = 0;
      Value *Vec, *Idx;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec) ||
          getValue(Record, OpNum, Type::getInt32Ty(Context), Idx))
        return Error("Invalid EXTRACTELT record");
      I = ExtractElementInst::Create(Vec, Idx);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_INSERTELT: { // INSERTELT: [ty, opval,opval,opval]
      unsigned OpNum = 0;
      Value *Vec, *Elt, *Idx;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec) ||
          getValue(Record, OpNum,
                   cast<VectorType>(Vec->getType())->getElementType(), Elt) ||
          getValue(Record, OpNum, Type::getInt32Ty(Context), Idx))
        return Error("Invalid INSERTELT record");
      I = InsertElementInst::Create(Vec, Elt, Idx);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_SHUFFLEVEC: {// SHUFFLEVEC: [opval,ty,opval,opval]
      unsigned OpNum = 0;
      Value *Vec1, *Vec2, *Mask;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec1) ||
          getValue(Record, OpNum, Vec1->getType(), Vec2))
        return Error("Invalid SHUFFLEVEC record");

      if (getValueTypePair(Record, OpNum, NextValueNo, Mask))
        return Error("Invalid SHUFFLEVEC record");
      I = new ShuffleVectorInst(Vec1, Vec2, Mask);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_CMP:   // CMP: [opty, opval, opval, pred]
      // Old form of ICmp/FCmp returning bool
      // Existed to differentiate between icmp/fcmp and vicmp/vfcmp which were
      // both legal on vectors but had different behaviour.
    case bitc::FUNC_CODE_INST_CMP2: { // CMP2: [opty, opval, opval, pred]
      // FCmp/ICmp returning bool or vector of bool

      unsigned OpNum = 0;
      Value *LHS, *RHS;
      if (getValueTypePair(Record, OpNum, NextValueNo, LHS) ||
          getValue(Record, OpNum, LHS->getType(), RHS) ||
          OpNum+1 != Record.size())
        return Error("Invalid CMP record");

      if (LHS->getType()->isFPOrFPVectorTy())
        I = new FCmpInst((FCmpInst::Predicate)Record[OpNum], LHS, RHS);
      else
        I = new ICmpInst((ICmpInst::Predicate)Record[OpNum], LHS, RHS);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_GETRESULT: { // GETRESULT: [ty, val, n]
      if (Record.size() != 2)
        return Error("Invalid GETRESULT record");
      unsigned OpNum = 0;
      Value *Op;
      getValueTypePair(Record, OpNum, NextValueNo, Op);
      unsigned Index = Record[1];
      I = ExtractValueInst::Create(Op, Index);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_RET: // RET: [opty,opval<optional>]
      {
        unsigned Size = Record.size();
        if (Size == 0) {
          I = ReturnInst::Create(Context);
          InstructionList.push_back(I);
          break;
        }

        unsigned OpNum = 0;
        SmallVector<Value *,4> Vs;
        do {
          Value *Op = NULL;
          if (getValueTypePair(Record, OpNum, NextValueNo, Op))
            return Error("Invalid RET record");
          Vs.push_back(Op);
        } while(OpNum != Record.size());

        const Type *ReturnType = F->getReturnType();
        // Handle multiple return values. FIXME: Remove in LLVM 3.0.
        if (Vs.size() > 1 ||
            (ReturnType->isStructTy() &&
             (Vs.empty() || Vs[0]->getType() != ReturnType))) {
          Value *RV = UndefValue::get(ReturnType);
          for (unsigned i = 0, e = Vs.size(); i != e; ++i) {
            I = InsertValueInst::Create(RV, Vs[i], i, "mrv");
            InstructionList.push_back(I);
            CurBB->getInstList().push_back(I);
            ValueList.AssignValue(I, NextValueNo++);
            RV = I;
          }
          I = ReturnInst::Create(Context, RV);
          InstructionList.push_back(I);
          break;
        }

        I = ReturnInst::Create(Context, Vs[0]);
        InstructionList.push_back(I);
        break;
      }
    case bitc::FUNC_CODE_INST_BR: { // BR: [bb#, bb#, opval] or [bb#]
      if (Record.size() != 1 && Record.size() != 3)
        return Error("Invalid BR record");
      BasicBlock *TrueDest = getBasicBlock(Record[0]);
      if (TrueDest == 0)
        return Error("Invalid BR record");

      if (Record.size() == 1) {
        I = BranchInst::Create(TrueDest);
        InstructionList.push_back(I);
      }
      else {
        BasicBlock *FalseDest = getBasicBlock(Record[1]);
        Value *Cond = getFnValueByID(Record[2], Type::getInt1Ty(Context));
        if (FalseDest == 0 || Cond == 0)
          return Error("Invalid BR record");
        I = BranchInst::Create(TrueDest, FalseDest, Cond);
        InstructionList.push_back(I);
      }
      break;
    }
    case bitc::FUNC_CODE_INST_SWITCH: { // SWITCH: [opty, op0, op1, ...]
      if (Record.size() < 3 || (Record.size() & 1) == 0)
        return Error("Invalid SWITCH record");
      const Type *OpTy = getTypeByID(Record[0]);
      Value *Cond = getFnValueByID(Record[1], OpTy);
      BasicBlock *Default = getBasicBlock(Record[2]);
      if (OpTy == 0 || Cond == 0 || Default == 0)
        return Error("Invalid SWITCH record");
      unsigned NumCases = (Record.size()-3)/2;
      SwitchInst *SI = SwitchInst::Create(Cond, Default, NumCases);
      InstructionList.push_back(SI);
      for (unsigned i = 0, e = NumCases; i != e; ++i) {
        ConstantInt *CaseVal =
          dyn_cast_or_null<ConstantInt>(getFnValueByID(Record[3+i*2], OpTy));
        BasicBlock *DestBB = getBasicBlock(Record[1+3+i*2]);
        if (CaseVal == 0 || DestBB == 0) {
          delete SI;
          return Error("Invalid SWITCH record!");
        }
        SI->addCase(CaseVal, DestBB);
      }
      I = SI;
      break;
    }
    case bitc::FUNC_CODE_INST_INDIRECTBR: { // INDIRECTBR: [opty, op0, op1, ...]
      if (Record.size() < 2)
        return Error("Invalid INDIRECTBR record");
      const Type *OpTy = getTypeByID(Record[0]);
      Value *Address = getFnValueByID(Record[1], OpTy);
      if (OpTy == 0 || Address == 0)
        return Error("Invalid INDIRECTBR record");
      unsigned NumDests = Record.size()-2;
      IndirectBrInst *IBI = IndirectBrInst::Create(Address, NumDests);
      InstructionList.push_back(IBI);
      for (unsigned i = 0, e = NumDests; i != e; ++i) {
        if (BasicBlock *DestBB = getBasicBlock(Record[2+i])) {
          IBI->addDestination(DestBB);
        } else {
          delete IBI;
          return Error("Invalid INDIRECTBR record!");
        }
      }
      I = IBI;
      break;
    }
        
    case bitc::FUNC_CODE_INST_INVOKE: {
      // INVOKE: [attrs, cc, normBB, unwindBB, fnty, op0,op1,op2, ...]
      if (Record.size() < 4) return Error("Invalid INVOKE record");
      AttrListPtr PAL = getAttributes(Record[0]);
      unsigned CCInfo = Record[1];
      BasicBlock *NormalBB = getBasicBlock(Record[2]);
      BasicBlock *UnwindBB = getBasicBlock(Record[3]);

      unsigned OpNum = 4;
      Value *Callee;
      if (getValueTypePair(Record, OpNum, NextValueNo, Callee))
        return Error("Invalid INVOKE record");

      const PointerType *CalleeTy = dyn_cast<PointerType>(Callee->getType());
      const FunctionType *FTy = !CalleeTy ? 0 :
        dyn_cast<FunctionType>(CalleeTy->getElementType());

      // Check that the right number of fixed parameters are here.
      if (FTy == 0 || NormalBB == 0 || UnwindBB == 0 ||
          Record.size() < OpNum+FTy->getNumParams())
        return Error("Invalid INVOKE record");

      SmallVector<Value*, 16> Ops;
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i, ++OpNum) {
        Ops.push_back(getFnValueByID(Record[OpNum], FTy->getParamType(i)));
        if (Ops.back() == 0) return Error("Invalid INVOKE record");
      }

      if (!FTy->isVarArg()) {
        if (Record.size() != OpNum)
          return Error("Invalid INVOKE record");
      } else {
        // Read type/value pairs for varargs params.
        while (OpNum != Record.size()) {
          Value *Op;
          if (getValueTypePair(Record, OpNum, NextValueNo, Op))
            return Error("Invalid INVOKE record");
          Ops.push_back(Op);
        }
      }

      I = InvokeInst::Create(Callee, NormalBB, UnwindBB,
                             Ops.begin(), Ops.end());
      InstructionList.push_back(I);
      cast<InvokeInst>(I)->setCallingConv(
        static_cast<CallingConv::ID>(CCInfo));
      cast<InvokeInst>(I)->setAttributes(PAL);
      break;
    }
    case bitc::FUNC_CODE_INST_UNWIND: // UNWIND
      I = new UnwindInst(Context);
      InstructionList.push_back(I);
      break;
    case bitc::FUNC_CODE_INST_UNREACHABLE: // UNREACHABLE
      I = new UnreachableInst(Context);
      InstructionList.push_back(I);
      break;
    case bitc::FUNC_CODE_INST_PHI: { // PHI: [ty, val0,bb0, ...]
      if (Record.size() < 1 || ((Record.size()-1)&1))
        return Error("Invalid PHI record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!Ty) return Error("Invalid PHI record");

      PHINode *PN = PHINode::Create(Ty);
      InstructionList.push_back(PN);
      PN->reserveOperandSpace((Record.size()-1)/2);

      for (unsigned i = 0, e = Record.size()-1; i != e; i += 2) {
        Value *V = getFnValueByID(Record[1+i], Ty);
        BasicBlock *BB = getBasicBlock(Record[2+i]);
        if (!V || !BB) return Error("Invalid PHI record");
        PN->addIncoming(V, BB);
      }
      I = PN;
      break;
    }

    case bitc::FUNC_CODE_INST_MALLOC: { // MALLOC: [instty, op, align]
      // Autoupgrade malloc instruction to malloc call.
      // FIXME: Remove in LLVM 3.0.
      if (Record.size() < 3)
        return Error("Invalid MALLOC record");
      const PointerType *Ty =
        dyn_cast_or_null<PointerType>(getTypeByID(Record[0]));
      Value *Size = getFnValueByID(Record[1], Type::getInt32Ty(Context));
      if (!Ty || !Size) return Error("Invalid MALLOC record");
      if (!CurBB) return Error("Invalid malloc instruction with no BB");
      const Type *Int32Ty = IntegerType::getInt32Ty(CurBB->getContext());
      Constant *AllocSize = ConstantExpr::getSizeOf(Ty->getElementType());
      AllocSize = ConstantExpr::getTruncOrBitCast(AllocSize, Int32Ty);
      I = CallInst::CreateMalloc(CurBB, Int32Ty, Ty->getElementType(),
                                 AllocSize, Size, NULL);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_FREE: { // FREE: [op, opty]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          OpNum != Record.size())
        return Error("Invalid FREE record");
      if (!CurBB) return Error("Invalid free instruction with no BB");
      I = CallInst::CreateFree(Op, CurBB);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_ALLOCA: { // ALLOCA: [instty, opty, op, align]
      // For backward compatibility, tolerate a lack of an opty, and use i32.
      // Remove this in LLVM 3.0.
      if (Record.size() < 3 || Record.size() > 4)
        return Error("Invalid ALLOCA record");
      unsigned OpNum = 0;
      const PointerType *Ty =
        dyn_cast_or_null<PointerType>(getTypeByID(Record[OpNum++]));
      const Type *OpTy = Record.size() == 4 ? getTypeByID(Record[OpNum++]) :
                                              Type::getInt32Ty(Context);
      Value *Size = getFnValueByID(Record[OpNum++], OpTy);
      unsigned Align = Record[OpNum++];
      if (!Ty || !Size) return Error("Invalid ALLOCA record");
      I = new AllocaInst(Ty->getElementType(), Size, (1 << Align) >> 1);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_LOAD: { // LOAD: [opty, op, align, vol]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          OpNum+2 != Record.size())
        return Error("Invalid LOAD record");

      I = new LoadInst(Op, "", Record[OpNum+1], (1 << Record[OpNum]) >> 1);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_STORE2: { // STORE2:[ptrty, ptr, val, align, vol]
      unsigned OpNum = 0;
      Value *Val, *Ptr;
      if (getValueTypePair(Record, OpNum, NextValueNo, Ptr) ||
          getValue(Record, OpNum,
                    cast<PointerType>(Ptr->getType())->getElementType(), Val) ||
          OpNum+2 != Record.size())
        return Error("Invalid STORE record");

      I = new StoreInst(Val, Ptr, Record[OpNum+1], (1 << Record[OpNum]) >> 1);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_STORE: { // STORE:[val, valty, ptr, align, vol]
      // FIXME: Legacy form of store instruction. Should be removed in LLVM 3.0.
      unsigned OpNum = 0;
      Value *Val, *Ptr;
      if (getValueTypePair(Record, OpNum, NextValueNo, Val) ||
          getValue(Record, OpNum,
                   PointerType::getUnqual(Val->getType()), Ptr)||
          OpNum+2 != Record.size())
        return Error("Invalid STORE record");

      I = new StoreInst(Val, Ptr, Record[OpNum+1], (1 << Record[OpNum]) >> 1);
      InstructionList.push_back(I);
      break;
    }
    // FIXME: Remove this in LLVM 3.0.
    case bitc::FUNC_CODE_INST_CALL:
      LLVM2_7MetadataDetected = true;
    case bitc::FUNC_CODE_INST_CALL2: {
      // CALL: [paramattrs, cc, fnty, fnid, arg0, arg1...]
      if (Record.size() < 3)
        return Error("Invalid CALL record");

      AttrListPtr PAL = getAttributes(Record[0]);
      unsigned CCInfo = Record[1];

      unsigned OpNum = 2;
      Value *Callee;
      if (getValueTypePair(Record, OpNum, NextValueNo, Callee))
        return Error("Invalid CALL record");

      const PointerType *OpTy = dyn_cast<PointerType>(Callee->getType());
      const FunctionType *FTy = 0;
      if (OpTy) FTy = dyn_cast<FunctionType>(OpTy->getElementType());
      if (!FTy || Record.size() < FTy->getNumParams()+OpNum)
        return Error("Invalid CALL record");

      SmallVector<Value*, 16> Args;
      // Read the fixed params.
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i, ++OpNum) {
        if (FTy->getParamType(i)->getTypeID()==Type::LabelTyID)
          Args.push_back(getBasicBlock(Record[OpNum]));
        else
          Args.push_back(getFnValueByID(Record[OpNum], FTy->getParamType(i)));
        if (Args.back() == 0) return Error("Invalid CALL record");
      }

      // Read type/value pairs for varargs params.
      if (!FTy->isVarArg()) {
        if (OpNum != Record.size())
          return Error("Invalid CALL record");
      } else {
        while (OpNum != Record.size()) {
          Value *Op;
          if (getValueTypePair(Record, OpNum, NextValueNo, Op))
            return Error("Invalid CALL record");
          Args.push_back(Op);
        }
      }

      I = CallInst::Create(Callee, Args.begin(), Args.end());
      InstructionList.push_back(I);
      cast<CallInst>(I)->setCallingConv(
        static_cast<CallingConv::ID>(CCInfo>>1));
      cast<CallInst>(I)->setTailCall(CCInfo & 1);
      cast<CallInst>(I)->setAttributes(PAL);
      break;
    }
    case bitc::FUNC_CODE_INST_VAARG: { // VAARG: [valistty, valist, instty]
      if (Record.size() < 3)
        return Error("Invalid VAARG record");
      const Type *OpTy = getTypeByID(Record[0]);
      Value *Op = getFnValueByID(Record[1], OpTy);
      const Type *ResTy = getTypeByID(Record[2]);
      if (!OpTy || !Op || !ResTy)
        return Error("Invalid VAARG record");
      I = new VAArgInst(Op, ResTy);
      InstructionList.push_back(I);
      break;
    }
    }

    // Add instruction to end of current BB.  If there is no current BB, reject
    // this file.
    if (CurBB == 0) {
      delete I;
      return Error("Invalid instruction with no BB");
    }
    CurBB->getInstList().push_back(I);

    // If this was a terminator instruction, move to the next block.
    if (isa<TerminatorInst>(I)) {
      ++CurBBNo;
      CurBB = CurBBNo < FunctionBBs.size() ? FunctionBBs[CurBBNo] : 0;
    }

    // Non-void values get registered in the value table for future use.
    if (I && !I->getType()->isVoidTy())
      ValueList.AssignValue(I, NextValueNo++);
  }

  // Check the function list for unresolved values.
  if (Argument *A = dyn_cast<Argument>(ValueList.back())) {
    if (A->getParent() == 0) {
      // We found at least one unresolved value.  Nuke them all to avoid leaks.
      for (unsigned i = ModuleValueListSize, e = ValueList.size(); i != e; ++i){
        if ((A = dyn_cast<Argument>(ValueList[i])) && A->getParent() == 0) {
          A->replaceAllUsesWith(UndefValue::get(A->getType()));
          delete A;
        }
      }
      return Error("Never resolved value found in function!");
    }
  }

  // FIXME: Check for unresolved forward-declared metadata references
  // and clean up leaks.

  // See if anything took the address of blocks in this function.  If so,
  // resolve them now.
  DenseMap<Function*, std::vector<BlockAddrRefTy> >::iterator BAFRI =
    BlockAddrFwdRefs.find(F);
  if (BAFRI != BlockAddrFwdRefs.end()) {
    std::vector<BlockAddrRefTy> &RefList = BAFRI->second;
    for (unsigned i = 0, e = RefList.size(); i != e; ++i) {
      unsigned BlockIdx = RefList[i].first;
      if (BlockIdx >= FunctionBBs.size())
        return Error("Invalid blockaddress block #");
    
      GlobalVariable *FwdRef = RefList[i].second;
      FwdRef->replaceAllUsesWith(BlockAddress::get(F, FunctionBBs[BlockIdx]));
      FwdRef->eraseFromParent();
    }
    
    BlockAddrFwdRefs.erase(BAFRI);
  }
  
  // FIXME: Remove this in LLVM 3.0.
  unsigned NewMDValueListSize = MDValueList.size();

  // Trim the value list down to the size it was before we parsed this function.
  ValueList.shrinkTo(ModuleValueListSize);
  MDValueList.shrinkTo(ModuleMDValueListSize);

  // Backwards compatibility hack: Function-local metadata numbers
  // were previously not reset between functions. This is now fixed,
  // however we still need to understand the old numbering in order
  // to be able to read old bitcode files.
  // FIXME: Remove this in LLVM 3.0.
  if (LLVM2_7MetadataDetected)
    MDValueList.resize(NewMDValueListSize);

  std::vector<BasicBlock*>().swap(FunctionBBs);

  return false;
}

//===----------------------------------------------------------------------===//
// GVMaterializer implementation
//===----------------------------------------------------------------------===//


bool BitcodeReader::isMaterializable(const GlobalValue *GV) const {
  if (const Function *F = dyn_cast<Function>(GV)) {
    return F->isDeclaration() &&
      DeferredFunctionInfo.count(const_cast<Function*>(F));
  }
  return false;
}

bool BitcodeReader::Materialize(GlobalValue *GV, std::string *ErrInfo) {
  Function *F = dyn_cast<Function>(GV);
  // If it's not a function or is already material, ignore the request.
  if (!F || !F->isMaterializable()) return false;

  DenseMap<Function*, uint64_t>::iterator DFII = DeferredFunctionInfo.find(F);
  assert(DFII != DeferredFunctionInfo.end() && "Deferred function not found!");

  // Move the bit stream to the saved position of the deferred function body.
  Stream.JumpToBit(DFII->second);

  if (ParseFunctionBody(F)) {
    if (ErrInfo) *ErrInfo = ErrorString;
    return true;
  }

  // Upgrade any old intrinsic calls in the function.
  for (UpgradedIntrinsicMap::iterator I = UpgradedIntrinsics.begin(),
       E = UpgradedIntrinsics.end(); I != E; ++I) {
    if (I->first != I->second) {
      for (Value::use_iterator UI = I->first->use_begin(),
           UE = I->first->use_end(); UI != UE; ) {
        if (CallInst* CI = dyn_cast<CallInst>(*UI++))
          UpgradeIntrinsicCall(CI, I->second);
      }
    }
  }

  return false;
}

bool BitcodeReader::isDematerializable(const GlobalValue *GV) const {
  const Function *F = dyn_cast<Function>(GV);
  if (!F || F->isDeclaration())
    return false;
  return DeferredFunctionInfo.count(const_cast<Function*>(F));
}

void BitcodeReader::Dematerialize(GlobalValue *GV) {
  Function *F = dyn_cast<Function>(GV);
  // If this function isn't dematerializable, this is a noop.
  if (!F || !isDematerializable(F))
    return;

  assert(DeferredFunctionInfo.count(F) && "No info to read function later?");

  // Just forget the function body, we can remat it later.
  F->deleteBody();
}


bool BitcodeReader::MaterializeModule(Module *M, std::string *ErrInfo) {
  assert(M == TheModule &&
         "Can only Materialize the Module this BitcodeReader is attached to.");
  // Iterate over the module, deserializing any functions that are still on
  // disk.
  for (Module::iterator F = TheModule->begin(), E = TheModule->end();
       F != E; ++F)
    if (F->isMaterializable() &&
        Materialize(F, ErrInfo))
      return true;

  // Upgrade any intrinsic calls that slipped through (should not happen!) and
  // delete the old functions to clean up. We can't do this unless the entire
  // module is materialized because there could always be another function body
  // with calls to the old function.
  for (std::vector<std::pair<Function*, Function*> >::iterator I =
       UpgradedIntrinsics.begin(), E = UpgradedIntrinsics.end(); I != E; ++I) {
    if (I->first != I->second) {
      for (Value::use_iterator UI = I->first->use_begin(),
           UE = I->first->use_end(); UI != UE; ) {
        if (CallInst* CI = dyn_cast<CallInst>(*UI++))
          UpgradeIntrinsicCall(CI, I->second);
      }
      if (!I->first->use_empty())
        I->first->replaceAllUsesWith(I->second);
      I->first->eraseFromParent();
    }
  }
  std::vector<std::pair<Function*, Function*> >().swap(UpgradedIntrinsics);

  // Check debug info intrinsics.
  CheckDebugInfoIntrinsics(TheModule);

  return false;
}


//===----------------------------------------------------------------------===//
// External interface
//===----------------------------------------------------------------------===//

/// getLazyBitcodeModule - lazy function-at-a-time loading from a file.
///
Module *llvm::getLazyBitcodeModule(MemoryBuffer *Buffer,
                                   LLVMContext& Context,
                                   std::string *ErrMsg) {
  Module *M = new Module(Buffer->getBufferIdentifier(), Context);
  BitcodeReader *R = new BitcodeReader(Buffer, Context);
  M->setMaterializer(R);
  if (R->ParseBitcodeInto(M)) {
    if (ErrMsg)
      *ErrMsg = R->getErrorString();

    delete M;  // Also deletes R.
    return 0;
  }
  // Have the BitcodeReader dtor delete 'Buffer'.
  R->setBufferOwned(true);
  return M;
}

/// ParseBitcodeFile - Read the specified bitcode file, returning the module.
/// If an error occurs, return null and fill in *ErrMsg if non-null.
Module *llvm::ParseBitcodeFile(MemoryBuffer *Buffer, LLVMContext& Context,
                               std::string *ErrMsg){
  Module *M = getLazyBitcodeModule(Buffer, Context, ErrMsg);
  if (!M) return 0;

  // Don't let the BitcodeReader dtor delete 'Buffer', regardless of whether
  // there was an error.
  static_cast<BitcodeReader*>(M->getMaterializer())->setBufferOwned(false);

  // Read in the entire module, and destroy the BitcodeReader.
  if (M->MaterializeAllPermanently(ErrMsg)) {
    delete M;
    return NULL;
  }
  return M;
}
