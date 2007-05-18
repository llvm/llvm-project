//===- BitcodeReader.cpp - Internal BitcodeReader implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License.  See LICENSE.TXT for details.
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
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
using namespace llvm;

void BitcodeReader::FreeState() {
  delete Buffer;
  Buffer = 0;
  std::vector<PATypeHolder>().swap(TypeList);
  ValueList.clear();
  std::vector<const ParamAttrsList*>().swap(ParamAttrs);
  std::vector<BasicBlock*>().swap(FunctionBBs);
  std::vector<Function*>().swap(FunctionsWithBodies);
  DeferredFunctionInfo.clear();
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
  case 0: return GlobalValue::ExternalLinkage;
  case 1: return GlobalValue::WeakLinkage;
  case 2: return GlobalValue::AppendingLinkage;
  case 3: return GlobalValue::InternalLinkage;
  case 4: return GlobalValue::LinkOnceLinkage;
  case 5: return GlobalValue::DLLImportLinkage;
  case 6: return GlobalValue::DLLExportLinkage;
  case 7: return GlobalValue::ExternalWeakLinkage;
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
  case bitc::BINOP_ADD:  return Instruction::Add;
  case bitc::BINOP_SUB:  return Instruction::Sub;
  case bitc::BINOP_MUL:  return Instruction::Mul;
  case bitc::BINOP_UDIV: return Instruction::UDiv;
  case bitc::BINOP_SDIV:
    return Ty->isFPOrFPVector() ? Instruction::FDiv : Instruction::SDiv;
  case bitc::BINOP_UREM: return Instruction::URem;
  case bitc::BINOP_SREM:
    return Ty->isFPOrFPVector() ? Instruction::FRem : Instruction::SRem;
  case bitc::BINOP_SHL:  return Instruction::Shl;
  case bitc::BINOP_LSHR: return Instruction::LShr;
  case bitc::BINOP_ASHR: return Instruction::AShr;
  case bitc::BINOP_AND:  return Instruction::And;
  case bitc::BINOP_OR:   return Instruction::Or;
  case bitc::BINOP_XOR:  return Instruction::Xor;
  }
}


namespace {
  /// @brief A class for maintaining the slot number definition
  /// as a placeholder for the actual definition for forward constants defs.
  class ConstantPlaceHolder : public ConstantExpr {
    ConstantPlaceHolder();                       // DO NOT IMPLEMENT
    void operator=(const ConstantPlaceHolder &); // DO NOT IMPLEMENT
  public:
    Use Op;
    ConstantPlaceHolder(const Type *Ty)
      : ConstantExpr(Ty, Instruction::UserOp1, &Op, 1),
        Op(UndefValue::get(Type::Int32Ty), this) {
    }
  };
}

Constant *BitcodeReaderValueList::getConstantFwdRef(unsigned Idx,
                                                    const Type *Ty) {
  if (Idx >= size()) {
    // Insert a bunch of null values.
    Uses.resize(Idx+1);
    OperandList = &Uses[0];
    NumOperands = Idx+1;
  }

  if (Value *V = Uses[Idx]) {
    assert(Ty == V->getType() && "Type mismatch in constant table!");
    return cast<Constant>(V);
  }

  // Create and return a placeholder, which will later be RAUW'd.
  Constant *C = new ConstantPlaceHolder(Ty);
  Uses[Idx].init(C, this);
  return C;
}

Value *BitcodeReaderValueList::getValueFwdRef(unsigned Idx, const Type *Ty) {
  if (Idx >= size()) {
    // Insert a bunch of null values.
    Uses.resize(Idx+1);
    OperandList = &Uses[0];
    NumOperands = Idx+1;
  }
  
  if (Value *V = Uses[Idx]) {
    assert((Ty == 0 || Ty == V->getType()) && "Type mismatch in value table!");
    return V;
  }
  
  // No type specified, must be invalid reference.
  if (Ty == 0) return 0;
  
  // Create and return a placeholder, which will later be RAUW'd.
  Value *V = new Argument(Ty);
  Uses[Idx].init(V, this);
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
    TypeList.push_back(OpaqueType::get());
  return TypeList.back().get();
}

//===----------------------------------------------------------------------===//
//  Functions for parsing blocks from the bitcode file
//===----------------------------------------------------------------------===//

bool BitcodeReader::ParseParamAttrBlock() {
  if (Stream.EnterSubBlock(bitc::PARAMATTR_BLOCK_ID))
    return Error("Malformed block record");
  
  if (!ParamAttrs.empty())
    return Error("Multiple PARAMATTR blocks found!");
  
  SmallVector<uint64_t, 64> Record;
  
  ParamAttrsVector Attrs;
  
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

      ParamAttrsWithIndex PAWI;
      for (unsigned i = 0, e = Record.size(); i != e; i += 2) {
        PAWI.index = Record[i];
        PAWI.attrs = Record[i+1];
        Attrs.push_back(PAWI);
      }
      ParamAttrs.push_back(ParamAttrsList::get(Attrs));
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
      ResultTy = Type::VoidTy;
      break;
    case bitc::TYPE_CODE_FLOAT:     // FLOAT
      ResultTy = Type::FloatTy;
      break;
    case bitc::TYPE_CODE_DOUBLE:    // DOUBLE
      ResultTy = Type::DoubleTy;
      break;
    case bitc::TYPE_CODE_LABEL:     // LABEL
      ResultTy = Type::LabelTy;
      break;
    case bitc::TYPE_CODE_OPAQUE:    // OPAQUE
      ResultTy = 0;
      break;
    case bitc::TYPE_CODE_INTEGER:   // INTEGER: [width]
      if (Record.size() < 1)
        return Error("Invalid Integer type record");
      
      ResultTy = IntegerType::get(Record[0]);
      break;
    case bitc::TYPE_CODE_POINTER:   // POINTER: [pointee type]
      if (Record.size() < 1)
        return Error("Invalid POINTER type record");
      ResultTy = PointerType::get(getTypeByID(Record[0], true));
      break;
    case bitc::TYPE_CODE_FUNCTION: {
      // FUNCTION: [vararg, attrid, retty, paramty x N]
      if (Record.size() < 3)
        return Error("Invalid FUNCTION type record");
      std::vector<const Type*> ArgTys;
      for (unsigned i = 3, e = Record.size(); i != e; ++i)
        ArgTys.push_back(getTypeByID(Record[i], true));
      
      ResultTy = FunctionType::get(getTypeByID(Record[2], true), ArgTys,
                                   Record[0], getParamAttrs(Record[1]));
      break;
    }
    case bitc::TYPE_CODE_STRUCT: {  // STRUCT: [ispacked, eltty x N]
      if (Record.size() < 1)
        return Error("Invalid STRUCT type record");
      std::vector<const Type*> EltTys;
      for (unsigned i = 1, e = Record.size(); i != e; ++i)
        EltTys.push_back(getTypeByID(Record[i], true));
      ResultTy = StructType::get(EltTys, Record[0]);
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
      TypeList.push_back(ResultTy ? ResultTy : OpaqueType::get());
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
        return Error("Invalid TST_ENTRY record");
      unsigned ValueID = Record[0];
      if (ValueID >= ValueList.size())
        return Error("Invalid Value ID in VST_ENTRY record");
      Value *V = ValueList[ValueID];
      
      V->setName(&ValueName[0], ValueName.size());
      ValueName.clear();
      break;
    }
    case bitc::VST_CODE_BBENTRY: {
      if (ConvertToString(Record, 1, ValueName))
        return Error("Invalid VST_BBENTRY record");
      BasicBlock *BB = getBasicBlock(Record[0]);
      if (BB == 0)
        return Error("Invalid BB ID in VST_BBENTRY record");
      
      BB->setName(&ValueName[0], ValueName.size());
      ValueName.clear();
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
  const Type *CurTy = Type::Int32Ty;
  unsigned NextCstNo = ValueList.size();
  while (1) {
    unsigned Code = Stream.ReadCode();
    if (Code == bitc::END_BLOCK) {
      if (NextCstNo != ValueList.size())
        return Error("Invalid constant reference!");
      
      if (Stream.ReadBlockEnd())
        return Error("Error at end of constants block");
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
    Value *V = 0;
    switch (Stream.ReadRecord(Code, Record)) {
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
      if (!isa<IntegerType>(CurTy) || Record.empty())
        return Error("Invalid CST_INTEGER record");
      V = ConstantInt::get(CurTy, DecodeSignRotatedValue(Record[0]));
      break;
    case bitc::CST_CODE_WIDE_INTEGER: {// WIDE_INTEGER: [n x intval]
      if (!isa<IntegerType>(CurTy) || Record.empty())
        return Error("Invalid WIDE_INTEGER record");
      
      unsigned NumWords = Record.size();
      SmallVector<uint64_t, 8> Words;
      Words.resize(NumWords);
      for (unsigned i = 0; i != NumWords; ++i)
        Words[i] = DecodeSignRotatedValue(Record[i]);
      V = ConstantInt::get(APInt(cast<IntegerType>(CurTy)->getBitWidth(),
                                 NumWords, &Words[0]));
      break;
    }
    case bitc::CST_CODE_FLOAT:     // FLOAT: [fpval]
      if (Record.empty())
        return Error("Invalid FLOAT record");
      if (CurTy == Type::FloatTy)
        V = ConstantFP::get(CurTy, BitsToFloat(Record[0]));
      else if (CurTy == Type::DoubleTy)
        V = ConstantFP::get(CurTy, BitsToDouble(Record[0]));
      else
        V = UndefValue::get(CurTy);
      break;
      
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
        V = ConstantExpr::get(Opc, LHS, RHS);
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
    case bitc::CST_CODE_CE_GEP: {  // CE_GEP:        [n x operands]
      if (Record.size() & 1) return Error("Invalid CE_GEP record");
      SmallVector<Constant*, 16> Elts;
      for (unsigned i = 0, e = Record.size(); i != e; i += 2) {
        const Type *ElTy = getTypeByID(Record[i]);
        if (!ElTy) return Error("Invalid CE_GEP record");
        Elts.push_back(ValueList.getConstantFwdRef(Record[i+1], ElTy));
      }
      V = ConstantExpr::getGetElementPtr(Elts[0], &Elts[1], Elts.size()-1);
      break;
    }
    case bitc::CST_CODE_CE_SELECT:  // CE_SELECT: [opval#, opval#, opval#]
      if (Record.size() < 3) return Error("Invalid CE_SELECT record");
      V = ConstantExpr::getSelect(ValueList.getConstantFwdRef(Record[0],
                                                              Type::Int1Ty),
                                  ValueList.getConstantFwdRef(Record[1],CurTy),
                                  ValueList.getConstantFwdRef(Record[2],CurTy));
      break;
    case bitc::CST_CODE_CE_EXTRACTELT: { // CE_EXTRACTELT: [opty, opval, opval]
      if (Record.size() < 3) return Error("Invalid CE_EXTRACTELT record");
      const VectorType *OpTy = 
        dyn_cast_or_null<VectorType>(getTypeByID(Record[0]));
      if (OpTy == 0) return Error("Invalid CE_EXTRACTELT record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2],
                                                  OpTy->getElementType());
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
      Constant *Op2 = ValueList.getConstantFwdRef(Record[2], Type::Int32Ty);
      V = ConstantExpr::getInsertElement(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_SHUFFLEVEC: { // CE_SHUFFLEVEC: [opval, opval, opval]
      const VectorType *OpTy = dyn_cast<VectorType>(CurTy);
      if (Record.size() < 3 || OpTy == 0)
        return Error("Invalid CE_INSERTELT record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[0], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[1], OpTy);
      const Type *ShufTy=VectorType::get(Type::Int32Ty, OpTy->getNumElements());
      Constant *Op2 = ValueList.getConstantFwdRef(Record[2], ShufTy);
      V = ConstantExpr::getShuffleVector(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_CMP: {     // CE_CMP: [opty, opval, opval, pred]
      if (Record.size() < 4) return Error("Invalid CE_CMP record");
      const Type *OpTy = getTypeByID(Record[0]);
      if (OpTy == 0) return Error("Invalid CE_CMP record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2], OpTy);

      if (OpTy->isFloatingPoint())
        V = ConstantExpr::getFCmp(Record[3], Op0, Op1);
      else
        V = ConstantExpr::getICmp(Record[3], Op0, Op1);
      break;
    }
    case bitc::CST_CODE_INLINEASM: {
      if (Record.size() < 2) return Error("Invalid INLINEASM record");
      std::string AsmStr, ConstrStr;
      bool HasSideEffects = Record[0];
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
                         AsmStr, ConstrStr, HasSideEffects);
      break;
    }
    }
    
    ValueList.AssignValue(V, NextCstNo);
    ++NextCstNo;
  }
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
  DeferredFunctionInfo[Fn] = std::make_pair(CurBit, Fn->getLinkage());
  
  // Set the functions linkage to GhostLinkage so we know it is lazily
  // deserialized.
  Fn->setLinkage(GlobalValue::GhostLinkage);
  
  // Skip over the function block for now.
  if (Stream.SkipBlock())
    return Error("Malformed block record");
  return false;
}

bool BitcodeReader::ParseModule(const std::string &ModuleID) {
  // Reject multiple MODULE_BLOCK's in a single bitstream.
  if (TheModule)
    return Error("Multiple MODULE_BLOCKs in same stream");
  
  if (Stream.EnterSubBlock(bitc::MODULE_BLOCK_ID))
    return Error("Malformed block record");

  // Otherwise, create the module.
  TheModule = new Module(ModuleID);
  
  SmallVector<uint64_t, 64> Record;
  std::vector<std::string> SectionTable;

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
        if (ParseParamAttrBlock())
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
    // GLOBALVAR: [type, isconst, initid, 
    //             linkage, alignment, section, visibility, threadlocal]
    case bitc::MODULE_CODE_GLOBALVAR: {
      if (Record.size() < 6)
        return Error("Invalid MODULE_CODE_GLOBALVAR record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!isa<PointerType>(Ty))
        return Error("Global not a pointer type!");
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
        new GlobalVariable(Ty, isConstant, Linkage, 0, "", TheModule);
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
    //             alignment, section, visibility]
    case bitc::MODULE_CODE_FUNCTION: {
      if (Record.size() < 8)
        return Error("Invalid MODULE_CODE_FUNCTION record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!isa<PointerType>(Ty))
        return Error("Function not a pointer type!");
      const FunctionType *FTy =
        dyn_cast<FunctionType>(cast<PointerType>(Ty)->getElementType());
      if (!FTy)
        return Error("Function not a pointer to function type!");

      Function *Func = new Function(FTy, GlobalValue::ExternalLinkage,
                                    "", TheModule);

      Func->setCallingConv(Record[1]);
      bool isProto = Record[2];
      Func->setLinkage(GetDecodedLinkage(Record[3]));
      
      assert(Func->getFunctionType()->getParamAttrs() == 
             getParamAttrs(Record[4]));
      
      Func->setAlignment((1 << Record[5]) >> 1);
      if (Record[6]) {
        if (Record[6]-1 >= SectionTable.size())
          return Error("Invalid section ID");
        Func->setSection(SectionTable[Record[6]-1]);
      }
      Func->setVisibility(GetDecodedVisibility(Record[7]));
      
      ValueList.push_back(Func);
      
      // If this is a function with a body, remember the prototype we are
      // creating now, so that we can match up the body with them later.
      if (!isProto)
        FunctionsWithBodies.push_back(Func);
      break;
    }
    // ALIAS: [alias type, aliasee val#, linkage]
    case bitc::MODULE_CODE_ALIAS: {
      if (Record.size() < 3)
        return Error("Invalid MODULE_ALIAS record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!isa<PointerType>(Ty))
        return Error("Function not a pointer type!");
      
      GlobalAlias *NewGA = new GlobalAlias(Ty, GetDecodedLinkage(Record[2]),
                                           "", 0, TheModule);
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


bool BitcodeReader::ParseBitcode() {
  TheModule = 0;
  
  if (Buffer->getBufferSize() & 3)
    return Error("Bitcode stream should be a multiple of 4 bytes in length");
  
  unsigned char *BufPtr = (unsigned char *)Buffer->getBufferStart();
  Stream.init(BufPtr, BufPtr+Buffer->getBufferSize());
  
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
      if (ParseModule(Buffer->getBufferIdentifier()))
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


/// ParseFunctionBody - Lazily parse the specified function body block.
bool BitcodeReader::ParseFunctionBody(Function *F) {
  if (Stream.EnterSubBlock(bitc::FUNCTION_BLOCK_ID))
    return Error("Malformed block record");
  
  unsigned ModuleValueListSize = ValueList.size();
  
  // Add all the function arguments to the value table.
  for(Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    ValueList.push_back(I);
  
  unsigned NextValueNo = ValueList.size();
  BasicBlock *CurBB = 0;
  unsigned CurBBNo = 0;

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
    switch (Stream.ReadRecord(Code, Record)) {
    default: // Default behavior: reject
      return Error("Unknown instruction");
    case bitc::FUNC_CODE_DECLAREBLOCKS:     // DECLAREBLOCKS: [nblocks]
      if (Record.size() < 1 || Record[0] == 0)
        return Error("Invalid DECLAREBLOCKS record");
      // Create all the basic blocks for the function.
      FunctionBBs.resize(Record[0]);
      for (unsigned i = 0, e = FunctionBBs.size(); i != e; ++i)
        FunctionBBs[i] = new BasicBlock("", F);
      CurBB = FunctionBBs[0];
      continue;
      
    case bitc::FUNC_CODE_INST_BINOP: {    // BINOP: [opval, ty, opval, opcode]
      unsigned OpNum = 0;
      Value *LHS, *RHS;
      if (getValueTypePair(Record, OpNum, NextValueNo, LHS) ||
          getValue(Record, OpNum, LHS->getType(), RHS) ||
          OpNum+1 != Record.size())
        return Error("Invalid BINOP record");
      
      int Opc = GetDecodedBinaryOpcode(Record[OpNum], LHS->getType());
      if (Opc == -1) return Error("Invalid BINOP record");
      I = BinaryOperator::create((Instruction::BinaryOps)Opc, LHS, RHS);
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
      I = CastInst::create((Instruction::CastOps)Opc, Op, ResTy);
      break;
    }
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

      I = new GetElementPtrInst(BasePtr, &GEPIdx[0], GEPIdx.size());
      break;
    }
      
    case bitc::FUNC_CODE_INST_SELECT: { // SELECT: [opval, ty, opval, opval]
      unsigned OpNum = 0;
      Value *TrueVal, *FalseVal, *Cond;
      if (getValueTypePair(Record, OpNum, NextValueNo, TrueVal) ||
          getValue(Record, OpNum, TrueVal->getType(), FalseVal) ||
          getValue(Record, OpNum, Type::Int1Ty, Cond))
        return Error("Invalid SELECT record");
      
      I = new SelectInst(Cond, TrueVal, FalseVal);
      break;
    }
      
    case bitc::FUNC_CODE_INST_EXTRACTELT: { // EXTRACTELT: [opty, opval, opval]
      unsigned OpNum = 0;
      Value *Vec, *Idx;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec) ||
          getValue(Record, OpNum, Type::Int32Ty, Idx))
        return Error("Invalid EXTRACTELT record");
      I = new ExtractElementInst(Vec, Idx);
      break;
    }
      
    case bitc::FUNC_CODE_INST_INSERTELT: { // INSERTELT: [ty, opval,opval,opval]
      unsigned OpNum = 0;
      Value *Vec, *Elt, *Idx;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec) ||
          getValue(Record, OpNum, 
                   cast<VectorType>(Vec->getType())->getElementType(), Elt) ||
          getValue(Record, OpNum, Type::Int32Ty, Idx))
        return Error("Invalid INSERTELT record");
      I = new InsertElementInst(Vec, Elt, Idx);
      break;
    }
      
    case bitc::FUNC_CODE_INST_SHUFFLEVEC: {// SHUFFLEVEC: [opval,ty,opval,opval]
      unsigned OpNum = 0;
      Value *Vec1, *Vec2, *Mask;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec1) ||
          getValue(Record, OpNum, Vec1->getType(), Vec2))
        return Error("Invalid SHUFFLEVEC record");

      const Type *MaskTy =
        VectorType::get(Type::Int32Ty, 
                        cast<VectorType>(Vec1->getType())->getNumElements());

      if (getValue(Record, OpNum, MaskTy, Mask))
        return Error("Invalid SHUFFLEVEC record");
      I = new ShuffleVectorInst(Vec1, Vec2, Mask);
      break;
    }
      
    case bitc::FUNC_CODE_INST_CMP: { // CMP: [opty, opval, opval, pred]
      unsigned OpNum = 0;
      Value *LHS, *RHS;
      if (getValueTypePair(Record, OpNum, NextValueNo, LHS) ||
          getValue(Record, OpNum, LHS->getType(), RHS) ||
          OpNum+1 != Record.size())
        return Error("Invalid CMP record");
      
      if (LHS->getType()->isFPOrFPVector())
        I = new FCmpInst((FCmpInst::Predicate)Record[OpNum], LHS, RHS);
      else
        I = new ICmpInst((ICmpInst::Predicate)Record[OpNum], LHS, RHS);
      break;
    }
    
    case bitc::FUNC_CODE_INST_RET: // RET: [opty,opval<optional>]
      if (Record.size() == 0) {
        I = new ReturnInst();
        break;
      } else {
        unsigned OpNum = 0;
        Value *Op;
        if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
            OpNum != Record.size())
          return Error("Invalid RET record");
        I = new ReturnInst(Op);
        break;
      }
    case bitc::FUNC_CODE_INST_BR: { // BR: [bb#, bb#, opval] or [bb#]
      if (Record.size() != 1 && Record.size() != 3)
        return Error("Invalid BR record");
      BasicBlock *TrueDest = getBasicBlock(Record[0]);
      if (TrueDest == 0)
        return Error("Invalid BR record");

      if (Record.size() == 1)
        I = new BranchInst(TrueDest);
      else {
        BasicBlock *FalseDest = getBasicBlock(Record[1]);
        Value *Cond = getFnValueByID(Record[2], Type::Int1Ty);
        if (FalseDest == 0 || Cond == 0)
          return Error("Invalid BR record");
        I = new BranchInst(TrueDest, FalseDest, Cond);
      }
      break;
    }
    case bitc::FUNC_CODE_INST_SWITCH: { // SWITCH: [opty, opval, n, n x ops]
      if (Record.size() < 3 || (Record.size() & 1) == 0)
        return Error("Invalid SWITCH record");
      const Type *OpTy = getTypeByID(Record[0]);
      Value *Cond = getFnValueByID(Record[1], OpTy);
      BasicBlock *Default = getBasicBlock(Record[2]);
      if (OpTy == 0 || Cond == 0 || Default == 0)
        return Error("Invalid SWITCH record");
      unsigned NumCases = (Record.size()-3)/2;
      SwitchInst *SI = new SwitchInst(Cond, Default, NumCases);
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
      
    case bitc::FUNC_CODE_INST_INVOKE: { // INVOKE: [cc,fnty, op0,op1,op2, ...]
      if (Record.size() < 4) return Error("Invalid INVOKE record");
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
      
      assert(FTy->getParamAttrs() == getParamAttrs(Record[0]));

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
      
      I = new InvokeInst(Callee, NormalBB, UnwindBB, &Ops[0], Ops.size());
      cast<InvokeInst>(I)->setCallingConv(CCInfo);
      break;
    }
    case bitc::FUNC_CODE_INST_UNWIND: // UNWIND
      I = new UnwindInst();
      break;
    case bitc::FUNC_CODE_INST_UNREACHABLE: // UNREACHABLE
      I = new UnreachableInst();
      break;
    case bitc::FUNC_CODE_INST_PHI: { // PHI: [ty, val0,bb0, ...]
      if (Record.size() < 1 || ((Record.size()-1)&1))
        return Error("Invalid PHI record");
      const Type *Ty = getTypeByID(Record[0]);
      if (!Ty) return Error("Invalid PHI record");
      
      PHINode *PN = new PHINode(Ty);
      PN->reserveOperandSpace(Record.size()-1);
      
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
      if (Record.size() < 3)
        return Error("Invalid MALLOC record");
      const PointerType *Ty =
        dyn_cast_or_null<PointerType>(getTypeByID(Record[0]));
      Value *Size = getFnValueByID(Record[1], Type::Int32Ty);
      unsigned Align = Record[2];
      if (!Ty || !Size) return Error("Invalid MALLOC record");
      I = new MallocInst(Ty->getElementType(), Size, (1 << Align) >> 1);
      break;
    }
    case bitc::FUNC_CODE_INST_FREE: { // FREE: [op, opty]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          OpNum != Record.size())
        return Error("Invalid FREE record");
      I = new FreeInst(Op);
      break;
    }
    case bitc::FUNC_CODE_INST_ALLOCA: { // ALLOCA: [instty, op, align]
      if (Record.size() < 3)
        return Error("Invalid ALLOCA record");
      const PointerType *Ty =
        dyn_cast_or_null<PointerType>(getTypeByID(Record[0]));
      Value *Size = getFnValueByID(Record[1], Type::Int32Ty);
      unsigned Align = Record[2];
      if (!Ty || !Size) return Error("Invalid ALLOCA record");
      I = new AllocaInst(Ty->getElementType(), Size, (1 << Align) >> 1);
      break;
    }
    case bitc::FUNC_CODE_INST_LOAD: { // LOAD: [opty, op, align, vol]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          OpNum+2 != Record.size())
        return Error("Invalid LOAD record");
      
      I = new LoadInst(Op, "", Record[OpNum+1], (1 << Record[OpNum]) >> 1);
      break;
    }
    case bitc::FUNC_CODE_INST_STORE: { // STORE:[val, valty, ptr, align, vol]
      unsigned OpNum = 0;
      Value *Val, *Ptr;
      if (getValueTypePair(Record, OpNum, NextValueNo, Val) ||
          getValue(Record, OpNum, PointerType::get(Val->getType()), Ptr) ||
          OpNum+2 != Record.size())
        return Error("Invalid STORE record");
      
      I = new StoreInst(Val, Ptr, Record[OpNum+1], (1 << Record[OpNum]) >> 1);
      break;
    }
    case bitc::FUNC_CODE_INST_CALL: { // CALL: [cc, fnty, fnid, arg0, arg1...]
      if (Record.size() < 2)
        return Error("Invalid CALL record");
      
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
      
      assert(FTy->getParamAttrs() == getParamAttrs(Record[0]));
      
      SmallVector<Value*, 16> Args;
      // Read the fixed params.
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i, ++OpNum) {
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
      
      I = new CallInst(Callee, &Args[0], Args.size());
      cast<CallInst>(I)->setCallingConv(CCInfo>>1);
      cast<CallInst>(I)->setTailCall(CCInfo & 1);
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
    if (I && I->getType() != Type::VoidTy)
      ValueList.AssignValue(I, NextValueNo++);
  }
  
  // Check the function list for unresolved values.
  if (Argument *A = dyn_cast<Argument>(ValueList.back())) {
    if (A->getParent() == 0) {
      // We found at least one unresolved value.  Nuke them all to avoid leaks.
      for (unsigned i = ModuleValueListSize, e = ValueList.size(); i != e; ++i){
        if ((A = dyn_cast<Argument>(ValueList.back())) && A->getParent() == 0) {
          A->replaceAllUsesWith(UndefValue::get(A->getType()));
          delete A;
        }
      }
      return Error("Never resolved value found in function!");
    }
  }
  
  // Trim the value list down to the size it was before we parsed this function.
  ValueList.shrinkTo(ModuleValueListSize);
  std::vector<BasicBlock*>().swap(FunctionBBs);
  
  return false;
}

//===----------------------------------------------------------------------===//
// ModuleProvider implementation
//===----------------------------------------------------------------------===//


bool BitcodeReader::materializeFunction(Function *F, std::string *ErrInfo) {
  // If it already is material, ignore the request.
  if (!F->hasNotBeenReadFromBytecode()) return false;
  
  DenseMap<Function*, std::pair<uint64_t, unsigned> >::iterator DFII = 
    DeferredFunctionInfo.find(F);
  assert(DFII != DeferredFunctionInfo.end() && "Deferred function not found!");
  
  // Move the bit stream to the saved position of the deferred function body and
  // restore the real linkage type for the function.
  Stream.JumpToBit(DFII->second.first);
  F->setLinkage((GlobalValue::LinkageTypes)DFII->second.second);
  
  if (ParseFunctionBody(F)) {
    if (ErrInfo) *ErrInfo = ErrorString;
    return true;
  }
  
  return false;
}

void BitcodeReader::dematerializeFunction(Function *F) {
  // If this function isn't materialized, or if it is a proto, this is a noop.
  if (F->hasNotBeenReadFromBytecode() || F->isDeclaration())
    return;
  
  assert(DeferredFunctionInfo.count(F) && "No info to read function later?");
  
  // Just forget the function body, we can remat it later.
  F->deleteBody();
  F->setLinkage(GlobalValue::GhostLinkage);
}


Module *BitcodeReader::materializeModule(std::string *ErrInfo) {
  for (DenseMap<Function*, std::pair<uint64_t, unsigned> >::iterator I = 
       DeferredFunctionInfo.begin(), E = DeferredFunctionInfo.end(); I != E;
       ++I) {
    Function *F = I->first;
    if (F->hasNotBeenReadFromBytecode() &&
        materializeFunction(F, ErrInfo))
      return 0;
  }
  return TheModule;
}


/// This method is provided by the parent ModuleProvde class and overriden
/// here. It simply releases the module from its provided and frees up our
/// state.
/// @brief Release our hold on the generated module
Module *BitcodeReader::releaseModule(std::string *ErrInfo) {
  // Since we're losing control of this Module, we must hand it back complete
  Module *M = ModuleProvider::releaseModule(ErrInfo);
  FreeState();
  return M;
}


//===----------------------------------------------------------------------===//
// External interface
//===----------------------------------------------------------------------===//

/// getBitcodeModuleProvider - lazy function-at-a-time loading from a file.
///
ModuleProvider *llvm::getBitcodeModuleProvider(MemoryBuffer *Buffer,
                                               std::string *ErrMsg) {
  BitcodeReader *R = new BitcodeReader(Buffer);
  if (R->ParseBitcode()) {
    if (ErrMsg)
      *ErrMsg = R->getErrorString();
    
    // Don't let the BitcodeReader dtor delete 'Buffer'.
    R->releaseMemoryBuffer();
    delete R;
    return 0;
  }
  return R;
}

/// ParseBitcodeFile - Read the specified bitcode file, returning the module.
/// If an error occurs, return null and fill in *ErrMsg if non-null.
Module *llvm::ParseBitcodeFile(MemoryBuffer *Buffer, std::string *ErrMsg){
  BitcodeReader *R;
  R = static_cast<BitcodeReader*>(getBitcodeModuleProvider(Buffer, ErrMsg));
  if (!R) return 0;
  
  // Read in the entire module.
  Module *M = R->materializeModule(ErrMsg);

  // Don't let the BitcodeReader dtor delete 'Buffer', regardless of whether
  // there was an error.
  R->releaseMemoryBuffer();
  
  // If there was no error, tell ModuleProvider not to delete it when its dtor
  // is run.
  if (M)
    M = R->releaseModule(ErrMsg);
  
  delete R;
  return M;
}
