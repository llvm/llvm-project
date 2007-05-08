//===--- Bitcode/Writer/BitcodeWriter.cpp - Bitcode Writer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Bitcode writer implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "ValueEnumerator.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Module.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

/// These are manifest constants used by the bitcode writer. They do not need to
/// be kept in sync with the reader, but need to be consistent within this file.
enum {
  CurVersion = 0,
  
  // VALUE_SYMTAB_BLOCK abbrev id's.
  VST_ENTRY_8_ABBREV = bitc::FIRST_APPLICATION_ABBREV,
  VST_ENTRY_7_ABBREV,
  VST_ENTRY_6_ABBREV,
  VST_BBENTRY_6_ABBREV,
  
  // CONSTANTS_BLOCK abbrev id's.
  CONSTANTS_SETTYPE_ABBREV = bitc::FIRST_APPLICATION_ABBREV,
  CONSTANTS_INTEGER_ABBREV,
  CONSTANTS_CE_CAST_Abbrev,
  CONSTANTS_NULL_Abbrev,
  
  // FUNCTION_BLOCK abbrev id's.
  FUNCTION_INST_LOAD_ABBREV = bitc::FIRST_APPLICATION_ABBREV,
  FUNCTION_INST_BINOP_ABBREV,
  FUNCTION_INST_CAST_ABBREV,
  FUNCTION_INST_RET_VOID_ABBREV,
  FUNCTION_INST_RET_VAL_ABBREV,
  FUNCTION_INST_UNREACHABLE_ABBREV
};


static unsigned GetEncodedCastOpcode(unsigned Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown cast instruction!");
  case Instruction::Trunc   : return bitc::CAST_TRUNC;
  case Instruction::ZExt    : return bitc::CAST_ZEXT;
  case Instruction::SExt    : return bitc::CAST_SEXT;
  case Instruction::FPToUI  : return bitc::CAST_FPTOUI;
  case Instruction::FPToSI  : return bitc::CAST_FPTOSI;
  case Instruction::UIToFP  : return bitc::CAST_UITOFP;
  case Instruction::SIToFP  : return bitc::CAST_SITOFP;
  case Instruction::FPTrunc : return bitc::CAST_FPTRUNC;
  case Instruction::FPExt   : return bitc::CAST_FPEXT;
  case Instruction::PtrToInt: return bitc::CAST_PTRTOINT;
  case Instruction::IntToPtr: return bitc::CAST_INTTOPTR;
  case Instruction::BitCast : return bitc::CAST_BITCAST;
  }
}

static unsigned GetEncodedBinaryOpcode(unsigned Opcode) {
  switch (Opcode) {
  default: assert(0 && "Unknown binary instruction!");
  case Instruction::Add:  return bitc::BINOP_ADD;
  case Instruction::Sub:  return bitc::BINOP_SUB;
  case Instruction::Mul:  return bitc::BINOP_MUL;
  case Instruction::UDiv: return bitc::BINOP_UDIV;
  case Instruction::FDiv:
  case Instruction::SDiv: return bitc::BINOP_SDIV;
  case Instruction::URem: return bitc::BINOP_UREM;
  case Instruction::FRem:
  case Instruction::SRem: return bitc::BINOP_SREM;
  case Instruction::Shl:  return bitc::BINOP_SHL;
  case Instruction::LShr: return bitc::BINOP_LSHR;
  case Instruction::AShr: return bitc::BINOP_ASHR;
  case Instruction::And:  return bitc::BINOP_AND;
  case Instruction::Or:   return bitc::BINOP_OR;
  case Instruction::Xor:  return bitc::BINOP_XOR;
  }
}



static void WriteStringRecord(unsigned Code, const std::string &Str, 
                              unsigned AbbrevToUse, BitstreamWriter &Stream) {
  SmallVector<unsigned, 64> Vals;
  
  // Code: [strchar x N]
  for (unsigned i = 0, e = Str.size(); i != e; ++i)
    Vals.push_back(Str[i]);
    
  // Emit the finished record.
  Stream.EmitRecord(Code, Vals, AbbrevToUse);
}

// Emit information about parameter attributes.
static void WriteParamAttrTable(const ValueEnumerator &VE, 
                                BitstreamWriter &Stream) {
  const std::vector<const ParamAttrsList*> &Attrs = VE.getParamAttrs();
  if (Attrs.empty()) return;
  
  Stream.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 3);

  SmallVector<uint64_t, 64> Record;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
    const ParamAttrsList *A = Attrs[i];
    for (unsigned op = 0, e = A->size(); op != e; ++op) {
      Record.push_back(A->getParamIndex(op));
      Record.push_back(A->getParamAttrsAtIndex(op));
    }
    
    Stream.EmitRecord(bitc::PARAMATTR_CODE_ENTRY, Record);
    Record.clear();
  }
  
  Stream.ExitBlock();
}

/// WriteTypeTable - Write out the type table for a module.
static void WriteTypeTable(const ValueEnumerator &VE, BitstreamWriter &Stream) {
  const ValueEnumerator::TypeList &TypeList = VE.getTypes();
  
  Stream.EnterSubblock(bitc::TYPE_BLOCK_ID, 4 /*count from # abbrevs */);
  SmallVector<uint64_t, 64> TypeVals;
  
  // Abbrev for TYPE_CODE_POINTER.
  BitCodeAbbrev *Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_POINTER));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                            Log2_32_Ceil(VE.getTypes().size()+1)));
  unsigned PtrAbbrev = Stream.EmitAbbrev(Abbv);
  
  // Abbrev for TYPE_CODE_FUNCTION.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_FUNCTION));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1));  // isvararg
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                            Log2_32_Ceil(VE.getParamAttrs().size()+1)));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                            Log2_32_Ceil(VE.getTypes().size()+1)));
  unsigned FunctionAbbrev = Stream.EmitAbbrev(Abbv);
  
  // Abbrev for TYPE_CODE_STRUCT.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_STRUCT));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1));  // ispacked
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                            Log2_32_Ceil(VE.getTypes().size()+1)));
  unsigned StructAbbrev = Stream.EmitAbbrev(Abbv);
 
  // Abbrev for TYPE_CODE_ARRAY.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_ARRAY));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));   // size
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                            Log2_32_Ceil(VE.getTypes().size()+1)));
  unsigned ArrayAbbrev = Stream.EmitAbbrev(Abbv);
  
  // Emit an entry count so the reader can reserve space.
  TypeVals.push_back(TypeList.size());
  Stream.EmitRecord(bitc::TYPE_CODE_NUMENTRY, TypeVals);
  TypeVals.clear();
  
  // Loop over all of the types, emitting each in turn.
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    const Type *T = TypeList[i].first;
    int AbbrevToUse = 0;
    unsigned Code = 0;
    
    switch (T->getTypeID()) {
    case Type::PackedStructTyID: // FIXME: Delete Type::PackedStructTyID.
    default: assert(0 && "Unknown type!");
    case Type::VoidTyID:   Code = bitc::TYPE_CODE_VOID;   break;
    case Type::FloatTyID:  Code = bitc::TYPE_CODE_FLOAT;  break;
    case Type::DoubleTyID: Code = bitc::TYPE_CODE_DOUBLE; break;
    case Type::LabelTyID:  Code = bitc::TYPE_CODE_LABEL;  break;
    case Type::OpaqueTyID: Code = bitc::TYPE_CODE_OPAQUE; break;
    case Type::IntegerTyID:
      // INTEGER: [width]
      Code = bitc::TYPE_CODE_INTEGER;
      TypeVals.push_back(cast<IntegerType>(T)->getBitWidth());
      break;
    case Type::PointerTyID:
      // POINTER: [pointee type]
      Code = bitc::TYPE_CODE_POINTER;
      TypeVals.push_back(VE.getTypeID(cast<PointerType>(T)->getElementType()));
      AbbrevToUse = PtrAbbrev;
      break;

    case Type::FunctionTyID: {
      const FunctionType *FT = cast<FunctionType>(T);
      // FUNCTION: [isvararg, attrid, retty, paramty x N]
      Code = bitc::TYPE_CODE_FUNCTION;
      TypeVals.push_back(FT->isVarArg());
      TypeVals.push_back(VE.getParamAttrID(FT->getParamAttrs()));
      TypeVals.push_back(VE.getTypeID(FT->getReturnType()));
      for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i)
        TypeVals.push_back(VE.getTypeID(FT->getParamType(i)));
      AbbrevToUse = FunctionAbbrev;
      break;
    }
    case Type::StructTyID: {
      const StructType *ST = cast<StructType>(T);
      // STRUCT: [ispacked, eltty x N]
      Code = bitc::TYPE_CODE_STRUCT;
      TypeVals.push_back(ST->isPacked());
      // Output all of the element types.
      for (StructType::element_iterator I = ST->element_begin(),
           E = ST->element_end(); I != E; ++I)
        TypeVals.push_back(VE.getTypeID(*I));
      AbbrevToUse = StructAbbrev;
      break;
    }
    case Type::ArrayTyID: {
      const ArrayType *AT = cast<ArrayType>(T);
      // ARRAY: [numelts, eltty]
      Code = bitc::TYPE_CODE_ARRAY;
      TypeVals.push_back(AT->getNumElements());
      TypeVals.push_back(VE.getTypeID(AT->getElementType()));
      AbbrevToUse = ArrayAbbrev;
      break;
    }
    case Type::VectorTyID: {
      const VectorType *VT = cast<VectorType>(T);
      // VECTOR [numelts, eltty]
      Code = bitc::TYPE_CODE_VECTOR;
      TypeVals.push_back(VT->getNumElements());
      TypeVals.push_back(VE.getTypeID(VT->getElementType()));
      break;
    }
    }

    // Emit the finished record.
    Stream.EmitRecord(Code, TypeVals, AbbrevToUse);
    TypeVals.clear();
  }
  
  Stream.ExitBlock();
}

static unsigned getEncodedLinkage(const GlobalValue *GV) {
  switch (GV->getLinkage()) {
  default: assert(0 && "Invalid linkage!");
  case GlobalValue::ExternalLinkage:     return 0;
  case GlobalValue::WeakLinkage:         return 1;
  case GlobalValue::AppendingLinkage:    return 2;
  case GlobalValue::InternalLinkage:     return 3;
  case GlobalValue::LinkOnceLinkage:     return 4;
  case GlobalValue::DLLImportLinkage:    return 5;
  case GlobalValue::DLLExportLinkage:    return 6;
  case GlobalValue::ExternalWeakLinkage: return 7;
  }
}

static unsigned getEncodedVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  default: assert(0 && "Invalid visibility!");
  case GlobalValue::DefaultVisibility:   return 0;
  case GlobalValue::HiddenVisibility:    return 1;
  case GlobalValue::ProtectedVisibility: return 2;
  }
}

// Emit top-level description of module, including target triple, inline asm,
// descriptors for global variables, and function prototype info.
static void WriteModuleInfo(const Module *M, const ValueEnumerator &VE,
                            BitstreamWriter &Stream) {
  // Emit the list of dependent libraries for the Module.
  for (Module::lib_iterator I = M->lib_begin(), E = M->lib_end(); I != E; ++I)
    WriteStringRecord(bitc::MODULE_CODE_DEPLIB, *I, 0/*TODO*/, Stream);

  // Emit various pieces of data attached to a module.
  if (!M->getTargetTriple().empty())
    WriteStringRecord(bitc::MODULE_CODE_TRIPLE, M->getTargetTriple(),
                      0/*TODO*/, Stream);
  if (!M->getDataLayout().empty())
    WriteStringRecord(bitc::MODULE_CODE_DATALAYOUT, M->getDataLayout(),
                      0/*TODO*/, Stream);
  if (!M->getModuleInlineAsm().empty())
    WriteStringRecord(bitc::MODULE_CODE_ASM, M->getModuleInlineAsm(),
                      0/*TODO*/, Stream);

  // Emit information about sections, computing how many there are.  Also
  // compute the maximum alignment value.
  std::map<std::string, unsigned> SectionMap;
  unsigned MaxAlignment = 0;
  unsigned MaxGlobalType = 0;
  for (Module::const_global_iterator GV = M->global_begin(),E = M->global_end();
       GV != E; ++GV) {
    MaxAlignment = std::max(MaxAlignment, GV->getAlignment());
    MaxGlobalType = std::max(MaxGlobalType, VE.getTypeID(GV->getType()));
    
    if (!GV->hasSection()) continue;
    // Give section names unique ID's.
    unsigned &Entry = SectionMap[GV->getSection()];
    if (Entry != 0) continue;
    WriteStringRecord(bitc::MODULE_CODE_SECTIONNAME, GV->getSection(),
                      0/*TODO*/, Stream);
    Entry = SectionMap.size();
  }
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    MaxAlignment = std::max(MaxAlignment, F->getAlignment());
    if (!F->hasSection()) continue;
    // Give section names unique ID's.
    unsigned &Entry = SectionMap[F->getSection()];
    if (Entry != 0) continue;
    WriteStringRecord(bitc::MODULE_CODE_SECTIONNAME, F->getSection(),
                      0/*TODO*/, Stream);
    Entry = SectionMap.size();
  }
  
  // Emit abbrev for globals, now that we know # sections and max alignment.
  unsigned SimpleGVarAbbrev = 0;
  if (!M->global_empty()) { 
    // Add an abbrev for common globals with no visibility or thread localness.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::MODULE_CODE_GLOBALVAR));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                              Log2_32_Ceil(MaxGlobalType+1)));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1));      // Constant.
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));        // Initializer.
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3));      // Linkage.
    if (MaxAlignment == 0)                                      // Alignment.
      Abbv->Add(BitCodeAbbrevOp(0));
    else {
      unsigned MaxEncAlignment = Log2_32(MaxAlignment)+1;
      Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                               Log2_32_Ceil(MaxEncAlignment+1)));
    }
    if (SectionMap.empty())                                    // Section.
      Abbv->Add(BitCodeAbbrevOp(0));
    else
      Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                               Log2_32_Ceil(SectionMap.size()+1)));
    // Don't bother emitting vis + thread local.
    SimpleGVarAbbrev = Stream.EmitAbbrev(Abbv);
  }
  
  // Emit the global variable information.
  SmallVector<unsigned, 64> Vals;
  for (Module::const_global_iterator GV = M->global_begin(),E = M->global_end();
       GV != E; ++GV) {
    unsigned AbbrevToUse = 0;

    // GLOBALVAR: [type, isconst, initid, 
    //             linkage, alignment, section, visibility, threadlocal]
    Vals.push_back(VE.getTypeID(GV->getType()));
    Vals.push_back(GV->isConstant());
    Vals.push_back(GV->isDeclaration() ? 0 :
                   (VE.getValueID(GV->getInitializer()) + 1));
    Vals.push_back(getEncodedLinkage(GV));
    Vals.push_back(Log2_32(GV->getAlignment())+1);
    Vals.push_back(GV->hasSection() ? SectionMap[GV->getSection()] : 0);
    if (GV->isThreadLocal() || 
        GV->getVisibility() != GlobalValue::DefaultVisibility) {
      Vals.push_back(getEncodedVisibility(GV));
      Vals.push_back(GV->isThreadLocal());
    } else {
      AbbrevToUse = SimpleGVarAbbrev;
    }
    
    Stream.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Vals, AbbrevToUse);
    Vals.clear();
  }

  // Emit the function proto information.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    // FUNCTION:  [type, callingconv, isproto, linkage, alignment, section,
    //             visibility]
    Vals.push_back(VE.getTypeID(F->getType()));
    Vals.push_back(F->getCallingConv());
    Vals.push_back(F->isDeclaration());
    Vals.push_back(getEncodedLinkage(F));
    
    // Note: we emit the param attr ID number for the function type of this
    // function.  In the future, we intend for attrs to be properties of
    // functions, instead of on the type.  This is to support this future work.
    Vals.push_back(VE.getParamAttrID(F->getFunctionType()->getParamAttrs()));
    
    Vals.push_back(Log2_32(F->getAlignment())+1);
    Vals.push_back(F->hasSection() ? SectionMap[F->getSection()] : 0);
    Vals.push_back(getEncodedVisibility(F));
    
    unsigned AbbrevToUse = 0;
    Stream.EmitRecord(bitc::MODULE_CODE_FUNCTION, Vals, AbbrevToUse);
    Vals.clear();
  }
  
  
  // Emit the alias information.
  for (Module::const_alias_iterator AI = M->alias_begin(), E = M->alias_end();
       AI != E; ++AI) {
    Vals.push_back(VE.getTypeID(AI->getType()));
    Vals.push_back(VE.getValueID(AI->getAliasee()));
    Vals.push_back(getEncodedLinkage(AI));
    unsigned AbbrevToUse = 0;
    Stream.EmitRecord(bitc::MODULE_CODE_ALIAS, Vals, AbbrevToUse);
    Vals.clear();
  }
}


static void WriteConstants(unsigned FirstVal, unsigned LastVal,
                           const ValueEnumerator &VE,
                           BitstreamWriter &Stream, bool isGlobal) {
  if (FirstVal == LastVal) return;
  
  Stream.EnterSubblock(bitc::CONSTANTS_BLOCK_ID, 4);

  unsigned AggregateAbbrev = 0;
  unsigned String8Abbrev = 0;
  unsigned CString7Abbrev = 0;
  unsigned CString6Abbrev = 0;
  // If this is a constant pool for the module, emit module-specific abbrevs.
  if (isGlobal) {
    // Abbrev for CST_CODE_AGGREGATE.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_AGGREGATE));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, Log2_32_Ceil(LastVal+1)));
    AggregateAbbrev = Stream.EmitAbbrev(Abbv);

    // Abbrev for CST_CODE_STRING.
    Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_STRING));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8));
    String8Abbrev = Stream.EmitAbbrev(Abbv);
    // Abbrev for CST_CODE_CSTRING.
    Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_CSTRING));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 7));
    CString7Abbrev = Stream.EmitAbbrev(Abbv);
    // Abbrev for CST_CODE_CSTRING.
    Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_CSTRING));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Char6));
    CString6Abbrev = Stream.EmitAbbrev(Abbv);
  }  
  
  SmallVector<uint64_t, 64> Record;

  const ValueEnumerator::ValueList &Vals = VE.getValues();
  const Type *LastTy = 0;
  for (unsigned i = FirstVal; i != LastVal; ++i) {
    const Value *V = Vals[i].first;
    // If we need to switch types, do so now.
    if (V->getType() != LastTy) {
      LastTy = V->getType();
      Record.push_back(VE.getTypeID(LastTy));
      Stream.EmitRecord(bitc::CST_CODE_SETTYPE, Record,
                        CONSTANTS_SETTYPE_ABBREV);
      Record.clear();
    }
    
    if (const InlineAsm *IA = dyn_cast<InlineAsm>(V)) {
      Record.push_back(unsigned(IA->hasSideEffects()));
      
      // Add the asm string.
      const std::string &AsmStr = IA->getAsmString();
      Record.push_back(AsmStr.size());
      for (unsigned i = 0, e = AsmStr.size(); i != e; ++i)
        Record.push_back(AsmStr[i]);
      
      // Add the constraint string.
      const std::string &ConstraintStr = IA->getConstraintString();
      Record.push_back(ConstraintStr.size());
      for (unsigned i = 0, e = ConstraintStr.size(); i != e; ++i)
        Record.push_back(ConstraintStr[i]);
      Stream.EmitRecord(bitc::CST_CODE_INLINEASM, Record);
      Record.clear();
      continue;
    }
    const Constant *C = cast<Constant>(V);
    unsigned Code = -1U;
    unsigned AbbrevToUse = 0;
    if (C->isNullValue()) {
      Code = bitc::CST_CODE_NULL;
    } else if (isa<UndefValue>(C)) {
      Code = bitc::CST_CODE_UNDEF;
    } else if (const ConstantInt *IV = dyn_cast<ConstantInt>(C)) {
      if (IV->getBitWidth() <= 64) {
        int64_t V = IV->getSExtValue();
        if (V >= 0)
          Record.push_back(V << 1);
        else
          Record.push_back((-V << 1) | 1);
        Code = bitc::CST_CODE_INTEGER;
        AbbrevToUse = CONSTANTS_INTEGER_ABBREV;
      } else {                             // Wide integers, > 64 bits in size.
        // We have an arbitrary precision integer value to write whose 
        // bit width is > 64. However, in canonical unsigned integer 
        // format it is likely that the high bits are going to be zero.
        // So, we only write the number of active words.
        unsigned NWords = IV->getValue().getActiveWords(); 
        const uint64_t *RawWords = IV->getValue().getRawData();
        for (unsigned i = 0; i != NWords; ++i) {
          int64_t V = RawWords[i];
          if (V >= 0)
            Record.push_back(V << 1);
          else
            Record.push_back((-V << 1) | 1);
        }
        Code = bitc::CST_CODE_WIDE_INTEGER;
      }
    } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      Code = bitc::CST_CODE_FLOAT;
      if (CFP->getType() == Type::FloatTy) {
        Record.push_back(FloatToBits((float)CFP->getValue()));
      } else {
        assert (CFP->getType() == Type::DoubleTy && "Unknown FP type!");
        Record.push_back(DoubleToBits((double)CFP->getValue()));
      }
    } else if (isa<ConstantArray>(C) && cast<ConstantArray>(C)->isString()) {
      // Emit constant strings specially.
      unsigned NumOps = C->getNumOperands();
      // If this is a null-terminated string, use the denser CSTRING encoding.
      if (C->getOperand(NumOps-1)->isNullValue()) {
        Code = bitc::CST_CODE_CSTRING;
        --NumOps;  // Don't encode the null, which isn't allowed by char6.
      } else {
        Code = bitc::CST_CODE_STRING;
        AbbrevToUse = String8Abbrev;
      }
      bool isCStr7 = Code == bitc::CST_CODE_CSTRING;
      bool isCStrChar6 = Code == bitc::CST_CODE_CSTRING;
      for (unsigned i = 0; i != NumOps; ++i) {
        unsigned char V = cast<ConstantInt>(C->getOperand(i))->getZExtValue();
        Record.push_back(V);
        isCStr7 &= (V & 128) == 0;
        if (isCStrChar6) 
          isCStrChar6 = BitCodeAbbrevOp::isChar6(V);
      }
      
      if (isCStrChar6)
        AbbrevToUse = CString6Abbrev;
      else if (isCStr7)
        AbbrevToUse = CString7Abbrev;
    } else if (isa<ConstantArray>(C) || isa<ConstantStruct>(V) ||
               isa<ConstantVector>(V)) {
      Code = bitc::CST_CODE_AGGREGATE;
      for (unsigned i = 0, e = C->getNumOperands(); i != e; ++i)
        Record.push_back(VE.getValueID(C->getOperand(i)));
      AbbrevToUse = AggregateAbbrev;
    } else if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
      switch (CE->getOpcode()) {
      default:
        if (Instruction::isCast(CE->getOpcode())) {
          Code = bitc::CST_CODE_CE_CAST;
          Record.push_back(GetEncodedCastOpcode(CE->getOpcode()));
          Record.push_back(VE.getTypeID(C->getOperand(0)->getType()));
          Record.push_back(VE.getValueID(C->getOperand(0)));
          AbbrevToUse = CONSTANTS_CE_CAST_Abbrev;
        } else {
          assert(CE->getNumOperands() == 2 && "Unknown constant expr!");
          Code = bitc::CST_CODE_CE_BINOP;
          Record.push_back(GetEncodedBinaryOpcode(CE->getOpcode()));
          Record.push_back(VE.getValueID(C->getOperand(0)));
          Record.push_back(VE.getValueID(C->getOperand(1)));
        }
        break;
      case Instruction::GetElementPtr:
        Code = bitc::CST_CODE_CE_GEP;
        for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i) {
          Record.push_back(VE.getTypeID(C->getOperand(i)->getType()));
          Record.push_back(VE.getValueID(C->getOperand(i)));
        }
        break;
      case Instruction::Select:
        Code = bitc::CST_CODE_CE_SELECT;
        Record.push_back(VE.getValueID(C->getOperand(0)));
        Record.push_back(VE.getValueID(C->getOperand(1)));
        Record.push_back(VE.getValueID(C->getOperand(2)));
        break;
      case Instruction::ExtractElement:
        Code = bitc::CST_CODE_CE_EXTRACTELT;
        Record.push_back(VE.getTypeID(C->getOperand(0)->getType()));
        Record.push_back(VE.getValueID(C->getOperand(0)));
        Record.push_back(VE.getValueID(C->getOperand(1)));
        break;
      case Instruction::InsertElement:
        Code = bitc::CST_CODE_CE_INSERTELT;
        Record.push_back(VE.getValueID(C->getOperand(0)));
        Record.push_back(VE.getValueID(C->getOperand(1)));
        Record.push_back(VE.getValueID(C->getOperand(2)));
        break;
      case Instruction::ShuffleVector:
        Code = bitc::CST_CODE_CE_SHUFFLEVEC;
        Record.push_back(VE.getValueID(C->getOperand(0)));
        Record.push_back(VE.getValueID(C->getOperand(1)));
        Record.push_back(VE.getValueID(C->getOperand(2)));
        break;
      case Instruction::ICmp:
      case Instruction::FCmp:
        Code = bitc::CST_CODE_CE_CMP;
        Record.push_back(VE.getTypeID(C->getOperand(0)->getType()));
        Record.push_back(VE.getValueID(C->getOperand(0)));
        Record.push_back(VE.getValueID(C->getOperand(1)));
        Record.push_back(CE->getPredicate());
        break;
      }
    } else {
      assert(0 && "Unknown constant!");
    }
    Stream.EmitRecord(Code, Record, AbbrevToUse);
    Record.clear();
  }

  Stream.ExitBlock();
}

static void WriteModuleConstants(const ValueEnumerator &VE,
                                 BitstreamWriter &Stream) {
  const ValueEnumerator::ValueList &Vals = VE.getValues();
  
  // Find the first constant to emit, which is the first non-globalvalue value.
  // We know globalvalues have been emitted by WriteModuleInfo.
  for (unsigned i = 0, e = Vals.size(); i != e; ++i) {
    if (!isa<GlobalValue>(Vals[i].first)) {
      WriteConstants(i, Vals.size(), VE, Stream, true);
      return;
    }
  }
}

/// PushValueAndType - The file has to encode both the value and type id for
/// many values, because we need to know what type to create for forward
/// references.  However, most operands are not forward references, so this type
/// field is not needed.
///
/// This function adds V's value ID to Vals.  If the value ID is higher than the
/// instruction ID, then it is a forward reference, and it also includes the
/// type ID.
static bool PushValueAndType(Value *V, unsigned InstID,
                             SmallVector<unsigned, 64> &Vals, 
                             ValueEnumerator &VE) {
  unsigned ValID = VE.getValueID(V);
  Vals.push_back(ValID);
  if (ValID >= InstID) {
    Vals.push_back(VE.getTypeID(V->getType()));
    return true;
  }
  return false;
}

/// WriteInstruction - Emit an instruction to the specified stream.
static void WriteInstruction(const Instruction &I, unsigned InstID,
                             ValueEnumerator &VE, BitstreamWriter &Stream,
                             SmallVector<unsigned, 64> &Vals) {
  unsigned Code = 0;
  unsigned AbbrevToUse = 0;
  switch (I.getOpcode()) {
  default:
    if (Instruction::isCast(I.getOpcode())) {
      Code = bitc::FUNC_CODE_INST_CAST;
      if (!PushValueAndType(I.getOperand(0), InstID, Vals, VE))
        AbbrevToUse = FUNCTION_INST_CAST_ABBREV;
      Vals.push_back(VE.getTypeID(I.getType()));
      Vals.push_back(GetEncodedCastOpcode(I.getOpcode()));
    } else {
      assert(isa<BinaryOperator>(I) && "Unknown instruction!");
      Code = bitc::FUNC_CODE_INST_BINOP;
      if (!PushValueAndType(I.getOperand(0), InstID, Vals, VE))
        AbbrevToUse = FUNCTION_INST_BINOP_ABBREV;
      Vals.push_back(VE.getValueID(I.getOperand(1)));
      Vals.push_back(GetEncodedBinaryOpcode(I.getOpcode()));
    }
    break;

  case Instruction::GetElementPtr:
    Code = bitc::FUNC_CODE_INST_GEP;
    for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
      PushValueAndType(I.getOperand(i), InstID, Vals, VE);
    break;
  case Instruction::Select:
    Code = bitc::FUNC_CODE_INST_SELECT;
    PushValueAndType(I.getOperand(1), InstID, Vals, VE);
    Vals.push_back(VE.getValueID(I.getOperand(2)));
    Vals.push_back(VE.getValueID(I.getOperand(0)));
    break;
  case Instruction::ExtractElement:
    Code = bitc::FUNC_CODE_INST_EXTRACTELT;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    Vals.push_back(VE.getValueID(I.getOperand(1)));
    break;
  case Instruction::InsertElement:
    Code = bitc::FUNC_CODE_INST_INSERTELT;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    Vals.push_back(VE.getValueID(I.getOperand(1)));
    Vals.push_back(VE.getValueID(I.getOperand(2)));
    break;
  case Instruction::ShuffleVector:
    Code = bitc::FUNC_CODE_INST_SHUFFLEVEC;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    Vals.push_back(VE.getValueID(I.getOperand(1)));
    Vals.push_back(VE.getValueID(I.getOperand(2)));
    break;
  case Instruction::ICmp:
  case Instruction::FCmp:
    Code = bitc::FUNC_CODE_INST_CMP;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    Vals.push_back(VE.getValueID(I.getOperand(1)));
    Vals.push_back(cast<CmpInst>(I).getPredicate());
    break;

  case Instruction::Ret:
    Code = bitc::FUNC_CODE_INST_RET;
    if (!I.getNumOperands())
      AbbrevToUse = FUNCTION_INST_RET_VOID_ABBREV;
    else if (!PushValueAndType(I.getOperand(0), InstID, Vals, VE))
      AbbrevToUse = FUNCTION_INST_RET_VAL_ABBREV;
    break;
  case Instruction::Br:
    Code = bitc::FUNC_CODE_INST_BR;
    Vals.push_back(VE.getValueID(I.getOperand(0)));
    if (cast<BranchInst>(I).isConditional()) {
      Vals.push_back(VE.getValueID(I.getOperand(1)));
      Vals.push_back(VE.getValueID(I.getOperand(2)));
    }
    break;
  case Instruction::Switch:
    Code = bitc::FUNC_CODE_INST_SWITCH;
    Vals.push_back(VE.getTypeID(I.getOperand(0)->getType()));
    for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
      Vals.push_back(VE.getValueID(I.getOperand(i)));
    break;
  case Instruction::Invoke: {
    const PointerType *PTy = cast<PointerType>(I.getOperand(0)->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    Code = bitc::FUNC_CODE_INST_INVOKE;
    
    // Note: we emit the param attr ID number for the function type of this
    // function.  In the future, we intend for attrs to be properties of
    // functions, instead of on the type.  This is to support this future work.
    Vals.push_back(VE.getParamAttrID(FTy->getParamAttrs()));
    
    Vals.push_back(cast<InvokeInst>(I).getCallingConv());
    Vals.push_back(VE.getValueID(I.getOperand(1)));      // normal dest
    Vals.push_back(VE.getValueID(I.getOperand(2)));      // unwind dest
    PushValueAndType(I.getOperand(0), InstID, Vals, VE); // callee
    
    // Emit value #'s for the fixed parameters.
    for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
      Vals.push_back(VE.getValueID(I.getOperand(i+3)));  // fixed param.

    // Emit type/value pairs for varargs params.
    if (FTy->isVarArg()) {
      for (unsigned i = 3+FTy->getNumParams(), e = I.getNumOperands();
           i != e; ++i)
        PushValueAndType(I.getOperand(i), InstID, Vals, VE); // vararg
    }
    break;
  }
  case Instruction::Unwind:
    Code = bitc::FUNC_CODE_INST_UNWIND;
    break;
  case Instruction::Unreachable:
    Code = bitc::FUNC_CODE_INST_UNREACHABLE;
    AbbrevToUse = FUNCTION_INST_UNREACHABLE_ABBREV;
    break;
  
  case Instruction::PHI:
    Code = bitc::FUNC_CODE_INST_PHI;
    Vals.push_back(VE.getTypeID(I.getType()));
    for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
      Vals.push_back(VE.getValueID(I.getOperand(i)));
    break;
    
  case Instruction::Malloc:
    Code = bitc::FUNC_CODE_INST_MALLOC;
    Vals.push_back(VE.getTypeID(I.getType()));
    Vals.push_back(VE.getValueID(I.getOperand(0))); // size.
    Vals.push_back(Log2_32(cast<MallocInst>(I).getAlignment())+1);
    break;
    
  case Instruction::Free:
    Code = bitc::FUNC_CODE_INST_FREE;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    break;
    
  case Instruction::Alloca:
    Code = bitc::FUNC_CODE_INST_ALLOCA;
    Vals.push_back(VE.getTypeID(I.getType()));
    Vals.push_back(VE.getValueID(I.getOperand(0))); // size.
    Vals.push_back(Log2_32(cast<AllocaInst>(I).getAlignment())+1);
    break;
    
  case Instruction::Load:
    Code = bitc::FUNC_CODE_INST_LOAD;
    if (!PushValueAndType(I.getOperand(0), InstID, Vals, VE))  // ptr
      AbbrevToUse = FUNCTION_INST_LOAD_ABBREV;
      
    Vals.push_back(Log2_32(cast<LoadInst>(I).getAlignment())+1);
    Vals.push_back(cast<LoadInst>(I).isVolatile());
    break;
  case Instruction::Store:
    Code = bitc::FUNC_CODE_INST_STORE;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);  // val.
    Vals.push_back(VE.getValueID(I.getOperand(1)));       // ptr.
    Vals.push_back(Log2_32(cast<StoreInst>(I).getAlignment())+1);
    Vals.push_back(cast<StoreInst>(I).isVolatile());
    break;
  case Instruction::Call: {
    const PointerType *PTy = cast<PointerType>(I.getOperand(0)->getType());
    const FunctionType *FTy = cast<FunctionType>(PTy->getElementType());

    Code = bitc::FUNC_CODE_INST_CALL;
    
    // Note: we emit the param attr ID number for the function type of this
    // function.  In the future, we intend for attrs to be properties of
    // functions, instead of on the type.  This is to support this future work.
    Vals.push_back(VE.getParamAttrID(FTy->getParamAttrs()));
    
    Vals.push_back((cast<CallInst>(I).getCallingConv() << 1) |
                   unsigned(cast<CallInst>(I).isTailCall()));
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);  // Callee
    
    // Emit value #'s for the fixed parameters.
    for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
      Vals.push_back(VE.getValueID(I.getOperand(i+1)));  // fixed param.
      
    // Emit type/value pairs for varargs params.
    if (FTy->isVarArg()) {
      unsigned NumVarargs = I.getNumOperands()-1-FTy->getNumParams();
      for (unsigned i = I.getNumOperands()-NumVarargs, e = I.getNumOperands();
           i != e; ++i)
        PushValueAndType(I.getOperand(i), InstID, Vals, VE);  // varargs
    }
    break;
  }
  case Instruction::VAArg:
    Code = bitc::FUNC_CODE_INST_VAARG;
    Vals.push_back(VE.getTypeID(I.getOperand(0)->getType()));   // valistty
    Vals.push_back(VE.getValueID(I.getOperand(0))); // valist.
    Vals.push_back(VE.getTypeID(I.getType())); // restype.
    break;
  }
  
  Stream.EmitRecord(Code, Vals, AbbrevToUse);
  Vals.clear();
}

// Emit names for globals/functions etc.
static void WriteValueSymbolTable(const ValueSymbolTable &VST,
                                  const ValueEnumerator &VE,
                                  BitstreamWriter &Stream) {
  if (VST.empty()) return;
  Stream.EnterSubblock(bitc::VALUE_SYMTAB_BLOCK_ID, 4);

  // FIXME: Set up the abbrev, we know how many values there are!
  // FIXME: We know if the type names can use 7-bit ascii.
  SmallVector<unsigned, 64> NameVals;
  
  for (ValueSymbolTable::const_iterator SI = VST.begin(), SE = VST.end();
       SI != SE; ++SI) {
    
    const ValueName &Name = *SI;
    
    // Figure out the encoding to use for the name.
    bool is7Bit = true;
    bool isChar6 = true;
    for (const char *C = Name.getKeyData(), *E = C+Name.getKeyLength();
         C != E; ++C) {
      if (isChar6) 
        isChar6 = BitCodeAbbrevOp::isChar6(*C);
      if ((unsigned char)*C & 128) {
        is7Bit = false;
        break;  // don't bother scanning the rest.
      }
    }
    
    unsigned AbbrevToUse = VST_ENTRY_8_ABBREV;
    
    // VST_ENTRY:   [valueid, namechar x N]
    // VST_BBENTRY: [bbid, namechar x N]
    unsigned Code;
    if (isa<BasicBlock>(SI->getValue())) {
      Code = bitc::VST_CODE_BBENTRY;
      if (isChar6)
        AbbrevToUse = VST_BBENTRY_6_ABBREV;
    } else {
      Code = bitc::VST_CODE_ENTRY;
      if (isChar6)
        AbbrevToUse = VST_ENTRY_6_ABBREV;
      else if (is7Bit)
        AbbrevToUse = VST_ENTRY_7_ABBREV;
    }
    
    NameVals.push_back(VE.getValueID(SI->getValue()));
    for (const char *P = Name.getKeyData(),
         *E = Name.getKeyData()+Name.getKeyLength(); P != E; ++P)
      NameVals.push_back((unsigned char)*P);
    
    // Emit the finished record.
    Stream.EmitRecord(Code, NameVals, AbbrevToUse);
    NameVals.clear();
  }
  Stream.ExitBlock();
}

/// WriteFunction - Emit a function body to the module stream.
static void WriteFunction(const Function &F, ValueEnumerator &VE, 
                          BitstreamWriter &Stream) {
  Stream.EnterSubblock(bitc::FUNCTION_BLOCK_ID, 4);
  VE.incorporateFunction(F);

  SmallVector<unsigned, 64> Vals;
  
  // Emit the number of basic blocks, so the reader can create them ahead of
  // time.
  Vals.push_back(VE.getBasicBlocks().size());
  Stream.EmitRecord(bitc::FUNC_CODE_DECLAREBLOCKS, Vals);
  Vals.clear();
  
  // If there are function-local constants, emit them now.
  unsigned CstStart, CstEnd;
  VE.getFunctionConstantRange(CstStart, CstEnd);
  WriteConstants(CstStart, CstEnd, VE, Stream, false);
  
  // Keep a running idea of what the instruction ID is. 
  unsigned InstID = CstEnd;
  
  // Finally, emit all the instructions, in order.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      WriteInstruction(*I, InstID, VE, Stream, Vals);
      if (I->getType() != Type::VoidTy)
        ++InstID;
    }
  
  // Emit names for all the instructions etc.
  WriteValueSymbolTable(F.getValueSymbolTable(), VE, Stream);
    
  VE.purgeFunction();
  Stream.ExitBlock();
}

/// WriteTypeSymbolTable - Emit a block for the specified type symtab.
static void WriteTypeSymbolTable(const TypeSymbolTable &TST,
                                 const ValueEnumerator &VE,
                                 BitstreamWriter &Stream) {
  if (TST.empty()) return;
  
  Stream.EnterSubblock(bitc::TYPE_SYMTAB_BLOCK_ID, 3);
  
  // 7-bit fixed width VST_CODE_ENTRY strings.
  BitCodeAbbrev *Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_ENTRY));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                            Log2_32_Ceil(VE.getTypes().size()+1)));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 7));
  unsigned V7Abbrev = Stream.EmitAbbrev(Abbv);
  
  SmallVector<unsigned, 64> NameVals;
  
  for (TypeSymbolTable::const_iterator TI = TST.begin(), TE = TST.end(); 
       TI != TE; ++TI) {
    // TST_ENTRY: [typeid, namechar x N]
    NameVals.push_back(VE.getTypeID(TI->second));
    
    const std::string &Str = TI->first;
    bool is7Bit = true;
    for (unsigned i = 0, e = Str.size(); i != e; ++i) {
      NameVals.push_back((unsigned char)Str[i]);
      if (Str[i] & 128)
        is7Bit = false;
    }
    
    // Emit the finished record.
    Stream.EmitRecord(bitc::VST_CODE_ENTRY, NameVals, is7Bit ? V7Abbrev : 0);
    NameVals.clear();
  }
  
  Stream.ExitBlock();
}

// Emit blockinfo, which defines the standard abbreviations etc.
static void WriteBlockInfo(const ValueEnumerator &VE, BitstreamWriter &Stream) {
  // We only want to emit block info records for blocks that have multiple
  // instances: CONSTANTS_BLOCK, FUNCTION_BLOCK and VALUE_SYMTAB_BLOCK.  Other
  // blocks can defined their abbrevs inline.
  Stream.EnterBlockInfoBlock(2);
  
  { // 8-bit fixed-width VST_ENTRY/VST_BBENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID, 
                                   Abbv) != VST_ENTRY_8_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  { // 7-bit fixed width VST_ENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_ENTRY));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 7));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_ENTRY_7_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // 6-bit char6 VST_ENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_ENTRY));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Char6));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_ENTRY_6_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // 6-bit char6 VST_BBENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_BBENTRY));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Char6));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_BBENTRY_6_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  
  
  { // SETTYPE abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_SETTYPE));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                              Log2_32_Ceil(VE.getTypes().size()+1)));
    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_SETTYPE_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  { // INTEGER abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_INTEGER));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_INTEGER_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  { // CE_CAST abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_CE_CAST));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 4));  // cast opc
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,       // typeid
                              Log2_32_Ceil(VE.getTypes().size()+1)));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));    // value id

    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_CE_CAST_Abbrev)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // NULL abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_NULL));
    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_NULL_Abbrev)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  // FIXME: This should only use space for first class types!
 
  { // INST_LOAD abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_LOAD));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Ptr
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 4)); // Align
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // volatile
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_LOAD_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // INST_BINOP abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_BINOP));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // LHS
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // RHS
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 4)); // opc
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_BINOP_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // INST_CAST abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_CAST));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6));    // OpVal
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,       // dest ty
                              Log2_32_Ceil(VE.getTypes().size()+1)));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 4));  // opc
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_CAST_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  { // INST_RET abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_RET));
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_RET_VOID_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // INST_RET abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_RET));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // ValID
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_RET_VAL_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  { // INST_UNREACHABLE abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_UNREACHABLE));
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_UNREACHABLE_ABBREV)
      assert(0 && "Unexpected abbrev ordering!");
  }
  
  Stream.ExitBlock();
}


/// WriteModule - Emit the specified module to the bitstream.
static void WriteModule(const Module *M, BitstreamWriter &Stream) {
  Stream.EnterSubblock(bitc::MODULE_BLOCK_ID, 3);
  
  // Emit the version number if it is non-zero.
  if (CurVersion) {
    SmallVector<unsigned, 1> Vals;
    Vals.push_back(CurVersion);
    Stream.EmitRecord(bitc::MODULE_CODE_VERSION, Vals);
  }
  
  // Analyze the module, enumerating globals, functions, etc.
  ValueEnumerator VE(M);

  // Emit blockinfo, which defines the standard abbreviations etc.
  WriteBlockInfo(VE, Stream);
  
  // Emit information about parameter attributes.
  WriteParamAttrTable(VE, Stream);
  
  // Emit information describing all of the types in the module.
  WriteTypeTable(VE, Stream);
  
  // Emit top-level description of module, including target triple, inline asm,
  // descriptors for global variables, and function prototype info.
  WriteModuleInfo(M, VE, Stream);
  
  // Emit constants.
  WriteModuleConstants(VE, Stream);
  
  // If we have any aggregate values in the value table, purge them - these can
  // only be used to initialize global variables.  Doing so makes the value
  // namespace smaller for code in functions.
  int NumNonAggregates = VE.PurgeAggregateValues();
  if (NumNonAggregates != -1) {
    SmallVector<unsigned, 1> Vals;
    Vals.push_back(NumNonAggregates);
    Stream.EmitRecord(bitc::MODULE_CODE_PURGEVALS, Vals);
  }
  
  // Emit function bodies.
  for (Module::const_iterator I = M->begin(), E = M->end(); I != E; ++I)
    if (!I->isDeclaration())
      WriteFunction(*I, VE, Stream);
  
  // Emit the type symbol table information.
  WriteTypeSymbolTable(M->getTypeSymbolTable(), VE, Stream);
  
  // Emit names for globals/functions etc.
  WriteValueSymbolTable(M->getValueSymbolTable(), VE, Stream);
  
  Stream.ExitBlock();
}


/// WriteBitcodeToFile - Write the specified module to the specified output
/// stream.
void llvm::WriteBitcodeToFile(const Module *M, std::ostream &Out) {
  std::vector<unsigned char> Buffer;
  BitstreamWriter Stream(Buffer);
  
  Buffer.reserve(256*1024);
  
  // Emit the file header.
  Stream.Emit((unsigned)'B', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit(0x0, 4);
  Stream.Emit(0xC, 4);
  Stream.Emit(0xE, 4);
  Stream.Emit(0xD, 4);

  // Emit the module.
  WriteModule(M, Stream);
  
  // Write the generated bitstream to "Out".
  Out.write((char*)&Buffer.front(), Buffer.size());
  
  // Make sure it hits disk now.
  Out.flush();
}
