//===--- Bitcode/Writer/BitcodeWriter.cpp - Bitcode Writer ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "llvm/Operator.h"
#include "llvm/ValueSymbolTable.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Program.h"
#include <cctype>
#include <map>
using namespace llvm;

static cl::opt<bool>
EnablePreserveUseListOrdering("enable-bc-uselist-preserve",
                              cl::desc("Turn on experimental support for "
                                       "use-list order preservation."),
                              cl::init(false), cl::Hidden);

/// These are manifest constants used by the bitcode writer. They do not need to
/// be kept in sync with the reader, but need to be consistent within this file.
enum {
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
  FUNCTION_INST_BINOP_FLAGS_ABBREV,
  FUNCTION_INST_CAST_ABBREV,
  FUNCTION_INST_RET_VOID_ABBREV,
  FUNCTION_INST_RET_VAL_ABBREV,
  FUNCTION_INST_UNREACHABLE_ABBREV,
  
  // SwitchInst Magic
  SWITCH_INST_MAGIC = 0x4B5 // May 2012 => 1205 => Hex
};

static unsigned GetEncodedCastOpcode(unsigned Opcode) {
  switch (Opcode) {
  default: llvm_unreachable("Unknown cast instruction!");
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
  default: llvm_unreachable("Unknown binary instruction!");
  case Instruction::Add:
  case Instruction::FAdd: return bitc::BINOP_ADD;
  case Instruction::Sub:
  case Instruction::FSub: return bitc::BINOP_SUB;
  case Instruction::Mul:
  case Instruction::FMul: return bitc::BINOP_MUL;
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

static unsigned GetEncodedRMWOperation(AtomicRMWInst::BinOp Op) {
  switch (Op) {
  default: llvm_unreachable("Unknown RMW operation!");
  case AtomicRMWInst::Xchg: return bitc::RMW_XCHG;
  case AtomicRMWInst::Add: return bitc::RMW_ADD;
  case AtomicRMWInst::Sub: return bitc::RMW_SUB;
  case AtomicRMWInst::And: return bitc::RMW_AND;
  case AtomicRMWInst::Nand: return bitc::RMW_NAND;
  case AtomicRMWInst::Or: return bitc::RMW_OR;
  case AtomicRMWInst::Xor: return bitc::RMW_XOR;
  case AtomicRMWInst::Max: return bitc::RMW_MAX;
  case AtomicRMWInst::Min: return bitc::RMW_MIN;
  case AtomicRMWInst::UMax: return bitc::RMW_UMAX;
  case AtomicRMWInst::UMin: return bitc::RMW_UMIN;
  }
}

static unsigned GetEncodedOrdering(AtomicOrdering Ordering) {
  switch (Ordering) {
  case NotAtomic: return bitc::ORDERING_NOTATOMIC;
  case Unordered: return bitc::ORDERING_UNORDERED;
  case Monotonic: return bitc::ORDERING_MONOTONIC;
  case Acquire: return bitc::ORDERING_ACQUIRE;
  case Release: return bitc::ORDERING_RELEASE;
  case AcquireRelease: return bitc::ORDERING_ACQREL;
  case SequentiallyConsistent: return bitc::ORDERING_SEQCST;
  }
  llvm_unreachable("Invalid ordering");
}

static unsigned GetEncodedSynchScope(SynchronizationScope SynchScope) {
  switch (SynchScope) {
  case SingleThread: return bitc::SYNCHSCOPE_SINGLETHREAD;
  case CrossThread: return bitc::SYNCHSCOPE_CROSSTHREAD;
  }
  llvm_unreachable("Invalid synch scope");
}

static void WriteStringRecord(unsigned Code, StringRef Str,
                              unsigned AbbrevToUse, BitstreamWriter &Stream) {
  SmallVector<unsigned, 64> Vals;

  // Code: [strchar x N]
  for (unsigned i = 0, e = Str.size(); i != e; ++i) {
    if (AbbrevToUse && !BitCodeAbbrevOp::isChar6(Str[i]))
      AbbrevToUse = 0;
    Vals.push_back(Str[i]);
  }

  // Emit the finished record.
  Stream.EmitRecord(Code, Vals, AbbrevToUse);
}

// Emit information about parameter attributes.
static void WriteAttributeTable(const ValueEnumerator &VE,
                                BitstreamWriter &Stream) {
  const std::vector<AttrListPtr> &Attrs = VE.getAttributes();
  if (Attrs.empty()) return;

  Stream.EnterSubblock(bitc::PARAMATTR_BLOCK_ID, 3);

  SmallVector<uint64_t, 64> Record;
  for (unsigned i = 0, e = Attrs.size(); i != e; ++i) {
    const AttrListPtr &A = Attrs[i];
    for (unsigned i = 0, e = A.getNumSlots(); i != e; ++i) {
      const AttributeWithIndex &PAWI = A.getSlot(i);
      Record.push_back(PAWI.Index);
      Record.push_back(Attributes::encodeLLVMAttributesForBitcode(PAWI.Attrs));
    }

    Stream.EmitRecord(bitc::PARAMATTR_CODE_ENTRY, Record);
    Record.clear();
  }

  Stream.ExitBlock();
}

/// WriteTypeTable - Write out the type table for a module.
static void WriteTypeTable(const ValueEnumerator &VE, BitstreamWriter &Stream) {
  const ValueEnumerator::TypeList &TypeList = VE.getTypes();

  Stream.EnterSubblock(bitc::TYPE_BLOCK_ID_NEW, 4 /*count from # abbrevs */);
  SmallVector<uint64_t, 64> TypeVals;

  uint64_t NumBits = Log2_32_Ceil(VE.getTypes().size()+1);

  // Abbrev for TYPE_CODE_POINTER.
  BitCodeAbbrev *Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_POINTER));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, NumBits));
  Abbv->Add(BitCodeAbbrevOp(0));  // Addrspace = 0
  unsigned PtrAbbrev = Stream.EmitAbbrev(Abbv);

  // Abbrev for TYPE_CODE_FUNCTION.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_FUNCTION));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1));  // isvararg
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, NumBits));

  unsigned FunctionAbbrev = Stream.EmitAbbrev(Abbv);

  // Abbrev for TYPE_CODE_STRUCT_ANON.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_STRUCT_ANON));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1));  // ispacked
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, NumBits));

  unsigned StructAnonAbbrev = Stream.EmitAbbrev(Abbv);

  // Abbrev for TYPE_CODE_STRUCT_NAME.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_STRUCT_NAME));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Char6));
  unsigned StructNameAbbrev = Stream.EmitAbbrev(Abbv);

  // Abbrev for TYPE_CODE_STRUCT_NAMED.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_STRUCT_NAMED));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1));  // ispacked
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, NumBits));

  unsigned StructNamedAbbrev = Stream.EmitAbbrev(Abbv);
  
  // Abbrev for TYPE_CODE_ARRAY.
  Abbv = new BitCodeAbbrev();
  Abbv->Add(BitCodeAbbrevOp(bitc::TYPE_CODE_ARRAY));
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));   // size
  Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, NumBits));

  unsigned ArrayAbbrev = Stream.EmitAbbrev(Abbv);

  // Emit an entry count so the reader can reserve space.
  TypeVals.push_back(TypeList.size());
  Stream.EmitRecord(bitc::TYPE_CODE_NUMENTRY, TypeVals);
  TypeVals.clear();

  // Loop over all of the types, emitting each in turn.
  for (unsigned i = 0, e = TypeList.size(); i != e; ++i) {
    Type *T = TypeList[i];
    int AbbrevToUse = 0;
    unsigned Code = 0;

    switch (T->getTypeID()) {
    default: llvm_unreachable("Unknown type!");
    case Type::VoidTyID:      Code = bitc::TYPE_CODE_VOID;   break;
    case Type::HalfTyID:      Code = bitc::TYPE_CODE_HALF;   break;
    case Type::FloatTyID:     Code = bitc::TYPE_CODE_FLOAT;  break;
    case Type::DoubleTyID:    Code = bitc::TYPE_CODE_DOUBLE; break;
    case Type::X86_FP80TyID:  Code = bitc::TYPE_CODE_X86_FP80; break;
    case Type::FP128TyID:     Code = bitc::TYPE_CODE_FP128; break;
    case Type::PPC_FP128TyID: Code = bitc::TYPE_CODE_PPC_FP128; break;
    case Type::LabelTyID:     Code = bitc::TYPE_CODE_LABEL;  break;
    case Type::MetadataTyID:  Code = bitc::TYPE_CODE_METADATA; break;
    case Type::X86_MMXTyID:   Code = bitc::TYPE_CODE_X86_MMX; break;
    case Type::IntegerTyID:
      // INTEGER: [width]
      Code = bitc::TYPE_CODE_INTEGER;
      TypeVals.push_back(cast<IntegerType>(T)->getBitWidth());
      break;
    case Type::PointerTyID: {
      PointerType *PTy = cast<PointerType>(T);
      // POINTER: [pointee type, address space]
      Code = bitc::TYPE_CODE_POINTER;
      TypeVals.push_back(VE.getTypeID(PTy->getElementType()));
      unsigned AddressSpace = PTy->getAddressSpace();
      TypeVals.push_back(AddressSpace);
      if (AddressSpace == 0) AbbrevToUse = PtrAbbrev;
      break;
    }
    case Type::FunctionTyID: {
      FunctionType *FT = cast<FunctionType>(T);
      // FUNCTION: [isvararg, retty, paramty x N]
      Code = bitc::TYPE_CODE_FUNCTION;
      TypeVals.push_back(FT->isVarArg());
      TypeVals.push_back(VE.getTypeID(FT->getReturnType()));
      for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i)
        TypeVals.push_back(VE.getTypeID(FT->getParamType(i)));
      AbbrevToUse = FunctionAbbrev;
      break;
    }
    case Type::StructTyID: {
      StructType *ST = cast<StructType>(T);
      // STRUCT: [ispacked, eltty x N]
      TypeVals.push_back(ST->isPacked());
      // Output all of the element types.
      for (StructType::element_iterator I = ST->element_begin(),
           E = ST->element_end(); I != E; ++I)
        TypeVals.push_back(VE.getTypeID(*I));
      
      if (ST->isLiteral()) {
        Code = bitc::TYPE_CODE_STRUCT_ANON;
        AbbrevToUse = StructAnonAbbrev;
      } else {
        if (ST->isOpaque()) {
          Code = bitc::TYPE_CODE_OPAQUE;
        } else {
          Code = bitc::TYPE_CODE_STRUCT_NAMED;
          AbbrevToUse = StructNamedAbbrev;
        }

        // Emit the name if it is present.
        if (!ST->getName().empty())
          WriteStringRecord(bitc::TYPE_CODE_STRUCT_NAME, ST->getName(),
                            StructNameAbbrev, Stream);
      }
      break;
    }
    case Type::ArrayTyID: {
      ArrayType *AT = cast<ArrayType>(T);
      // ARRAY: [numelts, eltty]
      Code = bitc::TYPE_CODE_ARRAY;
      TypeVals.push_back(AT->getNumElements());
      TypeVals.push_back(VE.getTypeID(AT->getElementType()));
      AbbrevToUse = ArrayAbbrev;
      break;
    }
    case Type::VectorTyID: {
      VectorType *VT = cast<VectorType>(T);
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
  case GlobalValue::ExternalLinkage:                 return 0;
  case GlobalValue::WeakAnyLinkage:                  return 1;
  case GlobalValue::AppendingLinkage:                return 2;
  case GlobalValue::InternalLinkage:                 return 3;
  case GlobalValue::LinkOnceAnyLinkage:              return 4;
  case GlobalValue::DLLImportLinkage:                return 5;
  case GlobalValue::DLLExportLinkage:                return 6;
  case GlobalValue::ExternalWeakLinkage:             return 7;
  case GlobalValue::CommonLinkage:                   return 8;
  case GlobalValue::PrivateLinkage:                  return 9;
  case GlobalValue::WeakODRLinkage:                  return 10;
  case GlobalValue::LinkOnceODRLinkage:              return 11;
  case GlobalValue::AvailableExternallyLinkage:      return 12;
  case GlobalValue::LinkerPrivateLinkage:            return 13;
  case GlobalValue::LinkerPrivateWeakLinkage:        return 14;
  case GlobalValue::LinkOnceODRAutoHideLinkage:      return 15;
  }
  llvm_unreachable("Invalid linkage");
}

static unsigned getEncodedVisibility(const GlobalValue *GV) {
  switch (GV->getVisibility()) {
  case GlobalValue::DefaultVisibility:   return 0;
  case GlobalValue::HiddenVisibility:    return 1;
  case GlobalValue::ProtectedVisibility: return 2;
  }
  llvm_unreachable("Invalid visibility");
}

static unsigned getEncodedThreadLocalMode(const GlobalVariable *GV) {
  switch (GV->getThreadLocalMode()) {
    case GlobalVariable::NotThreadLocal:         return 0;
    case GlobalVariable::GeneralDynamicTLSModel: return 1;
    case GlobalVariable::LocalDynamicTLSModel:   return 2;
    case GlobalVariable::InitialExecTLSModel:    return 3;
    case GlobalVariable::LocalExecTLSModel:      return 4;
  }
  llvm_unreachable("Invalid TLS model");
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

  // Emit information about sections and GC, computing how many there are. Also
  // compute the maximum alignment value.
  std::map<std::string, unsigned> SectionMap;
  std::map<std::string, unsigned> GCMap;
  unsigned MaxAlignment = 0;
  unsigned MaxGlobalType = 0;
  for (Module::const_global_iterator GV = M->global_begin(),E = M->global_end();
       GV != E; ++GV) {
    MaxAlignment = std::max(MaxAlignment, GV->getAlignment());
    MaxGlobalType = std::max(MaxGlobalType, VE.getTypeID(GV->getType()));
    if (GV->hasSection()) {
      // Give section names unique ID's.
      unsigned &Entry = SectionMap[GV->getSection()];
      if (!Entry) {
        WriteStringRecord(bitc::MODULE_CODE_SECTIONNAME, GV->getSection(),
                          0/*TODO*/, Stream);
        Entry = SectionMap.size();
      }
    }
  }
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    MaxAlignment = std::max(MaxAlignment, F->getAlignment());
    if (F->hasSection()) {
      // Give section names unique ID's.
      unsigned &Entry = SectionMap[F->getSection()];
      if (!Entry) {
        WriteStringRecord(bitc::MODULE_CODE_SECTIONNAME, F->getSection(),
                          0/*TODO*/, Stream);
        Entry = SectionMap.size();
      }
    }
    if (F->hasGC()) {
      // Same for GC names.
      unsigned &Entry = GCMap[F->getGC()];
      if (!Entry) {
        WriteStringRecord(bitc::MODULE_CODE_GCNAME, F->getGC(),
                          0/*TODO*/, Stream);
        Entry = GCMap.size();
      }
    }
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
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 4));      // Linkage.
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
    //             linkage, alignment, section, visibility, threadlocal,
    //             unnamed_addr]
    Vals.push_back(VE.getTypeID(GV->getType()));
    Vals.push_back(GV->isConstant());
    Vals.push_back(GV->isDeclaration() ? 0 :
                   (VE.getValueID(GV->getInitializer()) + 1));
    Vals.push_back(getEncodedLinkage(GV));
    Vals.push_back(Log2_32(GV->getAlignment())+1);
    Vals.push_back(GV->hasSection() ? SectionMap[GV->getSection()] : 0);
    if (GV->isThreadLocal() ||
        GV->getVisibility() != GlobalValue::DefaultVisibility ||
        GV->hasUnnamedAddr()) {
      Vals.push_back(getEncodedVisibility(GV));
      Vals.push_back(getEncodedThreadLocalMode(GV));
      Vals.push_back(GV->hasUnnamedAddr());
    } else {
      AbbrevToUse = SimpleGVarAbbrev;
    }

    Stream.EmitRecord(bitc::MODULE_CODE_GLOBALVAR, Vals, AbbrevToUse);
    Vals.clear();
  }

  // Emit the function proto information.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F) {
    // FUNCTION:  [type, callingconv, isproto, linkage, paramattrs, alignment,
    //             section, visibility, gc, unnamed_addr]
    Vals.push_back(VE.getTypeID(F->getType()));
    Vals.push_back(F->getCallingConv());
    Vals.push_back(F->isDeclaration());
    Vals.push_back(getEncodedLinkage(F));
    Vals.push_back(VE.getAttributeID(F->getAttributes()));
    Vals.push_back(Log2_32(F->getAlignment())+1);
    Vals.push_back(F->hasSection() ? SectionMap[F->getSection()] : 0);
    Vals.push_back(getEncodedVisibility(F));
    Vals.push_back(F->hasGC() ? GCMap[F->getGC()] : 0);
    Vals.push_back(F->hasUnnamedAddr());

    unsigned AbbrevToUse = 0;
    Stream.EmitRecord(bitc::MODULE_CODE_FUNCTION, Vals, AbbrevToUse);
    Vals.clear();
  }

  // Emit the alias information.
  for (Module::const_alias_iterator AI = M->alias_begin(), E = M->alias_end();
       AI != E; ++AI) {
    // ALIAS: [alias type, aliasee val#, linkage, visibility]
    Vals.push_back(VE.getTypeID(AI->getType()));
    Vals.push_back(VE.getValueID(AI->getAliasee()));
    Vals.push_back(getEncodedLinkage(AI));
    Vals.push_back(getEncodedVisibility(AI));
    unsigned AbbrevToUse = 0;
    Stream.EmitRecord(bitc::MODULE_CODE_ALIAS, Vals, AbbrevToUse);
    Vals.clear();
  }
}

static uint64_t GetOptimizationFlags(const Value *V) {
  uint64_t Flags = 0;

  if (const OverflowingBinaryOperator *OBO =
        dyn_cast<OverflowingBinaryOperator>(V)) {
    if (OBO->hasNoSignedWrap())
      Flags |= 1 << bitc::OBO_NO_SIGNED_WRAP;
    if (OBO->hasNoUnsignedWrap())
      Flags |= 1 << bitc::OBO_NO_UNSIGNED_WRAP;
  } else if (const PossiblyExactOperator *PEO =
               dyn_cast<PossiblyExactOperator>(V)) {
    if (PEO->isExact())
      Flags |= 1 << bitc::PEO_EXACT;
  }

  return Flags;
}

static void WriteMDNode(const MDNode *N,
                        const ValueEnumerator &VE,
                        BitstreamWriter &Stream,
                        SmallVector<uint64_t, 64> &Record) {
  for (unsigned i = 0, e = N->getNumOperands(); i != e; ++i) {
    if (N->getOperand(i)) {
      Record.push_back(VE.getTypeID(N->getOperand(i)->getType()));
      Record.push_back(VE.getValueID(N->getOperand(i)));
    } else {
      Record.push_back(VE.getTypeID(Type::getVoidTy(N->getContext())));
      Record.push_back(0);
    }
  }
  unsigned MDCode = N->isFunctionLocal() ? bitc::METADATA_FN_NODE :
                                           bitc::METADATA_NODE;
  Stream.EmitRecord(MDCode, Record, 0);
  Record.clear();
}

static void WriteModuleMetadata(const Module *M,
                                const ValueEnumerator &VE,
                                BitstreamWriter &Stream) {
  const ValueEnumerator::ValueList &Vals = VE.getMDValues();
  bool StartedMetadataBlock = false;
  unsigned MDSAbbrev = 0;
  SmallVector<uint64_t, 64> Record;
  for (unsigned i = 0, e = Vals.size(); i != e; ++i) {

    if (const MDNode *N = dyn_cast<MDNode>(Vals[i].first)) {
      if (!N->isFunctionLocal() || !N->getFunction()) {
        if (!StartedMetadataBlock) {
          Stream.EnterSubblock(bitc::METADATA_BLOCK_ID, 3);
          StartedMetadataBlock = true;
        }
        WriteMDNode(N, VE, Stream, Record);
      }
    } else if (const MDString *MDS = dyn_cast<MDString>(Vals[i].first)) {
      if (!StartedMetadataBlock)  {
        Stream.EnterSubblock(bitc::METADATA_BLOCK_ID, 3);

        // Abbrev for METADATA_STRING.
        BitCodeAbbrev *Abbv = new BitCodeAbbrev();
        Abbv->Add(BitCodeAbbrevOp(bitc::METADATA_STRING));
        Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
        Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8));
        MDSAbbrev = Stream.EmitAbbrev(Abbv);
        StartedMetadataBlock = true;
      }

      // Code: [strchar x N]
      Record.append(MDS->begin(), MDS->end());

      // Emit the finished record.
      Stream.EmitRecord(bitc::METADATA_STRING, Record, MDSAbbrev);
      Record.clear();
    }
  }

  // Write named metadata.
  for (Module::const_named_metadata_iterator I = M->named_metadata_begin(),
       E = M->named_metadata_end(); I != E; ++I) {
    const NamedMDNode *NMD = I;
    if (!StartedMetadataBlock)  {
      Stream.EnterSubblock(bitc::METADATA_BLOCK_ID, 3);
      StartedMetadataBlock = true;
    }

    // Write name.
    StringRef Str = NMD->getName();
    for (unsigned i = 0, e = Str.size(); i != e; ++i)
      Record.push_back(Str[i]);
    Stream.EmitRecord(bitc::METADATA_NAME, Record, 0/*TODO*/);
    Record.clear();

    // Write named metadata operands.
    for (unsigned i = 0, e = NMD->getNumOperands(); i != e; ++i)
      Record.push_back(VE.getValueID(NMD->getOperand(i)));
    Stream.EmitRecord(bitc::METADATA_NAMED_NODE, Record, 0);
    Record.clear();
  }

  if (StartedMetadataBlock)
    Stream.ExitBlock();
}

static void WriteFunctionLocalMetadata(const Function &F,
                                       const ValueEnumerator &VE,
                                       BitstreamWriter &Stream) {
  bool StartedMetadataBlock = false;
  SmallVector<uint64_t, 64> Record;
  const SmallVector<const MDNode *, 8> &Vals = VE.getFunctionLocalMDValues();
  for (unsigned i = 0, e = Vals.size(); i != e; ++i)
    if (const MDNode *N = Vals[i])
      if (N->isFunctionLocal() && N->getFunction() == &F) {
        if (!StartedMetadataBlock) {
          Stream.EnterSubblock(bitc::METADATA_BLOCK_ID, 3);
          StartedMetadataBlock = true;
        }
        WriteMDNode(N, VE, Stream, Record);
      }
      
  if (StartedMetadataBlock)
    Stream.ExitBlock();
}

static void WriteMetadataAttachment(const Function &F,
                                    const ValueEnumerator &VE,
                                    BitstreamWriter &Stream) {
  Stream.EnterSubblock(bitc::METADATA_ATTACHMENT_ID, 3);

  SmallVector<uint64_t, 64> Record;

  // Write metadata attachments
  // METADATA_ATTACHMENT - [m x [value, [n x [id, mdnode]]]
  SmallVector<std::pair<unsigned, MDNode*>, 4> MDs;
  
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      MDs.clear();
      I->getAllMetadataOtherThanDebugLoc(MDs);
      
      // If no metadata, ignore instruction.
      if (MDs.empty()) continue;

      Record.push_back(VE.getInstructionID(I));
      
      for (unsigned i = 0, e = MDs.size(); i != e; ++i) {
        Record.push_back(MDs[i].first);
        Record.push_back(VE.getValueID(MDs[i].second));
      }
      Stream.EmitRecord(bitc::METADATA_ATTACHMENT, Record, 0);
      Record.clear();
    }

  Stream.ExitBlock();
}

static void WriteModuleMetadataStore(const Module *M, BitstreamWriter &Stream) {
  SmallVector<uint64_t, 64> Record;

  // Write metadata kinds
  // METADATA_KIND - [n x [id, name]]
  SmallVector<StringRef, 4> Names;
  M->getMDKindNames(Names);
  
  if (Names.empty()) return;

  Stream.EnterSubblock(bitc::METADATA_BLOCK_ID, 3);
  
  for (unsigned MDKindID = 0, e = Names.size(); MDKindID != e; ++MDKindID) {
    Record.push_back(MDKindID);
    StringRef KName = Names[MDKindID];
    Record.append(KName.begin(), KName.end());
    
    Stream.EmitRecord(bitc::METADATA_KIND, Record, 0);
    Record.clear();
  }

  Stream.ExitBlock();
}

static void emitSignedInt64(SmallVectorImpl<uint64_t> &Vals, uint64_t V) {
  if ((int64_t)V >= 0)
    Vals.push_back(V << 1);
  else
    Vals.push_back((-V << 1) | 1);
}

static void EmitAPInt(SmallVectorImpl<uint64_t> &Vals,
                      unsigned &Code, unsigned &AbbrevToUse, const APInt &Val,
                      bool EmitSizeForWideNumbers = false
                      ) {
  if (Val.getBitWidth() <= 64) {
    uint64_t V = Val.getSExtValue();
    emitSignedInt64(Vals, V);
    Code = bitc::CST_CODE_INTEGER;
    AbbrevToUse = CONSTANTS_INTEGER_ABBREV;
  } else {
    // Wide integers, > 64 bits in size.
    // We have an arbitrary precision integer value to write whose
    // bit width is > 64. However, in canonical unsigned integer
    // format it is likely that the high bits are going to be zero.
    // So, we only write the number of active words.
    unsigned NWords = Val.getActiveWords();
    
    if (EmitSizeForWideNumbers)
      Vals.push_back(NWords);
    
    const uint64_t *RawWords = Val.getRawData();
    for (unsigned i = 0; i != NWords; ++i) {
      emitSignedInt64(Vals, RawWords[i]);
    }
    Code = bitc::CST_CODE_WIDE_INTEGER;
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
  Type *LastTy = 0;
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
      Record.push_back(unsigned(IA->hasSideEffects()) |
                       unsigned(IA->isAlignStack()) << 1 |
                       unsigned(IA->getDialect()&1) << 2);

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
      EmitAPInt(Record, Code, AbbrevToUse, IV->getValue());
    } else if (const ConstantFP *CFP = dyn_cast<ConstantFP>(C)) {
      Code = bitc::CST_CODE_FLOAT;
      Type *Ty = CFP->getType();
      if (Ty->isHalfTy() || Ty->isFloatTy() || Ty->isDoubleTy()) {
        Record.push_back(CFP->getValueAPF().bitcastToAPInt().getZExtValue());
      } else if (Ty->isX86_FP80Ty()) {
        // api needed to prevent premature destruction
        // bits are not in the same order as a normal i80 APInt, compensate.
        APInt api = CFP->getValueAPF().bitcastToAPInt();
        const uint64_t *p = api.getRawData();
        Record.push_back((p[1] << 48) | (p[0] >> 16));
        Record.push_back(p[0] & 0xffffLL);
      } else if (Ty->isFP128Ty() || Ty->isPPC_FP128Ty()) {
        APInt api = CFP->getValueAPF().bitcastToAPInt();
        const uint64_t *p = api.getRawData();
        Record.push_back(p[0]);
        Record.push_back(p[1]);
      } else {
        assert (0 && "Unknown FP type!");
      }
    } else if (isa<ConstantDataSequential>(C) &&
               cast<ConstantDataSequential>(C)->isString()) {
      const ConstantDataSequential *Str = cast<ConstantDataSequential>(C);
      // Emit constant strings specially.
      unsigned NumElts = Str->getNumElements();
      // If this is a null-terminated string, use the denser CSTRING encoding.
      if (Str->isCString()) {
        Code = bitc::CST_CODE_CSTRING;
        --NumElts;  // Don't encode the null, which isn't allowed by char6.
      } else {
        Code = bitc::CST_CODE_STRING;
        AbbrevToUse = String8Abbrev;
      }
      bool isCStr7 = Code == bitc::CST_CODE_CSTRING;
      bool isCStrChar6 = Code == bitc::CST_CODE_CSTRING;
      for (unsigned i = 0; i != NumElts; ++i) {
        unsigned char V = Str->getElementAsInteger(i);
        Record.push_back(V);
        isCStr7 &= (V & 128) == 0;
        if (isCStrChar6)
          isCStrChar6 = BitCodeAbbrevOp::isChar6(V);
      }
      
      if (isCStrChar6)
        AbbrevToUse = CString6Abbrev;
      else if (isCStr7)
        AbbrevToUse = CString7Abbrev;
    } else if (const ConstantDataSequential *CDS = 
                  dyn_cast<ConstantDataSequential>(C)) {
      Code = bitc::CST_CODE_DATA;
      Type *EltTy = CDS->getType()->getElementType();
      if (isa<IntegerType>(EltTy)) {
        for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i)
          Record.push_back(CDS->getElementAsInteger(i));
      } else if (EltTy->isFloatTy()) {
        for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
          union { float F; uint32_t I; };
          F = CDS->getElementAsFloat(i);
          Record.push_back(I);
        }
      } else {
        assert(EltTy->isDoubleTy() && "Unknown ConstantData element type");
        for (unsigned i = 0, e = CDS->getNumElements(); i != e; ++i) {
          union { double F; uint64_t I; };
          F = CDS->getElementAsDouble(i);
          Record.push_back(I);
        }
      }
    } else if (isa<ConstantArray>(C) || isa<ConstantStruct>(C) ||
               isa<ConstantVector>(C)) {
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
          uint64_t Flags = GetOptimizationFlags(CE);
          if (Flags != 0)
            Record.push_back(Flags);
        }
        break;
      case Instruction::GetElementPtr:
        Code = bitc::CST_CODE_CE_GEP;
        if (cast<GEPOperator>(C)->isInBounds())
          Code = bitc::CST_CODE_CE_INBOUNDS_GEP;
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
        // If the return type and argument types are the same, this is a
        // standard shufflevector instruction.  If the types are different,
        // then the shuffle is widening or truncating the input vectors, and
        // the argument type must also be encoded.
        if (C->getType() == C->getOperand(0)->getType()) {
          Code = bitc::CST_CODE_CE_SHUFFLEVEC;
        } else {
          Code = bitc::CST_CODE_CE_SHUFVEC_EX;
          Record.push_back(VE.getTypeID(C->getOperand(0)->getType()));
        }
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
    } else if (const BlockAddress *BA = dyn_cast<BlockAddress>(C)) {
      Code = bitc::CST_CODE_BLOCKADDRESS;
      Record.push_back(VE.getTypeID(BA->getFunction()->getType()));
      Record.push_back(VE.getValueID(BA->getFunction()));
      Record.push_back(VE.getGlobalBasicBlockID(BA->getBasicBlock()));
    } else {
#ifndef NDEBUG
      C->dump();
#endif
      llvm_unreachable("Unknown constant!");
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
/// type ID.  The value ID that is written is encoded relative to the InstID.
static bool PushValueAndType(const Value *V, unsigned InstID,
                             SmallVector<unsigned, 64> &Vals,
                             ValueEnumerator &VE) {
  unsigned ValID = VE.getValueID(V);
  // Make encoding relative to the InstID.
  Vals.push_back(InstID - ValID);
  if (ValID >= InstID) {
    Vals.push_back(VE.getTypeID(V->getType()));
    return true;
  }
  return false;
}

/// pushValue - Like PushValueAndType, but where the type of the value is
/// omitted (perhaps it was already encoded in an earlier operand).
static void pushValue(const Value *V, unsigned InstID,
                      SmallVector<unsigned, 64> &Vals,
                      ValueEnumerator &VE) {
  unsigned ValID = VE.getValueID(V);
  Vals.push_back(InstID - ValID);
}

static void pushValue64(const Value *V, unsigned InstID,
                        SmallVector<uint64_t, 128> &Vals,
                        ValueEnumerator &VE) {
  uint64_t ValID = VE.getValueID(V);
  Vals.push_back(InstID - ValID);
}

static void pushValueSigned(const Value *V, unsigned InstID,
                            SmallVector<uint64_t, 128> &Vals,
                            ValueEnumerator &VE) {
  unsigned ValID = VE.getValueID(V);
  int64_t diff = ((int32_t)InstID - (int32_t)ValID);
  emitSignedInt64(Vals, diff);
}

/// WriteInstruction - Emit an instruction to the specified stream.
static void WriteInstruction(const Instruction &I, unsigned InstID,
                             ValueEnumerator &VE, BitstreamWriter &Stream,
                             SmallVector<unsigned, 64> &Vals) {
  unsigned Code = 0;
  unsigned AbbrevToUse = 0;
  VE.setInstructionID(&I);
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
      pushValue(I.getOperand(1), InstID, Vals, VE);
      Vals.push_back(GetEncodedBinaryOpcode(I.getOpcode()));
      uint64_t Flags = GetOptimizationFlags(&I);
      if (Flags != 0) {
        if (AbbrevToUse == FUNCTION_INST_BINOP_ABBREV)
          AbbrevToUse = FUNCTION_INST_BINOP_FLAGS_ABBREV;
        Vals.push_back(Flags);
      }
    }
    break;

  case Instruction::GetElementPtr:
    Code = bitc::FUNC_CODE_INST_GEP;
    if (cast<GEPOperator>(&I)->isInBounds())
      Code = bitc::FUNC_CODE_INST_INBOUNDS_GEP;
    for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
      PushValueAndType(I.getOperand(i), InstID, Vals, VE);
    break;
  case Instruction::ExtractValue: {
    Code = bitc::FUNC_CODE_INST_EXTRACTVAL;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    const ExtractValueInst *EVI = cast<ExtractValueInst>(&I);
    for (const unsigned *i = EVI->idx_begin(), *e = EVI->idx_end(); i != e; ++i)
      Vals.push_back(*i);
    break;
  }
  case Instruction::InsertValue: {
    Code = bitc::FUNC_CODE_INST_INSERTVAL;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    PushValueAndType(I.getOperand(1), InstID, Vals, VE);
    const InsertValueInst *IVI = cast<InsertValueInst>(&I);
    for (const unsigned *i = IVI->idx_begin(), *e = IVI->idx_end(); i != e; ++i)
      Vals.push_back(*i);
    break;
  }
  case Instruction::Select:
    Code = bitc::FUNC_CODE_INST_VSELECT;
    PushValueAndType(I.getOperand(1), InstID, Vals, VE);
    pushValue(I.getOperand(2), InstID, Vals, VE);
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    break;
  case Instruction::ExtractElement:
    Code = bitc::FUNC_CODE_INST_EXTRACTELT;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    pushValue(I.getOperand(1), InstID, Vals, VE);
    break;
  case Instruction::InsertElement:
    Code = bitc::FUNC_CODE_INST_INSERTELT;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    pushValue(I.getOperand(1), InstID, Vals, VE);
    pushValue(I.getOperand(2), InstID, Vals, VE);
    break;
  case Instruction::ShuffleVector:
    Code = bitc::FUNC_CODE_INST_SHUFFLEVEC;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    pushValue(I.getOperand(1), InstID, Vals, VE);
    pushValue(I.getOperand(2), InstID, Vals, VE);
    break;
  case Instruction::ICmp:
  case Instruction::FCmp:
    // compare returning Int1Ty or vector of Int1Ty
    Code = bitc::FUNC_CODE_INST_CMP2;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    pushValue(I.getOperand(1), InstID, Vals, VE);
    Vals.push_back(cast<CmpInst>(I).getPredicate());
    break;

  case Instruction::Ret:
    {
      Code = bitc::FUNC_CODE_INST_RET;
      unsigned NumOperands = I.getNumOperands();
      if (NumOperands == 0)
        AbbrevToUse = FUNCTION_INST_RET_VOID_ABBREV;
      else if (NumOperands == 1) {
        if (!PushValueAndType(I.getOperand(0), InstID, Vals, VE))
          AbbrevToUse = FUNCTION_INST_RET_VAL_ABBREV;
      } else {
        for (unsigned i = 0, e = NumOperands; i != e; ++i)
          PushValueAndType(I.getOperand(i), InstID, Vals, VE);
      }
    }
    break;
  case Instruction::Br:
    {
      Code = bitc::FUNC_CODE_INST_BR;
      BranchInst &II = cast<BranchInst>(I);
      Vals.push_back(VE.getValueID(II.getSuccessor(0)));
      if (II.isConditional()) {
        Vals.push_back(VE.getValueID(II.getSuccessor(1)));
        pushValue(II.getCondition(), InstID, Vals, VE);
      }
    }
    break;
  case Instruction::Switch:
    {
      // Redefine Vals, since here we need to use 64 bit values
      // explicitly to store large APInt numbers.
      SmallVector<uint64_t, 128> Vals64;
      
      Code = bitc::FUNC_CODE_INST_SWITCH;
      SwitchInst &SI = cast<SwitchInst>(I);
      
      uint32_t SwitchRecordHeader = SI.hash() | (SWITCH_INST_MAGIC << 16); 
      Vals64.push_back(SwitchRecordHeader);      
      
      Vals64.push_back(VE.getTypeID(SI.getCondition()->getType()));
      pushValue64(SI.getCondition(), InstID, Vals64, VE);
      Vals64.push_back(VE.getValueID(SI.getDefaultDest()));
      Vals64.push_back(SI.getNumCases());
      for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end();
           i != e; ++i) {
        IntegersSubset& CaseRanges = i.getCaseValueEx();
        unsigned Code, Abbrev; // will unused.
        
        if (CaseRanges.isSingleNumber()) {
          Vals64.push_back(1/*NumItems = 1*/);
          Vals64.push_back(true/*IsSingleNumber = true*/);
          EmitAPInt(Vals64, Code, Abbrev, CaseRanges.getSingleNumber(0), true);
        } else {
          
          Vals64.push_back(CaseRanges.getNumItems());
          
          if (CaseRanges.isSingleNumbersOnly()) {
            for (unsigned ri = 0, rn = CaseRanges.getNumItems();
                 ri != rn; ++ri) {
              
              Vals64.push_back(true/*IsSingleNumber = true*/);
              
              EmitAPInt(Vals64, Code, Abbrev,
                        CaseRanges.getSingleNumber(ri), true);
            }
          } else
            for (unsigned ri = 0, rn = CaseRanges.getNumItems();
                 ri != rn; ++ri) {
              IntegersSubset::Range r = CaseRanges.getItem(ri);
              bool IsSingleNumber = CaseRanges.isSingleNumber(ri);
    
              Vals64.push_back(IsSingleNumber);
              
              EmitAPInt(Vals64, Code, Abbrev, r.getLow(), true);
              if (!IsSingleNumber)
                EmitAPInt(Vals64, Code, Abbrev, r.getHigh(), true);
            }
        }
        Vals64.push_back(VE.getValueID(i.getCaseSuccessor()));
      }
      
      Stream.EmitRecord(Code, Vals64, AbbrevToUse);
      
      // Also do expected action - clear external Vals collection:
      Vals.clear();
      return;
    }
    break;
  case Instruction::IndirectBr:
    Code = bitc::FUNC_CODE_INST_INDIRECTBR;
    Vals.push_back(VE.getTypeID(I.getOperand(0)->getType()));
    // Encode the address operand as relative, but not the basic blocks.
    pushValue(I.getOperand(0), InstID, Vals, VE);
    for (unsigned i = 1, e = I.getNumOperands(); i != e; ++i)
      Vals.push_back(VE.getValueID(I.getOperand(i)));
    break;
      
  case Instruction::Invoke: {
    const InvokeInst *II = cast<InvokeInst>(&I);
    const Value *Callee(II->getCalledValue());
    PointerType *PTy = cast<PointerType>(Callee->getType());
    FunctionType *FTy = cast<FunctionType>(PTy->getElementType());
    Code = bitc::FUNC_CODE_INST_INVOKE;

    Vals.push_back(VE.getAttributeID(II->getAttributes()));
    Vals.push_back(II->getCallingConv());
    Vals.push_back(VE.getValueID(II->getNormalDest()));
    Vals.push_back(VE.getValueID(II->getUnwindDest()));
    PushValueAndType(Callee, InstID, Vals, VE);

    // Emit value #'s for the fixed parameters.
    for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
      pushValue(I.getOperand(i), InstID, Vals, VE);  // fixed param.

    // Emit type/value pairs for varargs params.
    if (FTy->isVarArg()) {
      for (unsigned i = FTy->getNumParams(), e = I.getNumOperands()-3;
           i != e; ++i)
        PushValueAndType(I.getOperand(i), InstID, Vals, VE); // vararg
    }
    break;
  }
  case Instruction::Resume:
    Code = bitc::FUNC_CODE_INST_RESUME;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    break;
  case Instruction::Unreachable:
    Code = bitc::FUNC_CODE_INST_UNREACHABLE;
    AbbrevToUse = FUNCTION_INST_UNREACHABLE_ABBREV;
    break;

  case Instruction::PHI: {
    const PHINode &PN = cast<PHINode>(I);
    Code = bitc::FUNC_CODE_INST_PHI;
    // With the newer instruction encoding, forward references could give
    // negative valued IDs.  This is most common for PHIs, so we use
    // signed VBRs.
    SmallVector<uint64_t, 128> Vals64;
    Vals64.push_back(VE.getTypeID(PN.getType()));
    for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
      pushValueSigned(PN.getIncomingValue(i), InstID, Vals64, VE);
      Vals64.push_back(VE.getValueID(PN.getIncomingBlock(i)));
    }
    // Emit a Vals64 vector and exit.
    Stream.EmitRecord(Code, Vals64, AbbrevToUse);
    Vals64.clear();
    return;
  }

  case Instruction::LandingPad: {
    const LandingPadInst &LP = cast<LandingPadInst>(I);
    Code = bitc::FUNC_CODE_INST_LANDINGPAD;
    Vals.push_back(VE.getTypeID(LP.getType()));
    PushValueAndType(LP.getPersonalityFn(), InstID, Vals, VE);
    Vals.push_back(LP.isCleanup());
    Vals.push_back(LP.getNumClauses());
    for (unsigned I = 0, E = LP.getNumClauses(); I != E; ++I) {
      if (LP.isCatch(I))
        Vals.push_back(LandingPadInst::Catch);
      else
        Vals.push_back(LandingPadInst::Filter);
      PushValueAndType(LP.getClause(I), InstID, Vals, VE);
    }
    break;
  }

  case Instruction::Alloca:
    Code = bitc::FUNC_CODE_INST_ALLOCA;
    Vals.push_back(VE.getTypeID(I.getType()));
    Vals.push_back(VE.getTypeID(I.getOperand(0)->getType()));
    Vals.push_back(VE.getValueID(I.getOperand(0))); // size.
    Vals.push_back(Log2_32(cast<AllocaInst>(I).getAlignment())+1);
    break;

  case Instruction::Load:
    if (cast<LoadInst>(I).isAtomic()) {
      Code = bitc::FUNC_CODE_INST_LOADATOMIC;
      PushValueAndType(I.getOperand(0), InstID, Vals, VE);
    } else {
      Code = bitc::FUNC_CODE_INST_LOAD;
      if (!PushValueAndType(I.getOperand(0), InstID, Vals, VE))  // ptr
        AbbrevToUse = FUNCTION_INST_LOAD_ABBREV;
    }
    Vals.push_back(Log2_32(cast<LoadInst>(I).getAlignment())+1);
    Vals.push_back(cast<LoadInst>(I).isVolatile());
    if (cast<LoadInst>(I).isAtomic()) {
      Vals.push_back(GetEncodedOrdering(cast<LoadInst>(I).getOrdering()));
      Vals.push_back(GetEncodedSynchScope(cast<LoadInst>(I).getSynchScope()));
    }
    break;
  case Instruction::Store:
    if (cast<StoreInst>(I).isAtomic())
      Code = bitc::FUNC_CODE_INST_STOREATOMIC;
    else
      Code = bitc::FUNC_CODE_INST_STORE;
    PushValueAndType(I.getOperand(1), InstID, Vals, VE);  // ptrty + ptr
    pushValue(I.getOperand(0), InstID, Vals, VE);         // val.
    Vals.push_back(Log2_32(cast<StoreInst>(I).getAlignment())+1);
    Vals.push_back(cast<StoreInst>(I).isVolatile());
    if (cast<StoreInst>(I).isAtomic()) {
      Vals.push_back(GetEncodedOrdering(cast<StoreInst>(I).getOrdering()));
      Vals.push_back(GetEncodedSynchScope(cast<StoreInst>(I).getSynchScope()));
    }
    break;
  case Instruction::AtomicCmpXchg:
    Code = bitc::FUNC_CODE_INST_CMPXCHG;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);  // ptrty + ptr
    pushValue(I.getOperand(1), InstID, Vals, VE);         // cmp.
    pushValue(I.getOperand(2), InstID, Vals, VE);         // newval.
    Vals.push_back(cast<AtomicCmpXchgInst>(I).isVolatile());
    Vals.push_back(GetEncodedOrdering(
                     cast<AtomicCmpXchgInst>(I).getOrdering()));
    Vals.push_back(GetEncodedSynchScope(
                     cast<AtomicCmpXchgInst>(I).getSynchScope()));
    break;
  case Instruction::AtomicRMW:
    Code = bitc::FUNC_CODE_INST_ATOMICRMW;
    PushValueAndType(I.getOperand(0), InstID, Vals, VE);  // ptrty + ptr
    pushValue(I.getOperand(1), InstID, Vals, VE);         // val.
    Vals.push_back(GetEncodedRMWOperation(
                     cast<AtomicRMWInst>(I).getOperation()));
    Vals.push_back(cast<AtomicRMWInst>(I).isVolatile());
    Vals.push_back(GetEncodedOrdering(cast<AtomicRMWInst>(I).getOrdering()));
    Vals.push_back(GetEncodedSynchScope(
                     cast<AtomicRMWInst>(I).getSynchScope()));
    break;
  case Instruction::Fence:
    Code = bitc::FUNC_CODE_INST_FENCE;
    Vals.push_back(GetEncodedOrdering(cast<FenceInst>(I).getOrdering()));
    Vals.push_back(GetEncodedSynchScope(cast<FenceInst>(I).getSynchScope()));
    break;
  case Instruction::Call: {
    const CallInst &CI = cast<CallInst>(I);
    PointerType *PTy = cast<PointerType>(CI.getCalledValue()->getType());
    FunctionType *FTy = cast<FunctionType>(PTy->getElementType());

    Code = bitc::FUNC_CODE_INST_CALL;

    Vals.push_back(VE.getAttributeID(CI.getAttributes()));
    Vals.push_back((CI.getCallingConv() << 1) | unsigned(CI.isTailCall()));
    PushValueAndType(CI.getCalledValue(), InstID, Vals, VE);  // Callee

    // Emit value #'s for the fixed parameters.
    for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i) {
      // Check for labels (can happen with asm labels).
      if (FTy->getParamType(i)->isLabelTy())
        Vals.push_back(VE.getValueID(CI.getArgOperand(i)));
      else
        pushValue(CI.getArgOperand(i), InstID, Vals, VE);  // fixed param.
    }

    // Emit type/value pairs for varargs params.
    if (FTy->isVarArg()) {
      for (unsigned i = FTy->getNumParams(), e = CI.getNumArgOperands();
           i != e; ++i)
        PushValueAndType(CI.getArgOperand(i), InstID, Vals, VE);  // varargs
    }
    break;
  }
  case Instruction::VAArg:
    Code = bitc::FUNC_CODE_INST_VAARG;
    Vals.push_back(VE.getTypeID(I.getOperand(0)->getType()));   // valistty
    pushValue(I.getOperand(0), InstID, Vals, VE); // valist.
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

  // If there is function-local metadata, emit it now.
  WriteFunctionLocalMetadata(F, VE, Stream);

  // Keep a running idea of what the instruction ID is.
  unsigned InstID = CstEnd;

  bool NeedsMetadataAttachment = false;
  
  DebugLoc LastDL;
  
  // Finally, emit all the instructions, in order.
  for (Function::const_iterator BB = F.begin(), E = F.end(); BB != E; ++BB)
    for (BasicBlock::const_iterator I = BB->begin(), E = BB->end();
         I != E; ++I) {
      WriteInstruction(*I, InstID, VE, Stream, Vals);
      
      if (!I->getType()->isVoidTy())
        ++InstID;
      
      // If the instruction has metadata, write a metadata attachment later.
      NeedsMetadataAttachment |= I->hasMetadataOtherThanDebugLoc();
      
      // If the instruction has a debug location, emit it.
      DebugLoc DL = I->getDebugLoc();
      if (DL.isUnknown()) {
        // nothing todo.
      } else if (DL == LastDL) {
        // Just repeat the same debug loc as last time.
        Stream.EmitRecord(bitc::FUNC_CODE_DEBUG_LOC_AGAIN, Vals);
      } else {
        MDNode *Scope, *IA;
        DL.getScopeAndInlinedAt(Scope, IA, I->getContext());
        
        Vals.push_back(DL.getLine());
        Vals.push_back(DL.getCol());
        Vals.push_back(Scope ? VE.getValueID(Scope)+1 : 0);
        Vals.push_back(IA ? VE.getValueID(IA)+1 : 0);
        Stream.EmitRecord(bitc::FUNC_CODE_DEBUG_LOC, Vals);
        Vals.clear();
        
        LastDL = DL;
      }
    }

  // Emit names for all the instructions etc.
  WriteValueSymbolTable(F.getValueSymbolTable(), VE, Stream);

  if (NeedsMetadataAttachment)
    WriteMetadataAttachment(F, VE, Stream);
  VE.purgeFunction();
  Stream.ExitBlock();
}

// Emit blockinfo, which defines the standard abbreviations etc.
static void WriteBlockInfo(const ValueEnumerator &VE, BitstreamWriter &Stream) {
  // We only want to emit block info records for blocks that have multiple
  // instances: CONSTANTS_BLOCK, FUNCTION_BLOCK and VALUE_SYMTAB_BLOCK.
  // Other blocks can define their abbrevs inline.
  Stream.EnterBlockInfoBlock(2);

  { // 8-bit fixed-width VST_ENTRY/VST_BBENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 3));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 8));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_ENTRY_8_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }

  { // 7-bit fixed width VST_ENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_ENTRY));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 7));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_ENTRY_7_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // 6-bit char6 VST_ENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_ENTRY));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Char6));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_ENTRY_6_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // 6-bit char6 VST_BBENTRY strings.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::VST_CODE_BBENTRY));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Array));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Char6));
    if (Stream.EmitBlockInfoAbbrev(bitc::VALUE_SYMTAB_BLOCK_ID,
                                   Abbv) != VST_BBENTRY_6_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }



  { // SETTYPE abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_SETTYPE));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed,
                              Log2_32_Ceil(VE.getTypes().size()+1)));
    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_SETTYPE_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }

  { // INTEGER abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_INTEGER));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8));
    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_INTEGER_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
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
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // NULL abbrev for CONSTANTS_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::CST_CODE_NULL));
    if (Stream.EmitBlockInfoAbbrev(bitc::CONSTANTS_BLOCK_ID,
                                   Abbv) != CONSTANTS_NULL_Abbrev)
      llvm_unreachable("Unexpected abbrev ordering!");
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
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // INST_BINOP abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_BINOP));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // LHS
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // RHS
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 4)); // opc
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_BINOP_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // INST_BINOP_FLAGS abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_BINOP));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // LHS
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // RHS
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 4)); // opc
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 7)); // flags
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_BINOP_FLAGS_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
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
      llvm_unreachable("Unexpected abbrev ordering!");
  }

  { // INST_RET abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_RET));
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_RET_VOID_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // INST_RET abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_RET));
    Abbv->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // ValID
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_RET_VAL_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }
  { // INST_UNREACHABLE abbrev for FUNCTION_BLOCK.
    BitCodeAbbrev *Abbv = new BitCodeAbbrev();
    Abbv->Add(BitCodeAbbrevOp(bitc::FUNC_CODE_INST_UNREACHABLE));
    if (Stream.EmitBlockInfoAbbrev(bitc::FUNCTION_BLOCK_ID,
                                   Abbv) != FUNCTION_INST_UNREACHABLE_ABBREV)
      llvm_unreachable("Unexpected abbrev ordering!");
  }

  Stream.ExitBlock();
}

// Sort the Users based on the order in which the reader parses the bitcode 
// file.
static bool bitcodereader_order(const User *lhs, const User *rhs) {
  // TODO: Implement.
  return true;
}

static void WriteUseList(const Value *V, const ValueEnumerator &VE,
                         BitstreamWriter &Stream) {

  // One or zero uses can't get out of order.
  if (V->use_empty() || V->hasNUses(1))
    return;

  // Make a copy of the in-memory use-list for sorting.
  unsigned UseListSize = std::distance(V->use_begin(), V->use_end());
  SmallVector<const User*, 8> UseList;
  UseList.reserve(UseListSize);
  for (Value::const_use_iterator I = V->use_begin(), E = V->use_end();
       I != E; ++I) {
    const User *U = *I;
    UseList.push_back(U);
  }

  // Sort the copy based on the order read by the BitcodeReader.
  std::sort(UseList.begin(), UseList.end(), bitcodereader_order);

  // TODO: Generate a diff between the BitcodeWriter in-memory use-list and the
  // sorted list (i.e., the expected BitcodeReader in-memory use-list).

  // TODO: Emit the USELIST_CODE_ENTRYs.
}

static void WriteFunctionUseList(const Function *F, ValueEnumerator &VE,
                                 BitstreamWriter &Stream) {
  VE.incorporateFunction(*F);

  for (Function::const_arg_iterator AI = F->arg_begin(), AE = F->arg_end();
       AI != AE; ++AI)
    WriteUseList(AI, VE, Stream);
  for (Function::const_iterator BB = F->begin(), FE = F->end(); BB != FE;
       ++BB) {
    WriteUseList(BB, VE, Stream);
    for (BasicBlock::const_iterator II = BB->begin(), IE = BB->end(); II != IE;
         ++II) {
      WriteUseList(II, VE, Stream);
      for (User::const_op_iterator OI = II->op_begin(), E = II->op_end();
           OI != E; ++OI) {
        if ((isa<Constant>(*OI) && !isa<GlobalValue>(*OI)) ||
            isa<InlineAsm>(*OI))
          WriteUseList(*OI, VE, Stream);
      }
    }
  }
  VE.purgeFunction();
}

// Emit use-lists.
static void WriteModuleUseLists(const Module *M, ValueEnumerator &VE,
                                BitstreamWriter &Stream) {
  Stream.EnterSubblock(bitc::USELIST_BLOCK_ID, 3);

  // XXX: this modifies the module, but in a way that should never change the
  // behavior of any pass or codegen in LLVM. The problem is that GVs may
  // contain entries in the use_list that do not exist in the Module and are
  // not stored in the .bc file.
  for (Module::const_global_iterator I = M->global_begin(), E = M->global_end();
       I != E; ++I)
    I->removeDeadConstantUsers();
  
  // Write the global variables.
  for (Module::const_global_iterator GI = M->global_begin(), 
         GE = M->global_end(); GI != GE; ++GI) {
    WriteUseList(GI, VE, Stream);

    // Write the global variable initializers.
    if (GI->hasInitializer())
      WriteUseList(GI->getInitializer(), VE, Stream);
  }

  // Write the functions.
  for (Module::const_iterator FI = M->begin(), FE = M->end(); FI != FE; ++FI) {
    WriteUseList(FI, VE, Stream);
    if (!FI->isDeclaration())
      WriteFunctionUseList(FI, VE, Stream);
  }

  // Write the aliases.
  for (Module::const_alias_iterator AI = M->alias_begin(), AE = M->alias_end();
       AI != AE; ++AI) {
    WriteUseList(AI, VE, Stream);
    WriteUseList(AI->getAliasee(), VE, Stream);
  }

  Stream.ExitBlock();
}

/// WriteModule - Emit the specified module to the bitstream.
static void WriteModule(const Module *M, BitstreamWriter &Stream) {
  Stream.EnterSubblock(bitc::MODULE_BLOCK_ID, 3);

  SmallVector<unsigned, 1> Vals;
  unsigned CurVersion = 1;
  Vals.push_back(CurVersion);
  Stream.EmitRecord(bitc::MODULE_CODE_VERSION, Vals);

  // Analyze the module, enumerating globals, functions, etc.
  ValueEnumerator VE(M);

  // Emit blockinfo, which defines the standard abbreviations etc.
  WriteBlockInfo(VE, Stream);

  // Emit information about parameter attributes.
  WriteAttributeTable(VE, Stream);

  // Emit information describing all of the types in the module.
  WriteTypeTable(VE, Stream);

  // Emit top-level description of module, including target triple, inline asm,
  // descriptors for global variables, and function prototype info.
  WriteModuleInfo(M, VE, Stream);

  // Emit constants.
  WriteModuleConstants(VE, Stream);

  // Emit metadata.
  WriteModuleMetadata(M, VE, Stream);

  // Emit metadata.
  WriteModuleMetadataStore(M, Stream);

  // Emit names for globals/functions etc.
  WriteValueSymbolTable(M->getValueSymbolTable(), VE, Stream);

  // Emit use-lists.
  if (EnablePreserveUseListOrdering)
    WriteModuleUseLists(M, VE, Stream);

  // Emit function bodies.
  for (Module::const_iterator F = M->begin(), E = M->end(); F != E; ++F)
    if (!F->isDeclaration())
      WriteFunction(*F, VE, Stream);

  Stream.ExitBlock();
}

/// EmitDarwinBCHeader - If generating a bc file on darwin, we have to emit a
/// header and trailer to make it compatible with the system archiver.  To do
/// this we emit the following header, and then emit a trailer that pads the
/// file out to be a multiple of 16 bytes.
///
/// struct bc_header {
///   uint32_t Magic;         // 0x0B17C0DE
///   uint32_t Version;       // Version, currently always 0.
///   uint32_t BitcodeOffset; // Offset to traditional bitcode file.
///   uint32_t BitcodeSize;   // Size of traditional bitcode file.
///   uint32_t CPUType;       // CPU specifier.
///   ... potentially more later ...
/// };
enum {
  DarwinBCSizeFieldOffset = 3*4, // Offset to bitcode_size.
  DarwinBCHeaderSize = 5*4
};

static void WriteInt32ToBuffer(uint32_t Value, SmallVectorImpl<char> &Buffer,
                               uint32_t &Position) {
  Buffer[Position + 0] = (unsigned char) (Value >>  0);
  Buffer[Position + 1] = (unsigned char) (Value >>  8);
  Buffer[Position + 2] = (unsigned char) (Value >> 16);
  Buffer[Position + 3] = (unsigned char) (Value >> 24);
  Position += 4;
}

static void EmitDarwinBCHeaderAndTrailer(SmallVectorImpl<char> &Buffer,
                                         const Triple &TT) {
  unsigned CPUType = ~0U;

  // Match x86_64-*, i[3-9]86-*, powerpc-*, powerpc64-*, arm-*, thumb-*,
  // armv[0-9]-*, thumbv[0-9]-*, armv5te-*, or armv6t2-*. The CPUType is a magic
  // number from /usr/include/mach/machine.h.  It is ok to reproduce the
  // specific constants here because they are implicitly part of the Darwin ABI.
  enum {
    DARWIN_CPU_ARCH_ABI64      = 0x01000000,
    DARWIN_CPU_TYPE_X86        = 7,
    DARWIN_CPU_TYPE_ARM        = 12,
    DARWIN_CPU_TYPE_POWERPC    = 18
  };

  Triple::ArchType Arch = TT.getArch();
  if (Arch == Triple::x86_64)
    CPUType = DARWIN_CPU_TYPE_X86 | DARWIN_CPU_ARCH_ABI64;
  else if (Arch == Triple::x86)
    CPUType = DARWIN_CPU_TYPE_X86;
  else if (Arch == Triple::ppc)
    CPUType = DARWIN_CPU_TYPE_POWERPC;
  else if (Arch == Triple::ppc64)
    CPUType = DARWIN_CPU_TYPE_POWERPC | DARWIN_CPU_ARCH_ABI64;
  else if (Arch == Triple::arm || Arch == Triple::thumb)
    CPUType = DARWIN_CPU_TYPE_ARM;

  // Traditional Bitcode starts after header.
  assert(Buffer.size() >= DarwinBCHeaderSize &&
         "Expected header size to be reserved");
  unsigned BCOffset = DarwinBCHeaderSize;
  unsigned BCSize = Buffer.size()-DarwinBCHeaderSize;

  // Write the magic and version.
  unsigned Position = 0;
  WriteInt32ToBuffer(0x0B17C0DE , Buffer, Position);
  WriteInt32ToBuffer(0          , Buffer, Position); // Version.
  WriteInt32ToBuffer(BCOffset   , Buffer, Position);
  WriteInt32ToBuffer(BCSize     , Buffer, Position);
  WriteInt32ToBuffer(CPUType    , Buffer, Position);

  // If the file is not a multiple of 16 bytes, insert dummy padding.
  while (Buffer.size() & 15)
    Buffer.push_back(0);
}

/// WriteBitcodeToFile - Write the specified module to the specified output
/// stream.
void llvm::WriteBitcodeToFile(const Module *M, raw_ostream &Out) {
  SmallVector<char, 1024> Buffer;
  Buffer.reserve(256*1024);

  // If this is darwin or another generic macho target, reserve space for the
  // header.
  Triple TT(M->getTargetTriple());
  if (TT.isOSDarwin())
    Buffer.insert(Buffer.begin(), DarwinBCHeaderSize, 0);

  // Emit the module into the buffer.
  {
    BitstreamWriter Stream(Buffer);

    // Emit the file header.
    Stream.Emit((unsigned)'B', 8);
    Stream.Emit((unsigned)'C', 8);
    Stream.Emit(0x0, 4);
    Stream.Emit(0xC, 4);
    Stream.Emit(0xE, 4);
    Stream.Emit(0xD, 4);

    // Emit the module.
    WriteModule(M, Stream);
  }

  if (TT.isOSDarwin())
    EmitDarwinBCHeaderAndTrailer(Buffer, TT);

  // Write the generated bitstream to "Out".
  Out.write((char*)&Buffer.front(), Buffer.size());
}
