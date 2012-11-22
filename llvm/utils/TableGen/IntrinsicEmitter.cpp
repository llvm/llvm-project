//===- IntrinsicEmitter.cpp - Generate intrinsic information --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits information about intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenIntrinsics.h"
#include "CodeGenTarget.h"
#include "SequenceToOffsetTable.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringMatcher.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
using namespace llvm;

namespace {
class IntrinsicEmitter {
  RecordKeeper &Records;
  bool TargetOnly;
  std::string TargetPrefix;

public:
  IntrinsicEmitter(RecordKeeper &R, bool T)
    : Records(R), TargetOnly(T) {}

  void run(raw_ostream &OS);

  void EmitPrefix(raw_ostream &OS);

  void EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints,
                    raw_ostream &OS);

  void EmitFnNameRecognizer(const std::vector<CodeGenIntrinsic> &Ints,
                            raw_ostream &OS);
  void EmitIntrinsicToNameTable(const std::vector<CodeGenIntrinsic> &Ints,
                                raw_ostream &OS);
  void EmitIntrinsicToOverloadTable(const std::vector<CodeGenIntrinsic> &Ints,
                                    raw_ostream &OS);
  void EmitVerifier(const std::vector<CodeGenIntrinsic> &Ints,
                    raw_ostream &OS);
  void EmitGenerator(const std::vector<CodeGenIntrinsic> &Ints,
                     raw_ostream &OS);
  void EmitAttributes(const std::vector<CodeGenIntrinsic> &Ints,
                      raw_ostream &OS);
  void EmitModRefBehavior(const std::vector<CodeGenIntrinsic> &Ints,
                          raw_ostream &OS);
  void EmitIntrinsicToGCCBuiltinMap(const std::vector<CodeGenIntrinsic> &Ints,
                                    raw_ostream &OS);
  void EmitSuffix(raw_ostream &OS);
};
} // End anonymous namespace

//===----------------------------------------------------------------------===//
// IntrinsicEmitter Implementation
//===----------------------------------------------------------------------===//

void IntrinsicEmitter::run(raw_ostream &OS) {
  emitSourceFileHeader("Intrinsic Function Source Fragment", OS);

  std::vector<CodeGenIntrinsic> Ints = LoadIntrinsics(Records, TargetOnly);

  if (TargetOnly && !Ints.empty())
    TargetPrefix = Ints[0].TargetPrefix;

  EmitPrefix(OS);

  // Emit the enum information.
  EmitEnumInfo(Ints, OS);

  // Emit the intrinsic ID -> name table.
  EmitIntrinsicToNameTable(Ints, OS);

  // Emit the intrinsic ID -> overload table.
  EmitIntrinsicToOverloadTable(Ints, OS);

  // Emit the function name recognizer.
  EmitFnNameRecognizer(Ints, OS);
  
  // Emit the intrinsic declaration generator.
  EmitGenerator(Ints, OS);
  
  // Emit the intrinsic parameter attributes.
  EmitAttributes(Ints, OS);

  // Emit intrinsic alias analysis mod/ref behavior.
  EmitModRefBehavior(Ints, OS);

  // Emit code to translate GCC builtins into LLVM intrinsics.
  EmitIntrinsicToGCCBuiltinMap(Ints, OS);

  EmitSuffix(OS);
}

void IntrinsicEmitter::EmitPrefix(raw_ostream &OS) {
  OS << "// VisualStudio defines setjmp as _setjmp\n"
        "#if defined(_MSC_VER) && defined(setjmp) && \\\n"
        "                         !defined(setjmp_undefined_for_msvc)\n"
        "#  pragma push_macro(\"setjmp\")\n"
        "#  undef setjmp\n"
        "#  define setjmp_undefined_for_msvc\n"
        "#endif\n\n";
}

void IntrinsicEmitter::EmitSuffix(raw_ostream &OS) {
  OS << "#if defined(_MSC_VER) && defined(setjmp_undefined_for_msvc)\n"
        "// let's return it to _setjmp state\n"
        "#  pragma pop_macro(\"setjmp\")\n"
        "#  undef setjmp_undefined_for_msvc\n"
        "#endif\n\n";
}

void IntrinsicEmitter::EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints,
                                    raw_ostream &OS) {
  OS << "// Enum values for Intrinsics.h\n";
  OS << "#ifdef GET_INTRINSIC_ENUM_VALUES\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "    " << Ints[i].EnumName;
    OS << ((i != e-1) ? ", " : "  ");
    OS << std::string(40-Ints[i].EnumName.size(), ' ') 
      << "// " << Ints[i].Name << "\n";
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitFnNameRecognizer(const std::vector<CodeGenIntrinsic> &Ints, 
                     raw_ostream &OS) {
  // Build a 'first character of function name' -> intrinsic # mapping.
  std::map<char, std::vector<unsigned> > IntMapping;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    IntMapping[Ints[i].Name[5]].push_back(i);
  
  OS << "// Function name -> enum value recognizer code.\n";
  OS << "#ifdef GET_FUNCTION_RECOGNIZER\n";
  OS << "  StringRef NameR(Name+6, Len-6);   // Skip over 'llvm.'\n";
  OS << "  switch (Name[5]) {                  // Dispatch on first letter.\n";
  OS << "  default: break;\n";
  // Emit the intrinsic matching stuff by first letter.
  for (std::map<char, std::vector<unsigned> >::iterator I = IntMapping.begin(),
       E = IntMapping.end(); I != E; ++I) {
    OS << "  case '" << I->first << "':\n";
    std::vector<unsigned> &IntList = I->second;

    // Emit all the overloaded intrinsics first, build a table of the
    // non-overloaded ones.
    std::vector<StringMatcher::StringPair> MatchTable;
    
    for (unsigned i = 0, e = IntList.size(); i != e; ++i) {
      unsigned IntNo = IntList[i];
      std::string Result = "return " + TargetPrefix + "Intrinsic::" +
        Ints[IntNo].EnumName + ";";

      if (!Ints[IntNo].isOverloaded) {
        MatchTable.push_back(std::make_pair(Ints[IntNo].Name.substr(6),Result));
        continue;
      }

      // For overloaded intrinsics, only the prefix needs to match
      std::string TheStr = Ints[IntNo].Name.substr(6);
      TheStr += '.';  // Require "bswap." instead of bswap.
      OS << "    if (NameR.startswith(\"" << TheStr << "\")) "
         << Result << '\n';
    }
    
    // Emit the matcher logic for the fixed length strings.
    StringMatcher("NameR", MatchTable, OS).Emit(1);
    OS << "    break;  // end of '" << I->first << "' case.\n";
  }
  
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToNameTable(const std::vector<CodeGenIntrinsic> &Ints, 
                         raw_ostream &OS) {
  OS << "// Intrinsic ID to name table\n";
  OS << "#ifdef GET_INTRINSIC_NAME_TABLE\n";
  OS << "  // Note that entry #0 is the invalid intrinsic!\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    OS << "  \"" << Ints[i].Name << "\",\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToOverloadTable(const std::vector<CodeGenIntrinsic> &Ints, 
                         raw_ostream &OS) {
  OS << "// Intrinsic ID to overload bitset\n";
  OS << "#ifdef GET_INTRINSIC_OVERLOAD_TABLE\n";
  OS << "static const uint8_t OTable[] = {\n";
  OS << "  0";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    // Add one to the index so we emit a null bit for the invalid #0 intrinsic.
    if ((i+1)%8 == 0)
      OS << ",\n  0";
    if (Ints[i].isOverloaded)
      OS << " | (1<<" << (i+1)%8 << ')';
  }
  OS << "\n};\n\n";
  // OTable contains a true bit at the position if the intrinsic is overloaded.
  OS << "return (OTable[id/8] & (1 << (id%8))) != 0;\n";
  OS << "#endif\n\n";
}


// NOTE: This must be kept in synch with the copy in lib/VMCore/Function.cpp!
enum IIT_Info {
  // Common values should be encoded with 0-15.
  IIT_Done = 0,
  IIT_I1   = 1,
  IIT_I8   = 2,
  IIT_I16  = 3,
  IIT_I32  = 4,
  IIT_I64  = 5,
  IIT_F32  = 6,
  IIT_F64  = 7,
  IIT_V2   = 8,
  IIT_V4   = 9,
  IIT_V8   = 10,
  IIT_V16  = 11,
  IIT_V32  = 12,
  IIT_MMX  = 13,
  IIT_PTR  = 14,
  IIT_ARG  = 15,
  
  // Values from 16+ are only encodable with the inefficient encoding.
  IIT_METADATA = 16,
  IIT_EMPTYSTRUCT = 17,
  IIT_STRUCT2 = 18,
  IIT_STRUCT3 = 19,
  IIT_STRUCT4 = 20,
  IIT_STRUCT5 = 21,
  IIT_EXTEND_VEC_ARG = 22,
  IIT_TRUNC_VEC_ARG = 23,
  IIT_ANYPTR = 24
};


static void EncodeFixedValueType(MVT::SimpleValueType VT,
                                 std::vector<unsigned char> &Sig) {
  if (EVT(VT).isInteger()) {
    unsigned BitWidth = EVT(VT).getSizeInBits();
    switch (BitWidth) {
    default: PrintFatalError("unhandled integer type width in intrinsic!");
    case 1: return Sig.push_back(IIT_I1);
    case 8: return Sig.push_back(IIT_I8);
    case 16: return Sig.push_back(IIT_I16);
    case 32: return Sig.push_back(IIT_I32);
    case 64: return Sig.push_back(IIT_I64);
    }
  }
  
  switch (VT) {
  default: PrintFatalError("unhandled MVT in intrinsic!");
  case MVT::f32: return Sig.push_back(IIT_F32);
  case MVT::f64: return Sig.push_back(IIT_F64);
  case MVT::Metadata: return Sig.push_back(IIT_METADATA);
  case MVT::x86mmx: return Sig.push_back(IIT_MMX);
  // MVT::OtherVT is used to mean the empty struct type here.
  case MVT::Other: return Sig.push_back(IIT_EMPTYSTRUCT);
  }
}

#ifdef _MSC_VER
#pragma optimize("",off) // MSVC 2010 optimizer can't deal with this function.
#endif 

static void EncodeFixedType(Record *R, std::vector<unsigned char> &ArgCodes,
                            std::vector<unsigned char> &Sig) {
  
  if (R->isSubClassOf("LLVMMatchType")) {
    unsigned Number = R->getValueAsInt("Number");
    assert(Number < ArgCodes.size() && "Invalid matching number!");
    if (R->isSubClassOf("LLVMExtendedElementVectorType"))
      Sig.push_back(IIT_EXTEND_VEC_ARG);
    else if (R->isSubClassOf("LLVMTruncatedElementVectorType"))
      Sig.push_back(IIT_TRUNC_VEC_ARG);
    else
      Sig.push_back(IIT_ARG);
    return Sig.push_back((Number << 2) | ArgCodes[Number]);
  }
  
  MVT::SimpleValueType VT = getValueType(R->getValueAsDef("VT"));

  unsigned Tmp = 0;
  switch (VT) {
  default: break;
  case MVT::iPTRAny: ++Tmp; // FALL THROUGH.
  case MVT::vAny: ++Tmp; // FALL THROUGH.
  case MVT::fAny: ++Tmp; // FALL THROUGH.
  case MVT::iAny: {
    // If this is an "any" valuetype, then the type is the type of the next
    // type in the list specified to getIntrinsic().  
    Sig.push_back(IIT_ARG);
    
    // Figure out what arg # this is consuming, and remember what kind it was.
    unsigned ArgNo = ArgCodes.size();
    ArgCodes.push_back(Tmp);
    
    // Encode what sort of argument it must be in the low 2 bits of the ArgNo.
    return Sig.push_back((ArgNo << 2) | Tmp);
  }
  
  case MVT::iPTR: {
    unsigned AddrSpace = 0;
    if (R->isSubClassOf("LLVMQualPointerType")) {
      AddrSpace = R->getValueAsInt("AddrSpace");
      assert(AddrSpace < 256 && "Address space exceeds 255");
    }
    if (AddrSpace) {
      Sig.push_back(IIT_ANYPTR);
      Sig.push_back(AddrSpace);
    } else {
      Sig.push_back(IIT_PTR);
    }
    return EncodeFixedType(R->getValueAsDef("ElTy"), ArgCodes, Sig);
  }
  }
  
  if (EVT(VT).isVector()) {
    EVT VVT = VT;
    switch (VVT.getVectorNumElements()) {
    default: PrintFatalError("unhandled vector type width in intrinsic!");
    case 2: Sig.push_back(IIT_V2); break;
    case 4: Sig.push_back(IIT_V4); break;
    case 8: Sig.push_back(IIT_V8); break;
    case 16: Sig.push_back(IIT_V16); break;
    case 32: Sig.push_back(IIT_V32); break;
    }
    
    return EncodeFixedValueType(VVT.getVectorElementType().
                                getSimpleVT().SimpleTy, Sig);
  }

  EncodeFixedValueType(VT, Sig);
}

#ifdef _MSC_VER
#pragma optimize("",on)
#endif

/// ComputeFixedEncoding - If we can encode the type signature for this
/// intrinsic into 32 bits, return it.  If not, return ~0U.
static void ComputeFixedEncoding(const CodeGenIntrinsic &Int,
                                 std::vector<unsigned char> &TypeSig) {
  std::vector<unsigned char> ArgCodes;
  
  if (Int.IS.RetVTs.empty())
    TypeSig.push_back(IIT_Done);
  else if (Int.IS.RetVTs.size() == 1 &&
           Int.IS.RetVTs[0] == MVT::isVoid)
    TypeSig.push_back(IIT_Done);
  else {
    switch (Int.IS.RetVTs.size()) {
      case 1: break;
      case 2: TypeSig.push_back(IIT_STRUCT2); break;
      case 3: TypeSig.push_back(IIT_STRUCT3); break;
      case 4: TypeSig.push_back(IIT_STRUCT4); break;
      case 5: TypeSig.push_back(IIT_STRUCT5); break;
      default: assert(0 && "Unhandled case in struct");
    }
    
    for (unsigned i = 0, e = Int.IS.RetVTs.size(); i != e; ++i)
      EncodeFixedType(Int.IS.RetTypeDefs[i], ArgCodes, TypeSig);
  }
  
  for (unsigned i = 0, e = Int.IS.ParamTypeDefs.size(); i != e; ++i)
    EncodeFixedType(Int.IS.ParamTypeDefs[i], ArgCodes, TypeSig);
}

static void printIITEntry(raw_ostream &OS, unsigned char X) {
  OS << (unsigned)X;
}

void IntrinsicEmitter::EmitGenerator(const std::vector<CodeGenIntrinsic> &Ints, 
                                     raw_ostream &OS) {
  // If we can compute a 32-bit fixed encoding for this intrinsic, do so and
  // capture it in this vector, otherwise store a ~0U.
  std::vector<unsigned> FixedEncodings;
  
  SequenceToOffsetTable<std::vector<unsigned char> > LongEncodingTable;
  
  std::vector<unsigned char> TypeSig;
  
  // Compute the unique argument type info.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    // Get the signature for the intrinsic.
    TypeSig.clear();
    ComputeFixedEncoding(Ints[i], TypeSig);

    // Check to see if we can encode it into a 32-bit word.  We can only encode
    // 8 nibbles into a 32-bit word.
    if (TypeSig.size() <= 8) {
      bool Failed = false;
      unsigned Result = 0;
      for (unsigned i = 0, e = TypeSig.size(); i != e; ++i) {
        // If we had an unencodable argument, bail out.
        if (TypeSig[i] > 15) {
          Failed = true;
          break;
        }
        Result = (Result << 4) | TypeSig[e-i-1];
      }
      
      // If this could be encoded into a 31-bit word, return it.
      if (!Failed && (Result >> 31) == 0) {
        FixedEncodings.push_back(Result);
        continue;
      }
    }

    // Otherwise, we're going to unique the sequence into the
    // LongEncodingTable, and use its offset in the 32-bit table instead.
    LongEncodingTable.add(TypeSig);
      
    // This is a placehold that we'll replace after the table is laid out.
    FixedEncodings.push_back(~0U);
  }
  
  LongEncodingTable.layout();
  
  OS << "// Global intrinsic function declaration type table.\n";
  OS << "#ifdef GET_INTRINSIC_GENERATOR_GLOBAL\n";

  OS << "static const unsigned IIT_Table[] = {\n  ";
  
  for (unsigned i = 0, e = FixedEncodings.size(); i != e; ++i) {
    if ((i & 7) == 7)
      OS << "\n  ";
    
    // If the entry fit in the table, just emit it.
    if (FixedEncodings[i] != ~0U) {
      OS << "0x" << utohexstr(FixedEncodings[i]) << ", ";
      continue;
    }
    
    TypeSig.clear();
    ComputeFixedEncoding(Ints[i], TypeSig);

    
    // Otherwise, emit the offset into the long encoding table.  We emit it this
    // way so that it is easier to read the offset in the .def file.
    OS << "(1U<<31) | " << LongEncodingTable.get(TypeSig) << ", ";
  }
  
  OS << "0\n};\n\n";
  
  // Emit the shared table of register lists.
  OS << "static const unsigned char IIT_LongEncodingTable[] = {\n";
  if (!LongEncodingTable.empty())
    LongEncodingTable.emit(OS, printIITEntry);
  OS << "  255\n};\n\n";
  
  OS << "#endif\n\n";  // End of GET_INTRINSIC_GENERATOR_GLOBAL
}

enum ModRefKind {
  MRK_none,
  MRK_readonly,
  MRK_readnone
};

static ModRefKind getModRefKind(const CodeGenIntrinsic &intrinsic) {
  switch (intrinsic.ModRef) {
  case CodeGenIntrinsic::NoMem:
    return MRK_readnone;
  case CodeGenIntrinsic::ReadArgMem:
  case CodeGenIntrinsic::ReadMem:
    return MRK_readonly;
  case CodeGenIntrinsic::ReadWriteArgMem:
  case CodeGenIntrinsic::ReadWriteMem:
    return MRK_none;
  }
  llvm_unreachable("bad mod-ref kind");
}

namespace {
struct AttributeComparator {
  bool operator()(const CodeGenIntrinsic *L, const CodeGenIntrinsic *R) const {
    // Sort throwing intrinsics after non-throwing intrinsics.
    if (L->canThrow != R->canThrow)
      return R->canThrow;

    if (L->isNoReturn != R->isNoReturn)
      return R->isNoReturn;

    // Try to order by readonly/readnone attribute.
    ModRefKind LK = getModRefKind(*L);
    ModRefKind RK = getModRefKind(*R);
    if (LK != RK) return (LK > RK);

    // Order by argument attributes.
    // This is reliable because each side is already sorted internally.
    return (L->ArgumentAttributes < R->ArgumentAttributes);
  }
};
} // End anonymous namespace

/// EmitAttributes - This emits the Intrinsic::getAttributes method.
void IntrinsicEmitter::
EmitAttributes(const std::vector<CodeGenIntrinsic> &Ints, raw_ostream &OS) {
  OS << "// Add parameter attributes that are not common to all intrinsics.\n";
  OS << "#ifdef GET_INTRINSIC_ATTRIBUTES\n";
  if (TargetOnly)
    OS << "static AttrListPtr getAttributes(LLVMContext &C, " << TargetPrefix
       << "Intrinsic::ID id) {\n";
  else
    OS << "AttrListPtr Intrinsic::getAttributes(LLVMContext &C, ID id) {\n";

  // Compute the maximum number of attribute arguments and the map
  typedef std::map<const CodeGenIntrinsic*, unsigned,
                   AttributeComparator> UniqAttrMapTy;
  UniqAttrMapTy UniqAttributes;
  unsigned maxArgAttrs = 0;
  unsigned AttrNum = 0;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = Ints[i];
    maxArgAttrs =
      std::max(maxArgAttrs, unsigned(intrinsic.ArgumentAttributes.size()));
    unsigned &N = UniqAttributes[&intrinsic];
    if (N) continue;
    assert(AttrNum < 256 && "Too many unique attributes for table!");
    N = ++AttrNum;
  }

  // Emit an array of AttributeWithIndex.  Most intrinsics will have
  // at least one entry, for the function itself (index ~1), which is
  // usually nounwind.
  OS << "  static const uint8_t IntrinsicsToAttributesMap[] = {\n";

  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = Ints[i];

    OS << "    " << UniqAttributes[&intrinsic] << ", // "
       << intrinsic.Name << "\n";
  }
  OS << "  };\n\n";

  OS << "  AttributeWithIndex AWI[" << maxArgAttrs+1 << "];\n";
  OS << "  unsigned NumAttrs = 0;\n";
  OS << "  if (id != 0) {\n";
  OS << "    SmallVector<Attributes::AttrVal, 8> AttrVec;\n";
  OS << "    switch(IntrinsicsToAttributesMap[id - ";
  if (TargetOnly)
    OS << "Intrinsic::num_intrinsics";
  else
    OS << "1";
  OS << "]) {\n";
  OS << "    default: llvm_unreachable(\"Invalid attribute number\");\n";
  for (UniqAttrMapTy::const_iterator I = UniqAttributes.begin(),
       E = UniqAttributes.end(); I != E; ++I) {
    OS << "    case " << I->second << ":\n";

    const CodeGenIntrinsic &intrinsic = *(I->first);

    // Keep track of the number of attributes we're writing out.
    unsigned numAttrs = 0;

    // The argument attributes are alreadys sorted by argument index.
    unsigned ai = 0, ae = intrinsic.ArgumentAttributes.size();
    if (ae) {
      while (ai != ae) {
        unsigned argNo = intrinsic.ArgumentAttributes[ai].first;

        OS << "      AttrVec.clear();\n";

        do {
          switch (intrinsic.ArgumentAttributes[ai].second) {
          case CodeGenIntrinsic::NoCapture:
            OS << "      AttrVec.push_back(Attributes::NoCapture);\n";
            break;
          }

          ++ai;
        } while (ai != ae && intrinsic.ArgumentAttributes[ai].first == argNo);

        OS << "      AWI[" << numAttrs++ << "] = AttributeWithIndex::get(C, "
           << argNo+1 << ", AttrVec);\n";
      }
    }

    ModRefKind modRef = getModRefKind(intrinsic);

    if (!intrinsic.canThrow || modRef || intrinsic.isNoReturn) {
      OS << "      AttrVec.clear();\n";

      if (!intrinsic.canThrow)
        OS << "      AttrVec.push_back(Attributes::NoUnwind);\n";
      if (intrinsic.isNoReturn)
        OS << "      AttrVec.push_back(Attributes::NoReturn);\n";

      switch (modRef) {
      case MRK_none: break;
      case MRK_readonly:
        OS << "      AttrVec.push_back(Attributes::ReadOnly);\n";
        break;
      case MRK_readnone:
        OS << "      AttrVec.push_back(Attributes::ReadNone);\n"; 
        break;
      }
      OS << "      AWI[" << numAttrs++ << "] = AttributeWithIndex::get(C, "
         << "AttrListPtr::FunctionIndex, AttrVec);\n";
    }

    if (numAttrs) {
      OS << "      NumAttrs = " << numAttrs << ";\n";
      OS << "      break;\n";
    } else {
      OS << "      return AttrListPtr();\n";
    }
  }
  
  OS << "    }\n";
  OS << "  }\n";
  OS << "  return AttrListPtr::get(C, ArrayRef<AttributeWithIndex>(AWI, "
             "NumAttrs));\n";
  OS << "}\n";
  OS << "#endif // GET_INTRINSIC_ATTRIBUTES\n\n";
}

/// EmitModRefBehavior - Determine intrinsic alias analysis mod/ref behavior.
void IntrinsicEmitter::
EmitModRefBehavior(const std::vector<CodeGenIntrinsic> &Ints, raw_ostream &OS){
  OS << "// Determine intrinsic alias analysis mod/ref behavior.\n"
     << "#ifdef GET_INTRINSIC_MODREF_BEHAVIOR\n"
     << "assert(iid <= Intrinsic::" << Ints.back().EnumName << " && "
     << "\"Unknown intrinsic.\");\n\n";

  OS << "static const uint8_t IntrinsicModRefBehavior[] = {\n"
     << "  /* invalid */ UnknownModRefBehavior,\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    OS << "  /* " << TargetPrefix << Ints[i].EnumName << " */ ";
    switch (Ints[i].ModRef) {
    case CodeGenIntrinsic::NoMem:
      OS << "DoesNotAccessMemory,\n";
      break;
    case CodeGenIntrinsic::ReadArgMem:
      OS << "OnlyReadsArgumentPointees,\n";
      break;
    case CodeGenIntrinsic::ReadMem:
      OS << "OnlyReadsMemory,\n";
      break;
    case CodeGenIntrinsic::ReadWriteArgMem:
      OS << "OnlyAccessesArgumentPointees,\n";
      break;
    case CodeGenIntrinsic::ReadWriteMem:
      OS << "UnknownModRefBehavior,\n";
      break;
    }
  }
  OS << "};\n\n"
     << "return static_cast<ModRefBehavior>(IntrinsicModRefBehavior[iid]);\n"
     << "#endif // GET_INTRINSIC_MODREF_BEHAVIOR\n\n";
}

/// EmitTargetBuiltins - All of the builtins in the specified map are for the
/// same target, and we already checked it.
static void EmitTargetBuiltins(const std::map<std::string, std::string> &BIM,
                               const std::string &TargetPrefix,
                               raw_ostream &OS) {
  
  std::vector<StringMatcher::StringPair> Results;
  
  for (std::map<std::string, std::string>::const_iterator I = BIM.begin(),
       E = BIM.end(); I != E; ++I) {
    std::string ResultCode =
    "return " + TargetPrefix + "Intrinsic::" + I->second + ";";
    Results.push_back(StringMatcher::StringPair(I->first, ResultCode));
  }

  StringMatcher("BuiltinName", Results, OS).Emit();
}

        
void IntrinsicEmitter::
EmitIntrinsicToGCCBuiltinMap(const std::vector<CodeGenIntrinsic> &Ints, 
                             raw_ostream &OS) {
  typedef std::map<std::string, std::map<std::string, std::string> > BIMTy;
  BIMTy BuiltinMap;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      // Get the map for this target prefix.
      std::map<std::string, std::string> &BIM =BuiltinMap[Ints[i].TargetPrefix];
      
      if (!BIM.insert(std::make_pair(Ints[i].GCCBuiltinName,
                                     Ints[i].EnumName)).second)
        PrintFatalError("Intrinsic '" + Ints[i].TheDef->getName() +
              "': duplicate GCC builtin name!");
    }
  }
  
  OS << "// Get the LLVM intrinsic that corresponds to a GCC builtin.\n";
  OS << "// This is used by the C front-end.  The GCC builtin name is passed\n";
  OS << "// in as BuiltinName, and a target prefix (e.g. 'ppc') is passed\n";
  OS << "// in as TargetPrefix.  The result is assigned to 'IntrinsicID'.\n";
  OS << "#ifdef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN\n";
  
  if (TargetOnly) {
    OS << "static " << TargetPrefix << "Intrinsic::ID "
       << "getIntrinsicForGCCBuiltin(const char "
       << "*TargetPrefixStr, const char *BuiltinNameStr) {\n";
  } else {
    OS << "Intrinsic::ID Intrinsic::getIntrinsicForGCCBuiltin(const char "
       << "*TargetPrefixStr, const char *BuiltinNameStr) {\n";
  }
  
  OS << "  StringRef BuiltinName(BuiltinNameStr);\n";
  OS << "  StringRef TargetPrefix(TargetPrefixStr);\n\n";
  
  // Note: this could emit significantly better code if we cared.
  for (BIMTy::iterator I = BuiltinMap.begin(), E = BuiltinMap.end();I != E;++I){
    OS << "  ";
    if (!I->first.empty())
      OS << "if (TargetPrefix == \"" << I->first << "\") ";
    else
      OS << "/* Target Independent Builtins */ ";
    OS << "{\n";

    // Emit the comparisons for this target prefix.
    EmitTargetBuiltins(I->second, TargetPrefix, OS);
    OS << "  }\n";
  }
  OS << "  return ";
  if (!TargetPrefix.empty())
    OS << "(" << TargetPrefix << "Intrinsic::ID)";
  OS << "Intrinsic::not_intrinsic;\n";
  OS << "}\n";
  OS << "#endif\n\n";
}

namespace llvm {

void EmitIntrinsics(RecordKeeper &RK, raw_ostream &OS, bool TargetOnly = false) {
  IntrinsicEmitter(RK, TargetOnly).run(OS);
}

} // End llvm namespace
