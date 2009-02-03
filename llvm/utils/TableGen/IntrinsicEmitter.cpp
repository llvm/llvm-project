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

#include "CodeGenTarget.h"
#include "IntrinsicEmitter.h"
#include "Record.h"
#include "llvm/ADT/StringExtras.h"
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
// IntrinsicEmitter Implementation
//===----------------------------------------------------------------------===//

void IntrinsicEmitter::run(std::ostream &OS) {
  EmitSourceFileHeader("Intrinsic Function Source Fragment", OS);
  
  std::vector<CodeGenIntrinsic> Ints = LoadIntrinsics(Records);

  // Emit the enum information.
  EmitEnumInfo(Ints, OS);

  // Emit the intrinsic ID -> name table.
  EmitIntrinsicToNameTable(Ints, OS);
  
  // Emit the function name recognizer.
  EmitFnNameRecognizer(Ints, OS);
  
  // Emit the intrinsic verifier.
  EmitVerifier(Ints, OS);
  
  // Emit the intrinsic declaration generator.
  EmitGenerator(Ints, OS);
  
  // Emit the intrinsic parameter attributes.
  EmitAttributes(Ints, OS);

  // Emit a list of intrinsics with corresponding GCC builtins.
  EmitGCCBuiltinList(Ints, OS);

  // Emit code to translate GCC builtins into LLVM intrinsics.
  EmitIntrinsicToGCCBuiltinMap(Ints, OS);
}

void IntrinsicEmitter::EmitEnumInfo(const std::vector<CodeGenIntrinsic> &Ints,
                                    std::ostream &OS) {
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
                     std::ostream &OS) {
  // Build a function name -> intrinsic name mapping.
  std::map<std::string, unsigned> IntMapping;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    IntMapping[Ints[i].Name] = i;
    
  OS << "// Function name -> enum value recognizer code.\n";
  OS << "#ifdef GET_FUNCTION_RECOGNIZER\n";
  OS << "  switch (Name[5]) {\n";
  OS << "  default:\n";
  // Emit the intrinsics in sorted order.
  char LastChar = 0;
  for (std::map<std::string, unsigned>::iterator I = IntMapping.begin(),
       E = IntMapping.end(); I != E; ++I) {
    if (I->first[5] != LastChar) {
      LastChar = I->first[5];
      OS << "    break;\n";
      OS << "  case '" << LastChar << "':\n";
    }
    
    // For overloaded intrinsics, only the prefix needs to match
    if (Ints[I->second].isOverloaded)
      OS << "    if (Len > " << I->first.size()
       << " && !memcmp(Name, \"" << I->first << ".\", "
       << (I->first.size() + 1) << ")) return Intrinsic::"
       << Ints[I->second].EnumName << ";\n";
    else 
      OS << "    if (Len == " << I->first.size()
         << " && !memcmp(Name, \"" << I->first << "\", "
         << I->first.size() << ")) return Intrinsic::"
         << Ints[I->second].EnumName << ";\n";
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::
EmitIntrinsicToNameTable(const std::vector<CodeGenIntrinsic> &Ints, 
                         std::ostream &OS) {
  OS << "// Intrinsic ID to name table\n";
  OS << "#ifdef GET_INTRINSIC_NAME_TABLE\n";
  OS << "  // Note that entry #0 is the invalid intrinsic!\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    OS << "  \"" << Ints[i].Name << "\",\n";
  OS << "#endif\n\n";
}

static void EmitTypeForValueType(std::ostream &OS, MVT::SimpleValueType VT) {
  if (MVT(VT).isInteger()) {
    unsigned BitWidth = MVT(VT).getSizeInBits();
    OS << "IntegerType::get(" << BitWidth << ")";
  } else if (VT == MVT::Other) {
    // MVT::OtherVT is used to mean the empty struct type here.
    OS << "StructType::get(std::vector<const Type *>())";
  } else if (VT == MVT::f32) {
    OS << "Type::FloatTy";
  } else if (VT == MVT::f64) {
    OS << "Type::DoubleTy";
  } else if (VT == MVT::f80) {
    OS << "Type::X86_FP80Ty";
  } else if (VT == MVT::f128) {
    OS << "Type::FP128Ty";
  } else if (VT == MVT::ppcf128) {
    OS << "Type::PPC_FP128Ty";
  } else if (VT == MVT::isVoid) {
    OS << "Type::VoidTy";
  } else {
    assert(false && "Unsupported ValueType!");
  }
}

static void EmitTypeGenerate(std::ostream &OS, const Record *ArgType,
                             unsigned &ArgNo);

static void EmitTypeGenerate(std::ostream &OS,
                             const std::vector<Record*> &ArgTypes,
                             unsigned &ArgNo) {
  if (ArgTypes.size() == 1) {
    EmitTypeGenerate(OS, ArgTypes.front(), ArgNo);
    return;
  }

  OS << "StructType::get(";

  for (std::vector<Record*>::const_iterator
         I = ArgTypes.begin(), E = ArgTypes.end(); I != E; ++I) {
    EmitTypeGenerate(OS, *I, ArgNo);
    OS << ", ";
  }

  OS << " NULL)";
}

static void EmitTypeGenerate(std::ostream &OS, const Record *ArgType,
                             unsigned &ArgNo) {
  MVT::SimpleValueType VT = getValueType(ArgType->getValueAsDef("VT"));

  if (ArgType->isSubClassOf("LLVMMatchType")) {
    unsigned Number = ArgType->getValueAsInt("Number");
    assert(Number < ArgNo && "Invalid matching number!");
    if (ArgType->isSubClassOf("LLVMExtendedElementVectorType"))
      OS << "VectorType::getExtendedElementVectorType"
         << "(dyn_cast<VectorType>(Tys[" << Number << "]))";
    else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
      OS << "VectorType::getTruncatedElementVectorType"
         << "(dyn_cast<VectorType>(Tys[" << Number << "]))";
    else
      OS << "Tys[" << Number << "]";
  } else if (VT == MVT::iAny || VT == MVT::fAny) {
    // NOTE: The ArgNo variable here is not the absolute argument number, it is
    // the index of the "arbitrary" type in the Tys array passed to the
    // Intrinsic::getDeclaration function. Consequently, we only want to
    // increment it when we actually hit an overloaded type. Getting this wrong
    // leads to very subtle bugs!
    OS << "Tys[" << ArgNo++ << "]";
  } else if (MVT(VT).isVector()) {
    MVT VVT = VT;
    OS << "VectorType::get(";
    EmitTypeForValueType(OS, VVT.getVectorElementType().getSimpleVT());
    OS << ", " << VVT.getVectorNumElements() << ")";
  } else if (VT == MVT::iPTR) {
    OS << "PointerType::getUnqual(";
    EmitTypeGenerate(OS, ArgType->getValueAsDef("ElTy"), ArgNo);
    OS << ")";
  } else if (VT == MVT::iPTRAny) {
    // Make sure the user has passed us an argument type to overload. If not,
    // treat it as an ordinary (not overloaded) intrinsic.
    OS << "(" << ArgNo << " < numTys) ? Tys[" << ArgNo 
    << "] : PointerType::getUnqual(";
    EmitTypeGenerate(OS, ArgType->getValueAsDef("ElTy"), ArgNo);
    OS << ")";
    ++ArgNo;
  } else if (VT == MVT::isVoid) {
    if (ArgNo == 0)
      OS << "Type::VoidTy";
    else
      // MVT::isVoid is used to mean varargs here.
      OS << "...";
  } else {
    EmitTypeForValueType(OS, VT);
  }
}

/// RecordListComparator - Provide a determinstic comparator for lists of
/// records.
namespace {
  typedef std::pair<std::vector<Record*>, std::vector<Record*> > RecPair;
  struct RecordListComparator {
    bool operator()(const RecPair &LHS,
                    const RecPair &RHS) const {
      unsigned i = 0;
      const std::vector<Record*> *LHSVec = &LHS.first;
      const std::vector<Record*> *RHSVec = &RHS.first;
      unsigned RHSSize = RHSVec->size();
      unsigned LHSSize = LHSVec->size();

      do {
        if (i == RHSSize) return false;  // RHS is shorter than LHS.
        if ((*LHSVec)[i] != (*RHSVec)[i])
          return (*LHSVec)[i]->getName() < (*RHSVec)[i]->getName();
      } while (++i != LHSSize);

      if (i != RHSSize) return true;

      i = 0;
      LHSVec = &LHS.second;
      RHSVec = &RHS.second;
      RHSSize = RHSVec->size();
      LHSSize = LHSVec->size();

      for (i = 0; i != LHSSize; ++i) {
        if (i == RHSSize) return false;  // RHS is shorter than LHS.
        if ((*LHSVec)[i] != (*RHSVec)[i])
          return (*LHSVec)[i]->getName() < (*RHSVec)[i]->getName();
      }

      return i != RHSSize;
    }
  };
}

void IntrinsicEmitter::EmitVerifier(const std::vector<CodeGenIntrinsic> &Ints, 
                                    std::ostream &OS) {
  OS << "// Verifier::visitIntrinsicFunctionCall code.\n";
  OS << "#ifdef GET_INTRINSIC_VERIFIER\n";
  OS << "  switch (ID) {\n";
  OS << "  default: assert(0 && \"Invalid intrinsic!\");\n";
  
  // This checking can emit a lot of very common code.  To reduce the amount of
  // code that we emit, batch up cases that have identical types.  This avoids
  // problems where GCC can run out of memory compiling Verifier.cpp.
  typedef std::map<RecPair, std::vector<unsigned>, RecordListComparator> MapTy;
  MapTy UniqueArgInfos;
  
  // Compute the unique argument type info.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    UniqueArgInfos[make_pair(Ints[i].IS.RetTypeDefs,
                             Ints[i].IS.ParamTypeDefs)].push_back(i);

  // Loop through the array, emitting one comparison for each batch.
  for (MapTy::iterator I = UniqueArgInfos.begin(),
       E = UniqueArgInfos.end(); I != E; ++I) {
    for (unsigned i = 0, e = I->second.size(); i != e; ++i)
      OS << "  case Intrinsic::" << Ints[I->second[i]].EnumName << ":\t\t// "
         << Ints[I->second[i]].Name << "\n";
    
    const RecPair &ArgTypes = I->first;
    const std::vector<Record*> &RetTys = ArgTypes.first;
    const std::vector<Record*> &ParamTys = ArgTypes.second;

    OS << "    VerifyIntrinsicPrototype(ID, IF, " << RetTys.size() << ", "
       << ParamTys.size();

    // Emit return types.
    for (unsigned j = 0, je = RetTys.size(); j != je; ++j) {
      Record *ArgType = RetTys[j];
      OS << ", ";

      if (ArgType->isSubClassOf("LLVMMatchType")) {
        unsigned Number = ArgType->getValueAsInt("Number");
        assert(Number < j && "Invalid matching number!");
        if (ArgType->isSubClassOf("LLVMExtendedElementVectorType"))
          OS << "~(ExtendedElementVectorType | " << Number << ")";
        else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
          OS << "~(TruncatedElementVectorType | " << Number << ")";
        else
          OS << "~" << Number;
      } else {
        MVT::SimpleValueType VT = getValueType(ArgType->getValueAsDef("VT"));
        OS << getEnumName(VT);

        if (VT == MVT::isVoid && j != 0 && j != je - 1)
          throw "Var arg type not last argument";
      }
    }

    // Emit the parameter types.
    for (unsigned j = 0, je = ParamTys.size(); j != je; ++j) {
      Record *ArgType = ParamTys[j];
      OS << ", ";

      if (ArgType->isSubClassOf("LLVMMatchType")) {
        unsigned Number = ArgType->getValueAsInt("Number");
        assert(Number < j + RetTys.size() && "Invalid matching number!");
        if (ArgType->isSubClassOf("LLVMExtendedElementVectorType"))
          OS << "~(ExtendedElementVectorType | " << Number << ")";
        else if (ArgType->isSubClassOf("LLVMTruncatedElementVectorType"))
          OS << "~(TruncatedElementVectorType | " << Number << ")";
        else
          OS << "~" << Number;
      } else {
        MVT::SimpleValueType VT = getValueType(ArgType->getValueAsDef("VT"));
        OS << getEnumName(VT);

        if (VT == MVT::isVoid && j != 0 && j != je - 1)
          throw "Var arg type not last argument";
      }
    }
      
    OS << ");\n";
    OS << "    break;\n";
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitGenerator(const std::vector<CodeGenIntrinsic> &Ints, 
                                     std::ostream &OS) {
  OS << "// Code for generating Intrinsic function declarations.\n";
  OS << "#ifdef GET_INTRINSIC_GENERATOR\n";
  OS << "  switch (id) {\n";
  OS << "  default: assert(0 && \"Invalid intrinsic!\");\n";
  
  // Similar to GET_INTRINSIC_VERIFIER, batch up cases that have identical
  // types.
  typedef std::map<RecPair, std::vector<unsigned>, RecordListComparator> MapTy;
  MapTy UniqueArgInfos;
  
  // Compute the unique argument type info.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    UniqueArgInfos[make_pair(Ints[i].IS.RetTypeDefs,
                             Ints[i].IS.ParamTypeDefs)].push_back(i);

  // Loop through the array, emitting one generator for each batch.
  for (MapTy::iterator I = UniqueArgInfos.begin(),
       E = UniqueArgInfos.end(); I != E; ++I) {
    for (unsigned i = 0, e = I->second.size(); i != e; ++i)
      OS << "  case Intrinsic::" << Ints[I->second[i]].EnumName << ":\t\t// "
         << Ints[I->second[i]].Name << "\n";
    
    const RecPair &ArgTypes = I->first;
    const std::vector<Record*> &RetTys = ArgTypes.first;
    const std::vector<Record*> &ParamTys = ArgTypes.second;

    unsigned N = ParamTys.size();

    if (N > 1 &&
        getValueType(ParamTys[N - 1]->getValueAsDef("VT")) == MVT::isVoid) {
      OS << "    IsVarArg = true;\n";
      --N;
    }

    unsigned ArgNo = 0;
    OS << "    ResultTy = ";
    EmitTypeGenerate(OS, RetTys, ArgNo);
    OS << ";\n";
    
    for (unsigned j = 0; j != N; ++j) {
      OS << "    ArgTys.push_back(";
      EmitTypeGenerate(OS, ParamTys[j], ArgNo);
      OS << ");\n";
    }

    OS << "    break;\n";
  }

  OS << "  }\n";
  OS << "#endif\n\n";
}

/// EmitAttributes - This emits the Intrinsic::getAttributes method.
void IntrinsicEmitter::
EmitAttributes(const std::vector<CodeGenIntrinsic> &Ints, std::ostream &OS) {
  OS << "// Add parameter attributes that are not common to all intrinsics.\n";
  OS << "#ifdef GET_INTRINSIC_ATTRIBUTES\n";
  OS << "AttrListPtr Intrinsic::getAttributes(ID id) {";
  OS << "  // No intrinsic can throw exceptions.\n";
  OS << "  Attributes Attr = Attribute::NoUnwind;\n";
  OS << "  switch (id) {\n";
  OS << "  default: break;\n";
  unsigned MaxArgAttrs = 0;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    MaxArgAttrs =
      std::max(MaxArgAttrs, unsigned(Ints[i].ArgumentAttributes.size()));
    switch (Ints[i].ModRef) {
    default: break;
    case CodeGenIntrinsic::NoMem:
      OS << "  case Intrinsic::" << Ints[i].EnumName << ":\n";
      break;
    }
  }
  OS << "    Attr |= Attribute::ReadNone; // These do not access memory.\n";
  OS << "    break;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    switch (Ints[i].ModRef) {
    default: break;
    case CodeGenIntrinsic::ReadArgMem:
    case CodeGenIntrinsic::ReadMem:
      OS << "  case Intrinsic::" << Ints[i].EnumName << ":\n";
      break;
    }
  }
  OS << "    Attr |= Attribute::ReadOnly; // These do not write memory.\n";
  OS << "    break;\n";
  OS << "  }\n";
  OS << "  AttributeWithIndex AWI[" << MaxArgAttrs+1 << "];\n";
  OS << "  unsigned NumAttrs = 0;\n";
  OS << "  switch (id) {\n";
  OS << "  default: break;\n";
  
  // Add argument attributes for any intrinsics that have them.
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (Ints[i].ArgumentAttributes.empty()) continue;
    
    OS << "  case Intrinsic::" << Ints[i].EnumName << ":\n";

    std::vector<std::pair<unsigned, CodeGenIntrinsic::ArgAttribute> > ArgAttrs =
      Ints[i].ArgumentAttributes;
    // Sort by argument index.
    std::sort(ArgAttrs.begin(), ArgAttrs.end());

    unsigned NumArgsWithAttrs = 0;

    while (!ArgAttrs.empty()) {
      unsigned ArgNo = ArgAttrs[0].first;
      
      OS << "    AWI[" << NumArgsWithAttrs++ << "] = AttributeWithIndex::get("
         << ArgNo+1 << ", 0";

      while (!ArgAttrs.empty() && ArgAttrs[0].first == ArgNo) {
        switch (ArgAttrs[0].second) {
        default: assert(0 && "Unknown arg attribute");
        case CodeGenIntrinsic::NoCapture:
          OS << "|Attribute::NoCapture";
          break;
        }
        ArgAttrs.erase(ArgAttrs.begin());
      }
      OS << ");\n";
    }
    
    OS << "    NumAttrs = " << NumArgsWithAttrs << ";\n";
    OS << "    break;\n";
  }
  
  OS << "  }\n";
  OS << "  AWI[NumAttrs] = AttributeWithIndex::get(~0, Attr);\n";
  OS << "  return AttrListPtr::get(AWI, NumAttrs+1);\n";
  OS << "}\n";
  OS << "#endif // GET_INTRINSIC_ATTRIBUTES\n\n";
}

void IntrinsicEmitter::
EmitGCCBuiltinList(const std::vector<CodeGenIntrinsic> &Ints, std::ostream &OS){
  OS << "// Get the GCC builtin that corresponds to an LLVM intrinsic.\n";
  OS << "#ifdef GET_GCC_BUILTIN_NAME\n";
  OS << "  switch (F->getIntrinsicID()) {\n";
  OS << "  default: BuiltinName = \"\"; break;\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      OS << "  case Intrinsic::" << Ints[i].EnumName << ": BuiltinName = \""
         << Ints[i].GCCBuiltinName << "\"; break;\n";
    }
  }
  OS << "  }\n";
  OS << "#endif\n\n";
}

/// EmitBuiltinComparisons - Emit comparisons to determine whether the specified
/// sorted range of builtin names is equal to the current builtin.  This breaks
/// it down into a simple tree.
///
/// At this point, we know that all the builtins in the range have the same name
/// for the first 'CharStart' characters.  Only the end of the name needs to be
/// discriminated.
typedef std::map<std::string, std::string>::const_iterator StrMapIterator;
static void EmitBuiltinComparisons(StrMapIterator Start, StrMapIterator End,
                                   unsigned CharStart, unsigned Indent,
                                   std::ostream &OS) {
  if (Start == End) return; // empty range.
  
  // Determine what, if anything, is the same about all these strings.
  std::string CommonString = Start->first;
  unsigned NumInRange = 0;
  for (StrMapIterator I = Start; I != End; ++I, ++NumInRange) {
    // Find the first character that doesn't match.
    const std::string &ThisStr = I->first;
    unsigned NonMatchChar = CharStart;
    while (NonMatchChar < CommonString.size() && 
           NonMatchChar < ThisStr.size() &&
           CommonString[NonMatchChar] == ThisStr[NonMatchChar])
      ++NonMatchChar;
    // Truncate off pieces that don't match.
    CommonString.resize(NonMatchChar);
  }
  
  // Just compare the rest of the string.
  if (NumInRange == 1) {
    if (CharStart != CommonString.size()) {
      OS << std::string(Indent*2, ' ') << "if (!memcmp(BuiltinName";
      if (CharStart) OS << "+" << CharStart;
      OS << ", \"" << (CommonString.c_str()+CharStart) << "\", ";
      OS << CommonString.size() - CharStart << "))\n";
      ++Indent;
    }
    OS << std::string(Indent*2, ' ') << "IntrinsicID = Intrinsic::";
    OS << Start->second << ";\n";
    return;
  }

  // At this point, we potentially have a common prefix for these builtins, emit
  // a check for this common prefix.
  if (CommonString.size() != CharStart) {
    OS << std::string(Indent*2, ' ') << "if (!memcmp(BuiltinName";
    if (CharStart) OS << "+" << CharStart;
    OS << ", \"" << (CommonString.c_str()+CharStart) << "\", ";
    OS << CommonString.size()-CharStart << ")) {\n";
    
    EmitBuiltinComparisons(Start, End, CommonString.size(), Indent+1, OS);
    OS << std::string(Indent*2, ' ') << "}\n";
    return;
  }
  
  // Output a switch on the character that differs across the set.
  OS << std::string(Indent*2, ' ') << "switch (BuiltinName[" << CharStart
      << "]) {";
  if (CharStart)
    OS << "  // \"" << std::string(Start->first.begin(), 
                                   Start->first.begin()+CharStart) << "\"";
  OS << "\n";
  
  for (StrMapIterator I = Start; I != End; ) {
    char ThisChar = I->first[CharStart];
    OS << std::string(Indent*2, ' ') << "case '" << ThisChar << "':\n";
    // Figure out the range that has this common character.
    StrMapIterator NextChar = I;
    for (++NextChar; NextChar != End && NextChar->first[CharStart] == ThisChar;
         ++NextChar)
      /*empty*/;
    EmitBuiltinComparisons(I, NextChar, CharStart+1, Indent+1, OS);
    OS << std::string(Indent*2, ' ') << "  break;\n";
    I = NextChar;
  }
  OS << std::string(Indent*2, ' ') << "}\n";
}

/// EmitTargetBuiltins - All of the builtins in the specified map are for the
/// same target, and we already checked it.
static void EmitTargetBuiltins(const std::map<std::string, std::string> &BIM,
                               std::ostream &OS) {
  // Rearrange the builtins by length.
  std::vector<std::map<std::string, std::string> > BuiltinsByLen;
  BuiltinsByLen.reserve(100);
  
  for (StrMapIterator I = BIM.begin(), E = BIM.end(); I != E; ++I) {
    if (I->first.size() >= BuiltinsByLen.size())
      BuiltinsByLen.resize(I->first.size()+1);
    BuiltinsByLen[I->first.size()].insert(*I);
  }
  
  // Now that we have all the builtins by their length, emit a switch stmt.
  OS << "    switch (strlen(BuiltinName)) {\n";
  OS << "    default: break;\n";
  for (unsigned i = 0, e = BuiltinsByLen.size(); i != e; ++i) {
    if (BuiltinsByLen[i].empty()) continue;
    OS << "    case " << i << ":\n";
    EmitBuiltinComparisons(BuiltinsByLen[i].begin(), BuiltinsByLen[i].end(),
                           0, 3, OS);
    OS << "      break;\n";
  }
  OS << "    }\n";
}

        
void IntrinsicEmitter::
EmitIntrinsicToGCCBuiltinMap(const std::vector<CodeGenIntrinsic> &Ints, 
                             std::ostream &OS) {
  typedef std::map<std::string, std::map<std::string, std::string> > BIMTy;
  BIMTy BuiltinMap;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    if (!Ints[i].GCCBuiltinName.empty()) {
      // Get the map for this target prefix.
      std::map<std::string, std::string> &BIM =BuiltinMap[Ints[i].TargetPrefix];
      
      if (!BIM.insert(std::make_pair(Ints[i].GCCBuiltinName,
                                     Ints[i].EnumName)).second)
        throw "Intrinsic '" + Ints[i].TheDef->getName() +
              "': duplicate GCC builtin name!";
    }
  }
  
  OS << "// Get the LLVM intrinsic that corresponds to a GCC builtin.\n";
  OS << "// This is used by the C front-end.  The GCC builtin name is passed\n";
  OS << "// in as BuiltinName, and a target prefix (e.g. 'ppc') is passed\n";
  OS << "// in as TargetPrefix.  The result is assigned to 'IntrinsicID'.\n";
  OS << "#ifdef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN\n";
  OS << "  IntrinsicID = Intrinsic::not_intrinsic;\n";
  
  // Note: this could emit significantly better code if we cared.
  for (BIMTy::iterator I = BuiltinMap.begin(), E = BuiltinMap.end();I != E;++I){
    OS << "  ";
    if (!I->first.empty())
      OS << "if (!strcmp(TargetPrefix, \"" << I->first << "\")) ";
    else
      OS << "/* Target Independent Builtins */ ";
    OS << "{\n";

    // Emit the comparisons for this target prefix.
    EmitTargetBuiltins(I->second, OS);
    OS << "  }\n";
  }
  OS << "#endif\n\n";
}
