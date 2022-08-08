//===- IntrinsicEmitter.cpp - Generate intrinsic information --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend emits information about intrinsic functions.
//
//===----------------------------------------------------------------------===//

#include "CodeGenIntrinsics.h"
#include "CodeGenTarget.h"
#include "SequenceToOffsetTable.h"
#include "TableGenBackends.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
using namespace llvm;

cl::OptionCategory GenIntrinsicCat("Options for -gen-intrinsic-enums");
cl::opt<std::string>
    IntrinsicPrefix("intrinsic-prefix",
                    cl::desc("Generate intrinsics with this target prefix"),
                    cl::value_desc("target prefix"), cl::cat(GenIntrinsicCat));

namespace {
class IntrinsicEmitter {
  RecordKeeper &Records;

public:
  IntrinsicEmitter(RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS, bool Enums);

  void EmitEnumInfo(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitTargetInfo(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitIntrinsicToNameTable(const CodeGenIntrinsicTable &Ints,
                                raw_ostream &OS);
  void EmitIntrinsicToOverloadTable(const CodeGenIntrinsicTable &Ints,
                                    raw_ostream &OS);
  void EmitGenerator(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitAttributes(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitIntrinsicToBuiltinMap(const CodeGenIntrinsicTable &Ints, bool IsClang,
                                 raw_ostream &OS);
};
} // End anonymous namespace

//===----------------------------------------------------------------------===//
// IntrinsicEmitter Implementation
//===----------------------------------------------------------------------===//

void IntrinsicEmitter::run(raw_ostream &OS, bool Enums) {
  emitSourceFileHeader("Intrinsic Function Source Fragment", OS);

  CodeGenIntrinsicTable Ints(Records);

  if (Enums) {
    // Emit the enum information.
    EmitEnumInfo(Ints, OS);
  } else {
    // Emit the target metadata.
    EmitTargetInfo(Ints, OS);

    // Emit the intrinsic ID -> name table.
    EmitIntrinsicToNameTable(Ints, OS);

    // Emit the intrinsic ID -> overload table.
    EmitIntrinsicToOverloadTable(Ints, OS);

    // Emit the intrinsic declaration generator.
    EmitGenerator(Ints, OS);

    // Emit the intrinsic parameter attributes.
    EmitAttributes(Ints, OS);

    // Emit code to translate GCC builtins into LLVM intrinsics.
    EmitIntrinsicToBuiltinMap(Ints, true, OS);

    // Emit code to translate MS builtins into LLVM intrinsics.
    EmitIntrinsicToBuiltinMap(Ints, false, OS);
  }
}

void IntrinsicEmitter::EmitEnumInfo(const CodeGenIntrinsicTable &Ints,
                                    raw_ostream &OS) {
  // Find the TargetSet for which to generate enums. There will be an initial
  // set with an empty target prefix which will include target independent
  // intrinsics like dbg.value.
  const CodeGenIntrinsicTable::TargetSet *Set = nullptr;
  for (const auto &Target : Ints.Targets) {
    if (Target.Name == IntrinsicPrefix) {
      Set = &Target;
      break;
    }
  }
  if (!Set) {
    std::vector<std::string> KnownTargets;
    for (const auto &Target : Ints.Targets)
      if (!Target.Name.empty())
        KnownTargets.push_back(Target.Name);
    PrintFatalError("tried to generate intrinsics for unknown target " +
                    IntrinsicPrefix +
                    "\nKnown targets are: " + join(KnownTargets, ", ") + "\n");
  }

  // Generate a complete header for target specific intrinsics.
  if (!IntrinsicPrefix.empty()) {
    std::string UpperPrefix = StringRef(IntrinsicPrefix).upper();
    OS << "#ifndef LLVM_IR_INTRINSIC_" << UpperPrefix << "_ENUMS_H\n";
    OS << "#define LLVM_IR_INTRINSIC_" << UpperPrefix << "_ENUMS_H\n\n";
    OS << "namespace llvm {\n";
    OS << "namespace Intrinsic {\n";
    OS << "enum " << UpperPrefix << "Intrinsics : unsigned {\n";
  }

  OS << "// Enum values for intrinsics\n";
  for (unsigned i = Set->Offset, e = Set->Offset + Set->Count; i != e; ++i) {
    OS << "    " << Ints[i].EnumName;

    // Assign a value to the first intrinsic in this target set so that all
    // intrinsic ids are distinct.
    if (i == Set->Offset)
      OS << " = " << (Set->Offset + 1);

    OS << ", ";
    if (Ints[i].EnumName.size() < 40)
      OS.indent(40 - Ints[i].EnumName.size());
    OS << " // " << Ints[i].Name << "\n";
  }

  // Emit num_intrinsics into the target neutral enum.
  if (IntrinsicPrefix.empty()) {
    OS << "    num_intrinsics = " << (Ints.size() + 1) << "\n";
  } else {
    OS << "}; // enum\n";
    OS << "} // namespace Intrinsic\n";
    OS << "} // namespace llvm\n\n";
    OS << "#endif\n";
  }
}

void IntrinsicEmitter::EmitTargetInfo(const CodeGenIntrinsicTable &Ints,
                                    raw_ostream &OS) {
  OS << "// Target mapping\n";
  OS << "#ifdef GET_INTRINSIC_TARGET_DATA\n";
  OS << "struct IntrinsicTargetInfo {\n"
     << "  llvm::StringLiteral Name;\n"
     << "  size_t Offset;\n"
     << "  size_t Count;\n"
     << "};\n";
  OS << "static constexpr IntrinsicTargetInfo TargetInfos[] = {\n";
  for (auto Target : Ints.Targets)
    OS << "  {llvm::StringLiteral(\"" << Target.Name << "\"), " << Target.Offset
       << ", " << Target.Count << "},\n";
  OS << "};\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitIntrinsicToNameTable(
    const CodeGenIntrinsicTable &Ints, raw_ostream &OS) {
  OS << "// Intrinsic ID to name table\n";
  OS << "#ifdef GET_INTRINSIC_NAME_TABLE\n";
  OS << "  // Note that entry #0 is the invalid intrinsic!\n";
  for (unsigned i = 0, e = Ints.size(); i != e; ++i)
    OS << "  \"" << Ints[i].Name << "\",\n";
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitIntrinsicToOverloadTable(
    const CodeGenIntrinsicTable &Ints, raw_ostream &OS) {
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


// NOTE: This must be kept in synch with the copy in lib/IR/Function.cpp!
enum IIT_Info {
  // Common values should be encoded with 0-15.
  IIT_Done = 0,
  IIT_I1 = 1,
  IIT_I8 = 2,
  IIT_I16 = 3,
  IIT_I32 = 4,
  IIT_I64 = 5,
  IIT_F16 = 6,
  IIT_F32 = 7,
  IIT_F64 = 8,
  IIT_V2 = 9,
  IIT_V4 = 10,
  IIT_V8 = 11,
  IIT_V16 = 12,
  IIT_V32 = 13,
  IIT_PTR = 14,
  IIT_ARG = 15,

  // Values from 16+ are only encodable with the inefficient encoding.
  IIT_V64 = 16,
  IIT_MMX = 17,
  IIT_TOKEN = 18,
  IIT_METADATA = 19,
  IIT_EMPTYSTRUCT = 20,
  IIT_STRUCT2 = 21,
  IIT_STRUCT3 = 22,
  IIT_STRUCT4 = 23,
  IIT_STRUCT5 = 24,
  IIT_EXTEND_ARG = 25,
  IIT_TRUNC_ARG = 26,
  IIT_ANYPTR = 27,
  IIT_V1 = 28,
  IIT_VARARG = 29,
  IIT_HALF_VEC_ARG = 30,
  IIT_SAME_VEC_WIDTH_ARG = 31,
  IIT_PTR_TO_ARG = 32,
  IIT_PTR_TO_ELT = 33,
  IIT_VEC_OF_ANYPTRS_TO_ELT = 34,
  IIT_I128 = 35,
  IIT_V512 = 36,
  IIT_V1024 = 37,
  IIT_STRUCT6 = 38,
  IIT_STRUCT7 = 39,
  IIT_STRUCT8 = 40,
  IIT_F128 = 41,
  IIT_VEC_ELEMENT = 42,
  IIT_SCALABLE_VEC = 43,
  IIT_SUBDIVIDE2_ARG = 44,
  IIT_SUBDIVIDE4_ARG = 45,
  IIT_VEC_OF_BITCASTS_TO_INT = 46,
  IIT_V128 = 47,
  IIT_BF16 = 48,
  IIT_STRUCT9 = 49,
  IIT_V256 = 50,
  IIT_AMX = 51,
  IIT_PPCF128 = 52,
  IIT_V3 = 53,
  IIT_EXTERNREF = 54,
  IIT_FUNCREF = 55,
  IIT_ANYPTR_TO_ELT = 56,
  IIT_I2 = 57,
  IIT_I4 = 58,
};

static void EncodeFixedValueType(MVT::SimpleValueType VT,
                                 std::vector<unsigned char> &Sig) {
  // clang-format off
  if (MVT(VT).isInteger()) {
    unsigned BitWidth = MVT(VT).getFixedSizeInBits();
    switch (BitWidth) {
    default: PrintFatalError("unhandled integer type width in intrinsic!");
    case 1: return Sig.push_back(IIT_I1);
    case 2: return Sig.push_back(IIT_I2);
    case 4: return Sig.push_back(IIT_I4);
    case 8: return Sig.push_back(IIT_I8);
    case 16: return Sig.push_back(IIT_I16);
    case 32: return Sig.push_back(IIT_I32);
    case 64: return Sig.push_back(IIT_I64);
    case 128: return Sig.push_back(IIT_I128);
    }
  }

  switch (VT) {
  default: PrintFatalError("unhandled MVT in intrinsic!");
  case MVT::f16: return Sig.push_back(IIT_F16);
  case MVT::bf16: return Sig.push_back(IIT_BF16);
  case MVT::f32: return Sig.push_back(IIT_F32);
  case MVT::f64: return Sig.push_back(IIT_F64);
  case MVT::f128: return Sig.push_back(IIT_F128);
  case MVT::ppcf128: return Sig.push_back(IIT_PPCF128);
  case MVT::token: return Sig.push_back(IIT_TOKEN);
  case MVT::Metadata: return Sig.push_back(IIT_METADATA);
  case MVT::x86mmx: return Sig.push_back(IIT_MMX);
  case MVT::x86amx: return Sig.push_back(IIT_AMX);
  // MVT::OtherVT is used to mean the empty struct type here.
  case MVT::Other: return Sig.push_back(IIT_EMPTYSTRUCT);
  // MVT::isVoid is used to represent varargs here.
  case MVT::isVoid: return Sig.push_back(IIT_VARARG);
  case MVT::externref:
    return Sig.push_back(IIT_EXTERNREF);
  case MVT::funcref:
    return Sig.push_back(IIT_FUNCREF);
  }
  // clang-format on
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma optimize("",off) // MSVC 2015 optimizer can't deal with this function.
#endif

static void EncodeFixedType(Record *R, std::vector<unsigned char> &ArgCodes,
                            unsigned &NextArgCode,
                            std::vector<unsigned char> &Sig,
                            ArrayRef<unsigned char> Mapping) {

  if (R->isSubClassOf("LLVMMatchType")) {
    unsigned Number = Mapping[R->getValueAsInt("Number")];
    assert(Number < ArgCodes.size() && "Invalid matching number!");
    if (R->isSubClassOf("LLVMExtendedType"))
      Sig.push_back(IIT_EXTEND_ARG);
    else if (R->isSubClassOf("LLVMTruncatedType"))
      Sig.push_back(IIT_TRUNC_ARG);
    else if (R->isSubClassOf("LLVMHalfElementsVectorType"))
      Sig.push_back(IIT_HALF_VEC_ARG);
    else if (R->isSubClassOf("LLVMScalarOrSameVectorWidth")) {
      Sig.push_back(IIT_SAME_VEC_WIDTH_ARG);
      Sig.push_back((Number << 3) | ArgCodes[Number]);
      MVT::SimpleValueType VT = getValueType(R->getValueAsDef("ElTy"));
      EncodeFixedValueType(VT, Sig);
      return;
    }
    else if (R->isSubClassOf("LLVMPointerTo"))
      Sig.push_back(IIT_PTR_TO_ARG);
    else if (R->isSubClassOf("LLVMVectorOfAnyPointersToElt")) {
      Sig.push_back(IIT_VEC_OF_ANYPTRS_TO_ELT);
      // Encode overloaded ArgNo
      Sig.push_back(NextArgCode++);
      // Encode LLVMMatchType<Number> ArgNo
      Sig.push_back(Number);
      return;
    } else if (R->isSubClassOf("LLVMAnyPointerToElt")) {
      Sig.push_back(IIT_ANYPTR_TO_ELT);
      // Encode overloaded ArgNo
      Sig.push_back(NextArgCode++);
      // Encode LLVMMatchType<Number> ArgNo
      Sig.push_back(Number);
      return;
    } else if (R->isSubClassOf("LLVMPointerToElt"))
      Sig.push_back(IIT_PTR_TO_ELT);
    else if (R->isSubClassOf("LLVMVectorElementType"))
      Sig.push_back(IIT_VEC_ELEMENT);
    else if (R->isSubClassOf("LLVMSubdivide2VectorType"))
      Sig.push_back(IIT_SUBDIVIDE2_ARG);
    else if (R->isSubClassOf("LLVMSubdivide4VectorType"))
      Sig.push_back(IIT_SUBDIVIDE4_ARG);
    else if (R->isSubClassOf("LLVMVectorOfBitcastsToInt"))
      Sig.push_back(IIT_VEC_OF_BITCASTS_TO_INT);
    else
      Sig.push_back(IIT_ARG);
    return Sig.push_back((Number << 3) | 7 /*IITDescriptor::AK_MatchType*/);
  }

  MVT::SimpleValueType VT = getValueType(R->getValueAsDef("VT"));

  unsigned Tmp = 0;
  switch (VT) {
  default: break;
  case MVT::iPTRAny: ++Tmp; [[fallthrough]];
  case MVT::vAny: ++Tmp;    [[fallthrough]];
  case MVT::fAny: ++Tmp;    [[fallthrough]];
  case MVT::iAny: ++Tmp;    [[fallthrough]];
  case MVT::Any: {
    // If this is an "any" valuetype, then the type is the type of the next
    // type in the list specified to getIntrinsic().
    Sig.push_back(IIT_ARG);

    // Figure out what arg # this is consuming, and remember what kind it was.
    assert(NextArgCode < ArgCodes.size() && ArgCodes[NextArgCode] == Tmp &&
           "Invalid or no ArgCode associated with overloaded VT!");
    unsigned ArgNo = NextArgCode++;

    // Encode what sort of argument it must be in the low 3 bits of the ArgNo.
    return Sig.push_back((ArgNo << 3) | Tmp);
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
    return EncodeFixedType(R->getValueAsDef("ElTy"), ArgCodes, NextArgCode, Sig,
                           Mapping);
  }
  }

  if (MVT(VT).isVector()) {
    MVT VVT = VT;
    if (VVT.isScalableVector())
      Sig.push_back(IIT_SCALABLE_VEC);
    switch (VVT.getVectorMinNumElements()) {
    default: PrintFatalError("unhandled vector type width in intrinsic!");
    case 1: Sig.push_back(IIT_V1); break;
    case 2: Sig.push_back(IIT_V2); break;
    case 3: Sig.push_back(IIT_V3); break;
    case 4: Sig.push_back(IIT_V4); break;
    case 8: Sig.push_back(IIT_V8); break;
    case 16: Sig.push_back(IIT_V16); break;
    case 32: Sig.push_back(IIT_V32); break;
    case 64: Sig.push_back(IIT_V64); break;
    case 128: Sig.push_back(IIT_V128); break;
    case 256: Sig.push_back(IIT_V256); break;
    case 512: Sig.push_back(IIT_V512); break;
    case 1024: Sig.push_back(IIT_V1024); break;
    }

    return EncodeFixedValueType(VVT.getVectorElementType().SimpleTy, Sig);
  }

  EncodeFixedValueType(VT, Sig);
}

static void UpdateArgCodes(Record *R, std::vector<unsigned char> &ArgCodes,
                           unsigned int &NumInserted,
                           SmallVectorImpl<unsigned char> &Mapping) {
  if (R->isSubClassOf("LLVMMatchType")) {
    if (R->isSubClassOf("LLVMVectorOfAnyPointersToElt")) {
      ArgCodes.push_back(3 /*vAny*/);
      ++NumInserted;
    } else if (R->isSubClassOf("LLVMAnyPointerToElt")) {
      ArgCodes.push_back(4 /*iPTRAny*/);
      ++NumInserted;
    }
    return;
  }

  unsigned Tmp = 0;
  switch (getValueType(R->getValueAsDef("VT"))) {
  default: break;
  case MVT::iPTR:
    UpdateArgCodes(R->getValueAsDef("ElTy"), ArgCodes, NumInserted, Mapping);
    break;
  case MVT::iPTRAny:
    ++Tmp;
    [[fallthrough]];
  case MVT::vAny:
    ++Tmp;
    [[fallthrough]];
  case MVT::fAny:
    ++Tmp;
    [[fallthrough]];
  case MVT::iAny:
    ++Tmp;
    [[fallthrough]];
  case MVT::Any:
    unsigned OriginalIdx = ArgCodes.size() - NumInserted;
    assert(OriginalIdx >= Mapping.size());
    Mapping.resize(OriginalIdx+1);
    Mapping[OriginalIdx] = ArgCodes.size();
    ArgCodes.push_back(Tmp);
    break;
  }
}

#if defined(_MSC_VER) && !defined(__clang__)
#pragma optimize("",on)
#endif

/// ComputeFixedEncoding - If we can encode the type signature for this
/// intrinsic into 32 bits, return it.  If not, return ~0U.
static void ComputeFixedEncoding(const CodeGenIntrinsic &Int,
                                 std::vector<unsigned char> &TypeSig) {
  std::vector<unsigned char> ArgCodes;

  // Add codes for any overloaded result VTs.
  unsigned int NumInserted = 0;
  SmallVector<unsigned char, 8> ArgMapping;
  for (unsigned i = 0, e = Int.IS.RetVTs.size(); i != e; ++i)
    UpdateArgCodes(Int.IS.RetTypeDefs[i], ArgCodes, NumInserted, ArgMapping);

  // Add codes for any overloaded operand VTs.
  for (unsigned i = 0, e = Int.IS.ParamTypeDefs.size(); i != e; ++i)
    UpdateArgCodes(Int.IS.ParamTypeDefs[i], ArgCodes, NumInserted, ArgMapping);

  unsigned NextArgCode = 0;
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
      case 6: TypeSig.push_back(IIT_STRUCT6); break;
      case 7: TypeSig.push_back(IIT_STRUCT7); break;
      case 8: TypeSig.push_back(IIT_STRUCT8); break;
      case 9: TypeSig.push_back(IIT_STRUCT9); break;
      default: llvm_unreachable("Unhandled case in struct");
    }

    for (unsigned i = 0, e = Int.IS.RetVTs.size(); i != e; ++i)
      EncodeFixedType(Int.IS.RetTypeDefs[i], ArgCodes, NextArgCode, TypeSig,
                      ArgMapping);
  }

  for (unsigned i = 0, e = Int.IS.ParamTypeDefs.size(); i != e; ++i)
    EncodeFixedType(Int.IS.ParamTypeDefs[i], ArgCodes, NextArgCode, TypeSig,
                    ArgMapping);
}

static void printIITEntry(raw_ostream &OS, unsigned char X) {
  OS << (unsigned)X;
}

void IntrinsicEmitter::EmitGenerator(const CodeGenIntrinsicTable &Ints,
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
      OS << "0x" << Twine::utohexstr(FixedEncodings[i]) << ", ";
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

namespace {
struct AttributeComparator {
  bool operator()(const CodeGenIntrinsic *L, const CodeGenIntrinsic *R) const {
    // Sort throwing intrinsics after non-throwing intrinsics.
    if (L->canThrow != R->canThrow)
      return R->canThrow;

    if (L->isNoDuplicate != R->isNoDuplicate)
      return R->isNoDuplicate;

    if (L->isNoMerge != R->isNoMerge)
      return R->isNoMerge;

    if (L->isNoReturn != R->isNoReturn)
      return R->isNoReturn;

    if (L->isNoCallback != R->isNoCallback)
      return R->isNoCallback;

    if (L->isNoSync != R->isNoSync)
      return R->isNoSync;

    if (L->isNoFree != R->isNoFree)
      return R->isNoFree;

    if (L->isWillReturn != R->isWillReturn)
      return R->isWillReturn;

    if (L->isCold != R->isCold)
      return R->isCold;

    if (L->isConvergent != R->isConvergent)
      return R->isConvergent;

    if (L->isSpeculatable != R->isSpeculatable)
      return R->isSpeculatable;

    if (L->hasSideEffects != R->hasSideEffects)
      return R->hasSideEffects;

    // Try to order by readonly/readnone attribute.
    CodeGenIntrinsic::ModRefBehavior LK = L->ModRef;
    CodeGenIntrinsic::ModRefBehavior RK = R->ModRef;
    if (LK != RK) return (LK > RK);
    // Order by argument attributes.
    // This is reliable because each side is already sorted internally.
    return (L->ArgumentAttributes < R->ArgumentAttributes);
  }
};
} // End anonymous namespace

/// EmitAttributes - This emits the Intrinsic::getAttributes method.
void IntrinsicEmitter::EmitAttributes(const CodeGenIntrinsicTable &Ints,
                                      raw_ostream &OS) {
  OS << "// Add parameter attributes that are not common to all intrinsics.\n";
  OS << "#ifdef GET_INTRINSIC_ATTRIBUTES\n";
  OS << "AttributeList Intrinsic::getAttributes(LLVMContext &C, ID id) {\n";

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
    N = ++AttrNum;
    assert(N < 65536 && "Too many unique attributes for table!");
  }

  // Emit an array of AttributeList.  Most intrinsics will have at least one
  // entry, for the function itself (index ~1), which is usually nounwind.
  OS << "  static const uint16_t IntrinsicsToAttributesMap[] = {\n";

  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const CodeGenIntrinsic &intrinsic = Ints[i];

    OS << "    " << UniqAttributes[&intrinsic] << ", // "
       << intrinsic.Name << "\n";
  }
  OS << "  };\n\n";

  OS << "  AttributeList AS[" << maxArgAttrs + 1 << "];\n";
  OS << "  unsigned NumAttrs = 0;\n";
  OS << "  if (id != 0) {\n";
  OS << "    switch(IntrinsicsToAttributesMap[id - 1]) {\n";
  OS << "    default: llvm_unreachable(\"Invalid attribute number\");\n";
  for (auto UniqAttribute : UniqAttributes) {
    OS << "    case " << UniqAttribute.second << ": {\n";

    const CodeGenIntrinsic &Intrinsic = *(UniqAttribute.first);

    // Keep track of the number of attributes we're writing out.
    unsigned numAttrs = 0;

    // The argument attributes are alreadys sorted by argument index.
    unsigned Ai = 0, Ae = Intrinsic.ArgumentAttributes.size();
    if (Ae) {
      while (Ai != Ae) {
        unsigned AttrIdx = Intrinsic.ArgumentAttributes[Ai].Index;

        OS << "      const Attribute::AttrKind AttrParam" << AttrIdx << "[]= {";
        ListSeparator LS(",");

        bool AllValuesAreZero = true;
        SmallVector<uint64_t, 8> Values;
        do {
          switch (Intrinsic.ArgumentAttributes[Ai].Kind) {
          case CodeGenIntrinsic::NoCapture:
            OS << LS << "Attribute::NoCapture";
            break;
          case CodeGenIntrinsic::NoAlias:
            OS << LS << "Attribute::NoAlias";
            break;
          case CodeGenIntrinsic::NoUndef:
            OS << LS << "Attribute::NoUndef";
            break;
          case CodeGenIntrinsic::Returned:
            OS << LS << "Attribute::Returned";
            break;
          case CodeGenIntrinsic::ReadOnly:
            OS << LS << "Attribute::ReadOnly";
            break;
          case CodeGenIntrinsic::WriteOnly:
            OS << LS << "Attribute::WriteOnly";
            break;
          case CodeGenIntrinsic::ReadNone:
            OS << LS << "Attribute::ReadNone";
            break;
          case CodeGenIntrinsic::ImmArg:
            OS << LS << "Attribute::ImmArg";
            break;
          case CodeGenIntrinsic::Alignment:
            OS << LS << "Attribute::Alignment";
            break;
          }
          uint64_t V = Intrinsic.ArgumentAttributes[Ai].Value;
          Values.push_back(V);
          AllValuesAreZero &= (V == 0);

          ++Ai;
        } while (Ai != Ae && Intrinsic.ArgumentAttributes[Ai].Index == AttrIdx);
        OS << "};\n";

        // Generate attribute value array if not all attribute values are zero.
        if (!AllValuesAreZero) {
          OS << "      const uint64_t AttrValParam" << AttrIdx << "[]= {";
          ListSeparator LSV(",");
          for (const auto V : Values)
            OS << LSV << V;
          OS << "};\n";
        }

        OS << "      AS[" << numAttrs++ << "] = AttributeList::get(C, "
           << AttrIdx << ", AttrParam" << AttrIdx;
        if (!AllValuesAreZero)
          OS << ", AttrValParam" << AttrIdx;
        OS << ");\n";
      }
    }

    if (!Intrinsic.canThrow ||
        (Intrinsic.ModRef != CodeGenIntrinsic::ReadWriteMem &&
         !Intrinsic.hasSideEffects) ||
        Intrinsic.isNoReturn || Intrinsic.isNoCallback || Intrinsic.isNoSync ||
        Intrinsic.isNoFree || Intrinsic.isWillReturn || Intrinsic.isCold ||
        Intrinsic.isNoDuplicate || Intrinsic.isNoMerge ||
        Intrinsic.isConvergent || Intrinsic.isSpeculatable) {
      OS << "      const Attribute::AttrKind Atts[] = {";
      ListSeparator LS(",");
      if (!Intrinsic.canThrow)
        OS << LS << "Attribute::NoUnwind";
      if (Intrinsic.isNoReturn)
        OS << LS << "Attribute::NoReturn";
      if (Intrinsic.isNoCallback)
        OS << LS << "Attribute::NoCallback";
      if (Intrinsic.isNoSync)
        OS << LS << "Attribute::NoSync";
      if (Intrinsic.isNoFree)
        OS << LS << "Attribute::NoFree";
      if (Intrinsic.isWillReturn)
        OS << LS << "Attribute::WillReturn";
      if (Intrinsic.isCold)
        OS << LS << "Attribute::Cold";
      if (Intrinsic.isNoDuplicate)
        OS << LS << "Attribute::NoDuplicate";
      if (Intrinsic.isNoMerge)
        OS << LS << "Attribute::NoMerge";
      if (Intrinsic.isConvergent)
        OS << LS << "Attribute::Convergent";
      if (Intrinsic.isSpeculatable)
        OS << LS << "Attribute::Speculatable";

      switch (Intrinsic.ModRef) {
      case CodeGenIntrinsic::NoMem:
        if (Intrinsic.hasSideEffects)
          break;
        OS << LS;
        OS << "Attribute::ReadNone";
        break;
      case CodeGenIntrinsic::ReadArgMem:
        OS << LS;
        OS << "Attribute::ReadOnly,";
        OS << "Attribute::ArgMemOnly";
        break;
      case CodeGenIntrinsic::ReadMem:
        OS << LS;
        OS << "Attribute::ReadOnly";
        break;
      case CodeGenIntrinsic::ReadInaccessibleMem:
        OS << LS;
        OS << "Attribute::ReadOnly,";
        OS << "Attribute::InaccessibleMemOnly";
        break;
      case CodeGenIntrinsic::ReadInaccessibleMemOrArgMem:
        OS << LS;
        OS << "Attribute::ReadOnly,";
        OS << "Attribute::InaccessibleMemOrArgMemOnly";
        break;
      case CodeGenIntrinsic::WriteArgMem:
        OS << LS;
        OS << "Attribute::WriteOnly,";
        OS << "Attribute::ArgMemOnly";
        break;
      case CodeGenIntrinsic::WriteMem:
        OS << LS;
        OS << "Attribute::WriteOnly";
        break;
      case CodeGenIntrinsic::WriteInaccessibleMem:
        OS << LS;
        OS << "Attribute::WriteOnly,";
        OS << "Attribute::InaccessibleMemOnly";
        break;
      case CodeGenIntrinsic::WriteInaccessibleMemOrArgMem:
        OS << LS;
        OS << "Attribute::WriteOnly,";
        OS << "Attribute::InaccessibleMemOrArgMemOnly";
        break;
      case CodeGenIntrinsic::ReadWriteArgMem:
        OS << LS;
        OS << "Attribute::ArgMemOnly";
        break;
      case CodeGenIntrinsic::ReadWriteInaccessibleMem:
        OS << LS;
        OS << "Attribute::InaccessibleMemOnly";
        break;
      case CodeGenIntrinsic::ReadWriteInaccessibleMemOrArgMem:
        OS << LS;
        OS << "Attribute::InaccessibleMemOrArgMemOnly";
        break;
      case CodeGenIntrinsic::ReadWriteMem:
        break;
      }
      OS << "};\n";
      OS << "      AS[" << numAttrs++ << "] = AttributeList::get(C, "
         << "AttributeList::FunctionIndex, Atts);\n";
    }

    if (numAttrs) {
      OS << "      NumAttrs = " << numAttrs << ";\n";
      OS << "      break;\n";
      OS << "      }\n";
    } else {
      OS << "      return AttributeList();\n";
      OS << "      }\n";
    }
  }

  OS << "    }\n";
  OS << "  }\n";
  OS << "  return AttributeList::get(C, makeArrayRef(AS, NumAttrs));\n";
  OS << "}\n";
  OS << "#endif // GET_INTRINSIC_ATTRIBUTES\n\n";
}

void IntrinsicEmitter::EmitIntrinsicToBuiltinMap(
    const CodeGenIntrinsicTable &Ints, bool IsClang, raw_ostream &OS) {
  StringRef CompilerName = (IsClang ? "Clang" : "MS");
  StringRef UpperCompilerName = (IsClang ? "CLANG" : "MS");
  typedef std::map<std::string, std::map<std::string, std::string>> BIMTy;
  BIMTy BuiltinMap;
  StringToOffsetTable Table;
  for (unsigned i = 0, e = Ints.size(); i != e; ++i) {
    const std::string &BuiltinName =
        IsClang ? Ints[i].ClangBuiltinName : Ints[i].MSBuiltinName;
    if (!BuiltinName.empty()) {
      // Get the map for this target prefix.
      std::map<std::string, std::string> &BIM =
          BuiltinMap[Ints[i].TargetPrefix];

      if (!BIM.insert(std::make_pair(BuiltinName, Ints[i].EnumName)).second)
        PrintFatalError(Ints[i].TheDef->getLoc(),
                        "Intrinsic '" + Ints[i].TheDef->getName() +
                            "': duplicate " + CompilerName + " builtin name!");
      Table.GetOrAddStringOffset(BuiltinName);
    }
  }

  OS << "// Get the LLVM intrinsic that corresponds to a builtin.\n";
  OS << "// This is used by the C front-end.  The builtin name is passed\n";
  OS << "// in as BuiltinName, and a target prefix (e.g. 'ppc') is passed\n";
  OS << "// in as TargetPrefix.  The result is assigned to 'IntrinsicID'.\n";
  OS << "#ifdef GET_LLVM_INTRINSIC_FOR_" << UpperCompilerName << "_BUILTIN\n";

  OS << "Intrinsic::ID Intrinsic::getIntrinsicFor" << CompilerName
     << "Builtin(const char "
     << "*TargetPrefixStr, StringRef BuiltinNameStr) {\n";

  if (Table.Empty()) {
    OS << "  return Intrinsic::not_intrinsic;\n";
    OS << "}\n";
    OS << "#endif\n\n";
    return;
  }

  OS << "  static const char BuiltinNames[] = {\n";
  Table.EmitCharArray(OS);
  OS << "  };\n\n";

  OS << "  struct BuiltinEntry {\n";
  OS << "    Intrinsic::ID IntrinID;\n";
  OS << "    unsigned StrTabOffset;\n";
  OS << "    const char *getName() const {\n";
  OS << "      return &BuiltinNames[StrTabOffset];\n";
  OS << "    }\n";
  OS << "    bool operator<(StringRef RHS) const {\n";
  OS << "      return strncmp(getName(), RHS.data(), RHS.size()) < 0;\n";
  OS << "    }\n";
  OS << "  };\n";

  OS << "  StringRef TargetPrefix(TargetPrefixStr);\n\n";

  // Note: this could emit significantly better code if we cared.
  for (auto &I : BuiltinMap) {
    OS << "  ";
    if (!I.first.empty())
      OS << "if (TargetPrefix == \"" << I.first << "\") ";
    else
      OS << "/* Target Independent Builtins */ ";
    OS << "{\n";

    // Emit the comparisons for this target prefix.
    OS << "    static const BuiltinEntry " << I.first << "Names[] = {\n";
    for (const auto &P : I.second) {
      OS << "      {Intrinsic::" << P.second << ", "
         << Table.GetOrAddStringOffset(P.first) << "}, // " << P.first << "\n";
    }
    OS << "    };\n";
    OS << "    auto I = std::lower_bound(std::begin(" << I.first << "Names),\n";
    OS << "                              std::end(" << I.first << "Names),\n";
    OS << "                              BuiltinNameStr);\n";
    OS << "    if (I != std::end(" << I.first << "Names) &&\n";
    OS << "        I->getName() == BuiltinNameStr)\n";
    OS << "      return I->IntrinID;\n";
    OS << "  }\n";
  }
  OS << "  return ";
  OS << "Intrinsic::not_intrinsic;\n";
  OS << "}\n";
  OS << "#endif\n\n";
}

void llvm::EmitIntrinsicEnums(RecordKeeper &RK, raw_ostream &OS) {
  IntrinsicEmitter(RK).run(OS, /*Enums=*/true);
}

void llvm::EmitIntrinsicImpl(RecordKeeper &RK, raw_ostream &OS) {
  IntrinsicEmitter(RK).run(OS, /*Enums=*/false);
}
