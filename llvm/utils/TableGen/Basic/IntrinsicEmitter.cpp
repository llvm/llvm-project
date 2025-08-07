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
#include "SequenceToOffsetTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ModRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/StringToOffsetTable.h"
#include "llvm/TableGen/TableGenBackend.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>
using namespace llvm;

static cl::OptionCategory GenIntrinsicCat("Options for -gen-intrinsic-enums");
static cl::opt<std::string>
    IntrinsicPrefix("intrinsic-prefix",
                    cl::desc("Generate intrinsics with this target prefix"),
                    cl::value_desc("target prefix"), cl::cat(GenIntrinsicCat));

namespace {
class IntrinsicEmitter {
  const RecordKeeper &Records;

public:
  IntrinsicEmitter(const RecordKeeper &R) : Records(R) {}

  void run(raw_ostream &OS, bool Enums);

  void EmitEnumInfo(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitArgKind(raw_ostream &OS);
  void EmitIITInfo(raw_ostream &OS);
  void EmitTargetInfo(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitIntrinsicToNameTable(const CodeGenIntrinsicTable &Ints,
                                raw_ostream &OS);
  void EmitIntrinsicToOverloadTable(const CodeGenIntrinsicTable &Ints,
                                    raw_ostream &OS);
  void EmitGenerator(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitAttributes(const CodeGenIntrinsicTable &Ints, raw_ostream &OS);
  void EmitIntrinsicToBuiltinMap(const CodeGenIntrinsicTable &Ints,
                                 bool IsClang, raw_ostream &OS);
};

// Helper class to use with `TableGen::Emitter::OptClass`.
template <bool Enums> class IntrinsicEmitterOpt : public IntrinsicEmitter {
public:
  IntrinsicEmitterOpt(const RecordKeeper &R) : IntrinsicEmitter(R) {}
  void run(raw_ostream &OS) { IntrinsicEmitter::run(OS, Enums); }
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

    // Emit ArgKind for Intrinsics.h.
    EmitArgKind(OS);
  } else {
    // Emit IIT_Info constants.
    EmitIITInfo(OS);

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

    // Emit code to translate Clang builtins into LLVM intrinsics.
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
  using TargetSet = CodeGenIntrinsicTable::TargetSet;
  const TargetSet *Set = nullptr;
  for (const auto &Target : Ints.getTargets()) {
    if (Target.Name == IntrinsicPrefix) {
      Set = &Target;
      break;
    }
  }
  if (!Set) {
    // The first entry is for target independent intrinsics, so drop it.
    auto KnowTargets = Ints.getTargets().drop_front();
    PrintFatalError([KnowTargets](raw_ostream &OS) {
      OS << "tried to generate intrinsics for unknown target "
         << IntrinsicPrefix << "\nKnown targets are: ";
      interleaveComma(KnowTargets, OS,
                      [&OS](const TargetSet &Target) { OS << Target.Name; });
      OS << '\n';
    });
  }

  // Generate a complete header for target specific intrinsics.
  if (IntrinsicPrefix.empty()) {
    OS << "#ifdef GET_INTRINSIC_ENUM_VALUES\n";
  } else {
    std::string UpperPrefix = StringRef(IntrinsicPrefix).upper();
    OS << formatv("#ifndef LLVM_IR_INTRINSIC_{}_ENUMS_H\n", UpperPrefix);
    OS << formatv("#define LLVM_IR_INTRINSIC_{}_ENUMS_H\n", UpperPrefix);
    OS << "namespace llvm::Intrinsic {\n";
    OS << formatv("enum {}Intrinsics : unsigned {{\n", UpperPrefix);
  }

  OS << "// Enum values for intrinsics.\n";
  bool First = true;
  for (const auto &Int : Ints[*Set]) {
    OS << "    " << Int.EnumName;

    // Assign a value to the first intrinsic in this target set so that all
    // intrinsic ids are distinct.
    if (First) {
      OS << " = " << Set->Offset + 1;
      First = false;
    }

    OS << ", ";
    if (Int.EnumName.size() < 40)
      OS.indent(40 - Int.EnumName.size());
    OS << formatv(" // {}\n", Int.Name);
  }

  // Emit num_intrinsics into the target neutral enum.
  if (IntrinsicPrefix.empty()) {
    OS << formatv("    num_intrinsics = {}\n", Ints.size() + 1);
    OS << "#endif\n\n";
  } else {
    OS << R"(}; // enum
} // namespace llvm::Intrinsic
#endif

)";
  }
}

void IntrinsicEmitter::EmitArgKind(raw_ostream &OS) {
  if (!IntrinsicPrefix.empty())
    return;
  OS << "// llvm::Intrinsic::IITDescriptor::ArgKind.\n";
  OS << "#ifdef GET_INTRINSIC_ARGKIND\n";
  if (const auto RecArgKind = Records.getDef("ArgKind")) {
    for (const auto &RV : RecArgKind->getValues())
      OS << "    AK_" << RV.getName() << " = " << *RV.getValue() << ",\n";
  } else {
    OS << "#error \"ArgKind is not defined\"\n";
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitIITInfo(raw_ostream &OS) {
  OS << "#ifdef GET_INTRINSIC_IITINFO\n";
  std::array<StringRef, 256> RecsByNumber;
  auto IIT_Base = Records.getAllDerivedDefinitionsIfDefined("IIT_Base");
  for (const Record *Rec : IIT_Base) {
    auto Number = Rec->getValueAsInt("Number");
    assert(0 <= Number && Number < (int)RecsByNumber.size() &&
           "IIT_Info.Number should be uint8_t");
    assert(RecsByNumber[Number].empty() && "Duplicate IIT_Info.Number");
    RecsByNumber[Number] = Rec->getName();
  }
  if (IIT_Base.size() > 0) {
    for (unsigned I = 0, E = RecsByNumber.size(); I < E; ++I)
      if (!RecsByNumber[I].empty())
        OS << "  " << RecsByNumber[I] << " = " << I << ",\n";
  } else {
    OS << "#error \"class IIT_Base is not defined\"\n";
  }
  OS << "#endif\n\n";
}

void IntrinsicEmitter::EmitTargetInfo(const CodeGenIntrinsicTable &Ints,
                                      raw_ostream &OS) {
  OS << R"(// Target mapping.
#ifdef GET_INTRINSIC_TARGET_DATA
struct IntrinsicTargetInfo {
  StringLiteral Name;
  size_t Offset;
  size_t Count;
};
static constexpr IntrinsicTargetInfo TargetInfos[] = {
)";
  for (const auto [Name, Offset, Count] : Ints.getTargets())
    OS << formatv("  {{\"{}\", {}, {}},\n", Name, Offset, Count);
  OS << R"(};
#endif

)";
}

void IntrinsicEmitter::EmitIntrinsicToNameTable(
    const CodeGenIntrinsicTable &Ints, raw_ostream &OS) {
  // Built up a table of the intrinsic names.
  constexpr StringLiteral NotIntrinsic = "not_intrinsic";
  StringToOffsetTable Table;
  Table.GetOrAddStringOffset(NotIntrinsic);
  for (const auto &Int : Ints)
    Table.GetOrAddStringOffset(Int.Name);

  OS << R"(// Intrinsic ID to name table.
#ifdef GET_INTRINSIC_NAME_TABLE
// Note that entry #0 is the invalid intrinsic!

)";

  Table.EmitStringTableDef(OS, "IntrinsicNameTable");

  OS << R"(
static constexpr unsigned IntrinsicNameOffsetTable[] = {
)";

  OS << formatv("  {}, // {}\n", Table.GetStringOffset(NotIntrinsic),
                NotIntrinsic);
  for (const auto &Int : Ints)
    OS << formatv("  {}, // {}\n", Table.GetStringOffset(Int.Name), Int.Name);

  OS << R"(
}; // IntrinsicNameOffsetTable

#endif

)";
}

void IntrinsicEmitter::EmitIntrinsicToOverloadTable(
    const CodeGenIntrinsicTable &Ints, raw_ostream &OS) {
  OS << R"(// Intrinsic ID to overload bitset.
#ifdef GET_INTRINSIC_OVERLOAD_TABLE
static constexpr uint8_t OTable[] = {
  0
  )";
  for (auto [I, Int] : enumerate(Ints)) {
    // Add one to the index so we emit a null bit for the invalid #0 intrinsic.
    size_t Idx = I + 1;

    if (Idx % 8 == 0)
      OS << ",\n  0";
    if (Int.isOverloaded)
      OS << " | (1<<" << Idx % 8 << ')';
  }
  OS << "\n};\n\n";
  // OTable contains a true bit at the position if the intrinsic is overloaded.
  OS << "return (OTable[id/8] & (1 << (id%8))) != 0;\n";
  OS << "#endif\n\n";
}

using TypeSigTy = SmallVector<unsigned char>;

/// Computes type signature of the intrinsic \p Int.
static TypeSigTy ComputeTypeSignature(const CodeGenIntrinsic &Int) {
  TypeSigTy TypeSig;
  const Record *TypeInfo = Int.TheDef->getValueAsDef("TypeInfo");
  const ListInit *TypeList = TypeInfo->getValueAsListInit("TypeSig");

  for (const auto *TypeListEntry : TypeList->getElements())
    TypeSig.emplace_back(cast<IntInit>(TypeListEntry)->getValue());
  return TypeSig;
}

// Pack the type signature into 32-bit fixed encoding word.
static std::optional<uint32_t> encodePacked(const TypeSigTy &TypeSig) {
  if (TypeSig.size() > 8)
    return std::nullopt;

  uint32_t Result = 0;
  for (unsigned char C : reverse(TypeSig)) {
    if (C > 15)
      return std::nullopt;
    Result = (Result << 4) | C;
  }
  return Result;
}

void IntrinsicEmitter::EmitGenerator(const CodeGenIntrinsicTable &Ints,
                                     raw_ostream &OS) {
  // Note: the code below can be switched to use 32-bit fixed encoding by
  // flipping the flag below.
  constexpr bool Use16BitFixedEncoding = true;
  using FixedEncodingTy =
      std::conditional_t<Use16BitFixedEncoding, uint16_t, uint32_t>;
  constexpr unsigned FixedEncodingBits = sizeof(FixedEncodingTy) * CHAR_BIT;
  // Mask with all bits 1 except the most significant bit.
  const unsigned Mask = (1U << (FixedEncodingBits - 1)) - 1;
  const unsigned MSBPostion = FixedEncodingBits - 1;
  StringRef FixedEncodingTypeName =
      Use16BitFixedEncoding ? "uint16_t" : "uint32_t";

  // If we can compute a 16/32-bit fixed encoding for this intrinsic, do so and
  // capture it in this vector, otherwise store a ~0U.
  std::vector<FixedEncodingTy> FixedEncodings;
  SequenceToOffsetTable<TypeSigTy> LongEncodingTable;

  FixedEncodings.reserve(Ints.size());

  // Compute the unique argument type info.
  for (const CodeGenIntrinsic &Int : Ints) {
    // Get the signature for the intrinsic.
    TypeSigTy TypeSig = ComputeTypeSignature(Int);

    // Check to see if we can encode it into a 16/32 bit word.
    std::optional<uint32_t> Result = encodePacked(TypeSig);
    if (Result && (*Result & Mask) == Result) {
      FixedEncodings.push_back(static_cast<FixedEncodingTy>(*Result));
      continue;
    }

    LongEncodingTable.add(TypeSig);

    // This is a placehold that we'll replace after the table is laid out.
    FixedEncodings.push_back(static_cast<FixedEncodingTy>(~0U));
  }

  LongEncodingTable.layout();

  OS << formatv(R"(// Global intrinsic function declaration type table.
#ifdef GET_INTRINSIC_GENERATOR_GLOBAL
static constexpr {} IIT_Table[] = {{
  )",
                FixedEncodingTypeName);

  unsigned MaxOffset = 0;
  for (auto [Idx, FixedEncoding, Int] : enumerate(FixedEncodings, Ints)) {
    if ((Idx & 7) == 7)
      OS << "\n  ";

    // If the entry fit in the table, just emit it.
    if ((FixedEncoding & Mask) == FixedEncoding) {
      OS << "0x" << Twine::utohexstr(FixedEncoding) << ", ";
      continue;
    }

    TypeSigTy TypeSig = ComputeTypeSignature(Int);
    unsigned Offset = LongEncodingTable.get(TypeSig);
    MaxOffset = std::max(MaxOffset, Offset);

    // Otherwise, emit the offset into the long encoding table.  We emit it this
    // way so that it is easier to read the offset in the .def file.
    OS << formatv("(1U<<{}) | {}, ", MSBPostion, Offset);
  }

  OS << "0\n};\n\n";

  // verify that all offsets will fit in 16/32 bits.
  if ((MaxOffset & Mask) != MaxOffset)
    PrintFatalError("Offset of long encoding table exceeds encoding bits");

  // Emit the shared table of register lists.
  OS << "static constexpr unsigned char IIT_LongEncodingTable[] = {\n";
  if (!LongEncodingTable.empty())
    LongEncodingTable.emit(
        OS, [](raw_ostream &OS, unsigned char C) { OS << (unsigned)C; });
  OS << "  255\n};\n";
  OS << "#endif\n\n"; // End of GET_INTRINSIC_GENERATOR_GLOBAL
}

/// Returns the effective MemoryEffects for intrinsic \p Int.
static MemoryEffects getEffectiveME(const CodeGenIntrinsic &Int) {
  MemoryEffects ME = Int.ME;
  // TODO: IntrHasSideEffects should affect not only readnone intrinsics.
  if (ME.doesNotAccessMemory() && Int.hasSideEffects)
    ME = MemoryEffects::unknown();
  return ME;
}

static bool compareFnAttributes(const CodeGenIntrinsic *L,
                                const CodeGenIntrinsic *R) {
  auto TieBoolAttributes = [](const CodeGenIntrinsic *I) -> auto {
    // Sort throwing intrinsics after non-throwing intrinsics.
    return std::tie(I->canThrow, I->isNoDuplicate, I->isNoMerge, I->isNoReturn,
                    I->isNoCallback, I->isNoSync, I->isNoFree, I->isWillReturn,
                    I->isCold, I->isConvergent, I->isSpeculatable,
                    I->hasSideEffects, I->isStrictFP);
  };

  auto TieL = TieBoolAttributes(L);
  auto TieR = TieBoolAttributes(R);

  if (TieL != TieR)
    return TieL < TieR;

  // Try to order by readonly/readnone attribute.
  uint32_t LME = getEffectiveME(*L).toIntValue();
  uint32_t RME = getEffectiveME(*R).toIntValue();
  if (LME != RME)
    return LME > RME;

  return false;
}

/// Returns true if \p Int has a non-empty set of function attributes. Note that
/// NoUnwind = !canThrow, so we need to negate it's sense to test if the
// intrinsic has NoUnwind attribute.
static bool hasFnAttributes(const CodeGenIntrinsic &Int) {
  return !Int.canThrow || Int.isNoReturn || Int.isNoCallback || Int.isNoSync ||
         Int.isNoFree || Int.isWillReturn || Int.isCold || Int.isNoDuplicate ||
         Int.isNoMerge || Int.isConvergent || Int.isSpeculatable ||
         Int.isStrictFP || getEffectiveME(Int) != MemoryEffects::unknown();
}

namespace {
struct FnAttributeComparator {
  bool operator()(const CodeGenIntrinsic *L, const CodeGenIntrinsic *R) const {
    return compareFnAttributes(L, R);
  }
};

struct AttributeComparator {
  bool operator()(const CodeGenIntrinsic *L, const CodeGenIntrinsic *R) const {
    // Order all intrinsics with no functiona attributes before all intrinsics
    // with function attributes.
    bool HasFnAttrLHS = hasFnAttributes(*L);
    bool HasFnAttrRHS = hasFnAttributes(*R);

    // Order by argument attributes if function `hasFnAttributes` is equal.
    // This is reliable because each side is already sorted internally.
    return std::tie(HasFnAttrLHS, L->ArgumentAttributes) <
           std::tie(HasFnAttrRHS, R->ArgumentAttributes);
  }
};
} // End anonymous namespace

/// Returns the name of the IR enum for argument attribute kind \p Kind.
static StringRef getArgAttrEnumName(CodeGenIntrinsic::ArgAttrKind Kind) {
  switch (Kind) {
  case CodeGenIntrinsic::NoCapture:
    llvm_unreachable("Handled separately");
  case CodeGenIntrinsic::NoAlias:
    return "NoAlias";
  case CodeGenIntrinsic::NoUndef:
    return "NoUndef";
  case CodeGenIntrinsic::NonNull:
    return "NonNull";
  case CodeGenIntrinsic::Returned:
    return "Returned";
  case CodeGenIntrinsic::ReadOnly:
    return "ReadOnly";
  case CodeGenIntrinsic::WriteOnly:
    return "WriteOnly";
  case CodeGenIntrinsic::ReadNone:
    return "ReadNone";
  case CodeGenIntrinsic::ImmArg:
    return "ImmArg";
  case CodeGenIntrinsic::Alignment:
    return "Alignment";
  case CodeGenIntrinsic::Dereferenceable:
    return "Dereferenceable";
  case CodeGenIntrinsic::Range:
    return "Range";
  }
  llvm_unreachable("Unknown CodeGenIntrinsic::ArgAttrKind enum");
}

/// EmitAttributes - This emits the Intrinsic::getAttributes method.
void IntrinsicEmitter::EmitAttributes(const CodeGenIntrinsicTable &Ints,
                                      raw_ostream &OS) {
  OS << R"(// Add parameter attributes that are not common to all intrinsics.
#ifdef GET_INTRINSIC_ATTRIBUTES
static AttributeSet getIntrinsicArgAttributeSet(LLVMContext &C, unsigned ID,
                                                Type *ArgType) {
  unsigned BitWidth = ArgType->getScalarSizeInBits();
  switch (ID) {
  default: llvm_unreachable("Invalid attribute set number");)";
  // Compute unique argument attribute sets.
  std::map<SmallVector<CodeGenIntrinsic::ArgAttribute, 0>, unsigned>
      UniqArgAttributes;
  for (const CodeGenIntrinsic &Int : Ints) {
    for (auto &Attrs : Int.ArgumentAttributes) {
      if (Attrs.empty())
        continue;

      unsigned ID = UniqArgAttributes.size();
      if (!UniqArgAttributes.try_emplace(Attrs, ID).second)
        continue;

      assert(is_sorted(Attrs) && "Argument attributes are not sorted");

      OS << formatv(R"(
  case {}:
    return AttributeSet::get(C, {{
)",
                    ID);
      for (const CodeGenIntrinsic::ArgAttribute &Attr : Attrs) {
        if (Attr.Kind == CodeGenIntrinsic::NoCapture) {
          OS << "      Attribute::getWithCaptureInfo(C, "
                "CaptureInfo::none()),\n";
          continue;
        }
        StringRef AttrName = getArgAttrEnumName(Attr.Kind);
        if (Attr.Kind == CodeGenIntrinsic::Alignment ||
            Attr.Kind == CodeGenIntrinsic::Dereferenceable)
          OS << formatv("      Attribute::get(C, Attribute::{}, {}),\n",
                        AttrName, Attr.Value);
        else if (Attr.Kind == CodeGenIntrinsic::Range)
          // This allows implicitTrunc because the range may only fit the
          // type based on rules implemented in the IR verifier. E.g. the
          // [-1, 1] range for ucmp/scmp intrinsics requires a minimum i2 type.
          // Give the verifier a chance to diagnose this instead of asserting
          // here.
          OS << formatv("      Attribute::get(C, Attribute::{}, "
                        "ConstantRange(APInt(BitWidth, {}, /*isSigned=*/true, "
                        "/*implicitTrunc=*/true), APInt(BitWidth, {}, "
                        "/*isSigned=*/true, /*implicitTrunc=*/true))),\n",
                        AttrName, (int64_t)Attr.Value, (int64_t)Attr.Value2);
        else
          OS << formatv("      Attribute::get(C, Attribute::{}),\n", AttrName);
      }
      OS << "    });";
    }
  }
  OS << R"(
  }
} // getIntrinsicArgAttributeSet
)";

  // Compute unique function attribute sets.
  std::map<const CodeGenIntrinsic *, unsigned, FnAttributeComparator>
      UniqFnAttributes;
  OS << R"(
static AttributeSet getIntrinsicFnAttributeSet(LLVMContext &C, unsigned ID) {
  switch (ID) {
    default: llvm_unreachable("Invalid attribute set number");)";

  for (const CodeGenIntrinsic &Int : Ints) {
    unsigned ID = UniqFnAttributes.size();
    if (!UniqFnAttributes.try_emplace(&Int, ID).second)
      continue;
    OS << formatv(R"(
  case {}:
    return AttributeSet::get(C, {{
)",
                  ID);
    auto addAttribute = [&OS](StringRef Attr) {
      OS << formatv("      Attribute::get(C, Attribute::{}),\n", Attr);
    };
    if (!Int.canThrow)
      addAttribute("NoUnwind");
    if (Int.isNoReturn)
      addAttribute("NoReturn");
    if (Int.isNoCallback)
      addAttribute("NoCallback");
    if (Int.isNoSync)
      addAttribute("NoSync");
    if (Int.isNoFree)
      addAttribute("NoFree");
    if (Int.isWillReturn)
      addAttribute("WillReturn");
    if (Int.isCold)
      addAttribute("Cold");
    if (Int.isNoDuplicate)
      addAttribute("NoDuplicate");
    if (Int.isNoMerge)
      addAttribute("NoMerge");
    if (Int.isConvergent)
      addAttribute("Convergent");
    if (Int.isSpeculatable)
      addAttribute("Speculatable");
    if (Int.isStrictFP)
      addAttribute("StrictFP");

    const MemoryEffects ME = getEffectiveME(Int);
    if (ME != MemoryEffects::unknown()) {
      OS << formatv("      // {}\n", ME);
      OS << formatv("      Attribute::getWithMemoryEffects(C, "
                    "MemoryEffects::createFromIntValue({})),\n",
                    ME.toIntValue());
    }
    OS << "    });";
  }
  OS << R"(
  }
} // getIntrinsicFnAttributeSet

static constexpr uint16_t IntrinsicsToAttributesMap[] = {)";

  // Compute the maximum number of attribute arguments and the map. For function
  // attributes, we only consider whether the intrinsics has any function
  // arguments or not.
  std::map<const CodeGenIntrinsic *, unsigned, AttributeComparator>
      UniqAttributes;
  for (const CodeGenIntrinsic &Int : Ints) {
    unsigned ID = UniqAttributes.size();
    UniqAttributes.try_emplace(&Int, ID);
  }

  // Emit an array of AttributeList.  Most intrinsics will have at least one
  // entry, for the function itself (index ~1), which is usually nounwind.
  for (const CodeGenIntrinsic &Int : Ints) {
    uint16_t FnAttrIndex = UniqFnAttributes[&Int];
    OS << formatv("\n    {} << 8 | {}, // {}", FnAttrIndex,
                  UniqAttributes[&Int], Int.Name);
  }

  // Assign a 16-bit packed ID for each intrinsic. The lower 8-bits will be its
  // "argument attribute ID" (index in UniqAttributes) and upper 8 bits will be
  // its "function attribute ID" (index in UniqFnAttributes).
  if (UniqAttributes.size() > 256)
    PrintFatalError("Too many unique argument attributes for table!");
  if (UniqFnAttributes.size() > 256)
    PrintFatalError("Too many unique function attributes for table!");

  OS << R"(
};

AttributeList Intrinsic::getAttributes(LLVMContext &C, ID id,
                                       FunctionType *FT) {)";
  // Find the max number of attributes to create the local array.
  unsigned MaxNumAttrs = 0;
  for (const auto [IntPtr, UniqueID] : UniqAttributes) {
    const CodeGenIntrinsic &Int = *IntPtr;
    unsigned NumAttrs =
        llvm::count_if(Int.ArgumentAttributes,
                       [](const auto &Attrs) { return !Attrs.empty(); });
    NumAttrs += hasFnAttributes(Int);
    MaxNumAttrs = std::max(MaxNumAttrs, NumAttrs);
  }

  OS << formatv(R"(
  if (id == 0)
    return AttributeList();

  uint16_t PackedID = IntrinsicsToAttributesMap[id - 1];
  uint8_t FnAttrID = PackedID >> 8;
  std::pair<unsigned, AttributeSet> AS[{}];
  unsigned NumAttrs = 0;
  bool HasFnAttr = false;
  switch(PackedID & 0xFF) {{
    default: llvm_unreachable("Invalid attribute number");
)",
                MaxNumAttrs);

  for (const auto [IntPtr, UniqueID] : UniqAttributes) {
    OS << formatv("  case {}:\n", UniqueID);
    const CodeGenIntrinsic &Int = *IntPtr;

    unsigned NumAttrs = 0;

    for (const auto &[AttrIdx, Attrs] : enumerate(Int.ArgumentAttributes)) {
      if (Attrs.empty())
        continue;

      unsigned ArgAttrID = UniqArgAttributes.find(Attrs)->second;
      OS << formatv("    AS[{}] = {{{}, getIntrinsicArgAttributeSet(C, {}, "
                    "FT->getContainedType({}))};\n",
                    NumAttrs++, AttrIdx, ArgAttrID, AttrIdx);
    }

    if (hasFnAttributes(Int))
      OS << "    HasFnAttr = true;\n";

    if (NumAttrs)
      OS << formatv("    NumAttrs = {};\n", NumAttrs);
    OS << "    break;\n";
  }

  OS << R"(  }
  if (HasFnAttr) {
    AS[NumAttrs++] = {AttributeList::FunctionIndex,
                      getIntrinsicFnAttributeSet(C, FnAttrID)};
  }
  return AttributeList::get(C, ArrayRef(AS, NumAttrs));
}
#endif // GET_INTRINSIC_ATTRIBUTES

)";
}

void IntrinsicEmitter::EmitIntrinsicToBuiltinMap(
    const CodeGenIntrinsicTable &Ints, bool IsClang, raw_ostream &OS) {
  StringRef CompilerName = IsClang ? "Clang" : "MS";
  StringRef UpperCompilerName = IsClang ? "CLANG" : "MS";

  // map<TargetPrefix, pair<map<BuiltinName, EnumName>, CommonPrefix>.
  // Note that we iterate over both the maps in the code below and both
  // iterations need to iterate in sorted key order. For the inner map, entries
  // need to be emitted in the sorted order of `BuiltinName` with `CommonPrefix`
  // rempved, because we use std::lower_bound to search these entries. For the
  // outer map as well, entries need to be emitted in sorter order of
  // `TargetPrefix` as we use std::lower_bound to search these entries.
  using BIMEntryTy =
      std::pair<std::map<StringRef, StringRef>, std::optional<StringRef>>;
  std::map<StringRef, BIMEntryTy> BuiltinMap;

  for (const CodeGenIntrinsic &Int : Ints) {
    StringRef BuiltinName = IsClang ? Int.ClangBuiltinName : Int.MSBuiltinName;
    if (BuiltinName.empty())
      continue;
    // Get the map for this target prefix.
    auto &[Map, CommonPrefix] = BuiltinMap[Int.TargetPrefix];

    if (!Map.try_emplace(BuiltinName, Int.EnumName).second)
      PrintFatalError(Int.TheDef->getLoc(),
                      "Intrinsic '" + Int.TheDef->getName() + "': duplicate " +
                          CompilerName + " builtin name!");

    // Update common prefix.
    if (!CommonPrefix) {
      // For the first builtin for this target, initialize the common prefix.
      CommonPrefix = BuiltinName;
      continue;
    }

    // Update the common prefix. Note that this assumes that `take_front` will
    // never set the `Data` pointer in CommonPrefix to nullptr.
    const char *Mismatch = mismatch(*CommonPrefix, BuiltinName).first;
    *CommonPrefix = CommonPrefix->take_front(Mismatch - CommonPrefix->begin());
  }

  // Populate the string table with the names of all the builtins after
  // removing this common prefix.
  StringToOffsetTable Table;
  for (const auto &[TargetPrefix, Entry] : BuiltinMap) {
    auto &[Map, CommonPrefix] = Entry;
    for (auto &[BuiltinName, EnumName] : Map) {
      StringRef Suffix = BuiltinName.substr(CommonPrefix->size());
      Table.GetOrAddStringOffset(Suffix);
    }
  }

  OS << formatv(R"(
// Get the LLVM intrinsic that corresponds to a builtin. This is used by the
// C front-end. The builtin name is passed in as BuiltinName, and a target
// prefix (e.g. 'ppc') is passed in as TargetPrefix.
#ifdef GET_LLVM_INTRINSIC_FOR_{}_BUILTIN
Intrinsic::ID
Intrinsic::getIntrinsicFor{}Builtin(StringRef TargetPrefix, 
                                      StringRef BuiltinName) {{
  using namespace Intrinsic;
)",
                UpperCompilerName, CompilerName);

  if (BuiltinMap.empty()) {
    OS << formatv(R"(
  return not_intrinsic;
  }
#endif  // GET_LLVM_INTRINSIC_FOR_{}_BUILTIN
)",
                  UpperCompilerName);
    return;
  }

  if (!Table.empty()) {
    Table.EmitStringTableDef(OS, "BuiltinNames");

    OS << R"(
  struct BuiltinEntry {
    ID IntrinsicID;
    unsigned StrTabOffset;
    const char *getName() const { return BuiltinNames[StrTabOffset].data(); }
    bool operator<(StringRef RHS) const {
      return strncmp(getName(), RHS.data(), RHS.size()) < 0;
    }
  };

)";
  }

  // Emit a per target table of bultin names.
  bool HasTargetIndependentBuiltins = false;
  StringRef TargetIndepndentCommonPrefix;
  for (const auto &[TargetPrefix, Entry] : BuiltinMap) {
    const auto &[Map, CommonPrefix] = Entry;
    if (!TargetPrefix.empty()) {
      OS << formatv("  // Builtins for {0}.\n", TargetPrefix);
    } else {
      OS << "  // Target independent builtins.\n";
      HasTargetIndependentBuiltins = true;
      TargetIndepndentCommonPrefix = *CommonPrefix;
    }

    // Emit the builtin table for this target prefix.
    OS << formatv("  static constexpr BuiltinEntry {}Names[] = {{\n",
                  TargetPrefix);
    for (const auto &[BuiltinName, EnumName] : Map) {
      StringRef Suffix = BuiltinName.substr(CommonPrefix->size());
      OS << formatv("    {{{}, {}}, // {}\n", EnumName,
                    *Table.GetStringOffset(Suffix), BuiltinName);
    }
    OS << formatv("  }; // {}Names\n\n", TargetPrefix);
  }

  // After emitting the builtin tables for all targets, emit a lookup table for
  // all targets. We will use binary search, similar to the table for builtin
  // names to lookup into this table.
  OS << R"(
  struct TargetEntry {
    StringLiteral TargetPrefix;
    ArrayRef<BuiltinEntry> Names;
    StringLiteral CommonPrefix;
    bool operator<(StringRef RHS) const {
      return TargetPrefix < RHS;
    };
  };
  static constexpr TargetEntry TargetTable[] = {
)";

  for (const auto &[TargetPrefix, Entry] : BuiltinMap) {
    const auto &[Map, CommonPrefix] = Entry;
    if (TargetPrefix.empty())
      continue;
    OS << formatv(R"(    {{"{0}", {0}Names, "{1}"},)", TargetPrefix,
                  CommonPrefix)
       << "\n";
  }
  OS << "  };\n";

  // Now for the actual lookup, first check the target independent table if
  // we emitted one.
  if (HasTargetIndependentBuiltins) {
    OS << formatv(R"(
  // Check if it's a target independent builtin.
  // Copy the builtin name so we can use it in consume_front without clobbering
  // if for the lookup in the target specific table.
  StringRef Suffix = BuiltinName;
  if (Suffix.consume_front("{}")) {{
    auto II = lower_bound(Names, Suffix);
    if (II != std::end(Names) && II->getName() == Suffix)
      return II->IntrinsicID;
  }
)",
                  TargetIndepndentCommonPrefix);
  }

  // If a target independent builtin was not found, lookup the target specific.
  OS << formatv(R"(
  auto TI = lower_bound(TargetTable, TargetPrefix);
  if (TI == std::end(TargetTable) || TI->TargetPrefix != TargetPrefix)
    return not_intrinsic;
  // This is the last use of BuiltinName, so no need to copy before using it in
  // consume_front.
  if (!BuiltinName.consume_front(TI->CommonPrefix))
    return not_intrinsic;
  auto II = lower_bound(TI->Names, BuiltinName);
  if (II == std::end(TI->Names) || II->getName() != BuiltinName)
    return not_intrinsic;
  return II->IntrinsicID;
}
#endif // GET_LLVM_INTRINSIC_FOR_{}_BUILTIN

)",
                UpperCompilerName);
}

static TableGen::Emitter::OptClass<IntrinsicEmitterOpt</*Enums=*/true>>
    X("gen-intrinsic-enums", "Generate intrinsic enums");

static TableGen::Emitter::OptClass<IntrinsicEmitterOpt</*Enums=*/false>>
    Y("gen-intrinsic-impl", "Generate intrinsic implementation code");
