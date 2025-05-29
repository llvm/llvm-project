//===- HLSLRootSignature.cpp - HLSL Root Signature helper objects ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helpers for working with HLSL Root Signatures.
///
//===----------------------------------------------------------------------===//

#include "llvm/Frontend/HLSL/HLSLRootSignature.h"
#include "llvm/ADT/bit.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace hlsl {
namespace rootsig {

static raw_ostream &operator<<(raw_ostream &OS, const Register &Reg) {
  switch (Reg.ViewType) {
  case RegisterType::BReg:
    OS << "b";
    break;
  case RegisterType::TReg:
    OS << "t";
    break;
  case RegisterType::UReg:
    OS << "u";
    break;
  case RegisterType::SReg:
    OS << "s";
    break;
  }
  OS << Reg.Number;
  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const ShaderVisibility &Visibility) {
  switch (Visibility) {
  case ShaderVisibility::All:
    OS << "All";
    break;
  case ShaderVisibility::Vertex:
    OS << "Vertex";
    break;
  case ShaderVisibility::Hull:
    OS << "Hull";
    break;
  case ShaderVisibility::Domain:
    OS << "Domain";
    break;
  case ShaderVisibility::Geometry:
    OS << "Geometry";
    break;
  case ShaderVisibility::Pixel:
    OS << "Pixel";
    break;
  case ShaderVisibility::Amplification:
    OS << "Amplification";
    break;
  case ShaderVisibility::Mesh:
    OS << "Mesh";
    break;
  }

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS, const ClauseType &Type) {
  switch (Type) {
  case ClauseType::CBuffer:
    OS << "CBV";
    break;
  case ClauseType::SRV:
    OS << "SRV";
    break;
  case ClauseType::UAV:
    OS << "UAV";
    break;
  case ClauseType::Sampler:
    OS << "Sampler";
    break;
  }

  return OS;
}

static raw_ostream &operator<<(raw_ostream &OS,
                               const DescriptorRangeFlags &Flags) {
  bool FlagSet = false;
  unsigned Remaining = llvm::to_underlying(Flags);
  while (Remaining) {
    unsigned Bit = 1u << llvm::countr_zero(Remaining);
    if (Remaining & Bit) {
      if (FlagSet)
        OS << " | ";

      switch (static_cast<DescriptorRangeFlags>(Bit)) {
      case DescriptorRangeFlags::DescriptorsVolatile:
        OS << "DescriptorsVolatile";
        break;
      case DescriptorRangeFlags::DataVolatile:
        OS << "DataVolatile";
        break;
      case DescriptorRangeFlags::DataStaticWhileSetAtExecute:
        OS << "DataStaticWhileSetAtExecute";
        break;
      case DescriptorRangeFlags::DataStatic:
        OS << "DataStatic";
        break;
      case DescriptorRangeFlags::DescriptorsStaticKeepingBufferBoundsChecks:
        OS << "DescriptorsStaticKeepingBufferBoundsChecks";
        break;
      default:
        OS << "invalid: " << Bit;
        break;
      }

      FlagSet = true;
    }
    Remaining &= ~Bit;
  }

  if (!FlagSet)
    OS << "None";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const DescriptorTable &Table) {
  OS << "DescriptorTable(numClauses = " << Table.NumClauses
     << ", visibility = " << Table.Visibility << ")";

  return OS;
}

raw_ostream &operator<<(raw_ostream &OS, const DescriptorTableClause &Clause) {
  OS << Clause.Type << "(" << Clause.Reg
     << ", numDescriptors = " << Clause.NumDescriptors
     << ", space = " << Clause.Space << ", offset = ";
  if (Clause.Offset == DescriptorTableOffsetAppend)
    OS << "DescriptorTableOffsetAppend";
  else
    OS << Clause.Offset;
  OS << ", flags = " << Clause.Flags << ")";

  return OS;
}

void dumpRootElements(raw_ostream &OS, ArrayRef<RootElement> Elements) {
  OS << "RootElements{";
  bool First = true;
  for (const RootElement &Element : Elements) {
    if (!First)
      OS << ",";
    OS << " ";
    if (const auto &Clause = std::get_if<DescriptorTableClause>(&Element))
      OS << *Clause;
    if (const auto &Table = std::get_if<DescriptorTable>(&Element))
      OS << *Table;
    First = false;
  }
  OS << "}";
}

MDNode *MetadataBuilder::BuildRootSignature() {
  for (const RootElement &Element : Elements) {
    MDNode *ElementMD = nullptr;
    if (const auto &Clause = std::get_if<DescriptorTableClause>(&Element))
      ElementMD = BuildDescriptorTableClause(*Clause);
    if (const auto &Table = std::get_if<DescriptorTable>(&Element))
      ElementMD = BuildDescriptorTable(*Table);

    // FIXME(#126586): remove once all RootElemnt variants are handled in a
    // visit or otherwise
    assert(ElementMD != nullptr &&
           "Constructed an unhandled root element type.");

    GeneratedMetadata.push_back(ElementMD);
  }

  return MDNode::get(Ctx, GeneratedMetadata);
}

MDNode *MetadataBuilder::BuildDescriptorTable(const DescriptorTable &Table) {
  IRBuilder<> Builder(Ctx);
  SmallVector<Metadata *> TableOperands;
  // Set the mandatory arguments
  TableOperands.push_back(MDString::get(Ctx, "DescriptorTable"));
  TableOperands.push_back(ConstantAsMetadata::get(
      Builder.getInt32(llvm::to_underlying(Table.Visibility))));

  // Remaining operands are references to the table's clauses. The in-memory
  // representation of the Root Elements created from parsing will ensure that
  // the previous N elements are the clauses for this table.
  assert(Table.NumClauses <= GeneratedMetadata.size() &&
         "Table expected all owned clauses to be generated already");
  // So, add a refence to each clause to our operands
  TableOperands.append(GeneratedMetadata.end() - Table.NumClauses,
                       GeneratedMetadata.end());
  // Then, remove those clauses from the general list of Root Elements
  GeneratedMetadata.pop_back_n(Table.NumClauses);

  return MDNode::get(Ctx, TableOperands);
}

MDNode *MetadataBuilder::BuildDescriptorTableClause(
    const DescriptorTableClause &Clause) {
  IRBuilder<> Builder(Ctx);
  std::string Name;
  llvm::raw_string_ostream OS(Name);
  OS << Clause.Type;
  return MDNode::get(
      Ctx, {
               MDString::get(Ctx, OS.str()),
               ConstantAsMetadata::get(Builder.getInt32(Clause.NumDescriptors)),
               ConstantAsMetadata::get(Builder.getInt32(Clause.Reg.Number)),
               ConstantAsMetadata::get(Builder.getInt32(Clause.Space)),
               ConstantAsMetadata::get(Builder.getInt32(Clause.Offset)),
               ConstantAsMetadata::get(
                   Builder.getInt32(llvm::to_underlying(Clause.Flags))),
           });
}

std::optional<const RangeInfo *>
ResourceRange::getOverlapping(const RangeInfo &Info) const {
  MapT::const_iterator Interval = Intervals.find(Info.LowerBound);
  if (!Interval.valid() || Info.UpperBound < Interval.start())
    return std::nullopt;
  return Interval.value();
}

const RangeInfo *ResourceRange::lookup(uint32_t X) const {
  return Intervals.lookup(X, nullptr);
}

std::optional<const RangeInfo *> ResourceRange::insert(const RangeInfo &Info) {
  uint32_t LowerBound = Info.LowerBound;
  uint32_t UpperBound = Info.UpperBound;

  std::optional<const RangeInfo *> Res = std::nullopt;
  MapT::iterator Interval = Intervals.begin();

  while (true) {
    if (UpperBound < LowerBound)
      break;

    Interval.advanceTo(LowerBound);
    if (!Interval.valid()) // No interval found
      break;

    // Let Interval = [x;y] and [LowerBound;UpperBound] = [a;b] and note that
    // a <= y implicitly from Intervals.find(LowerBound)
    if (UpperBound < Interval.start())
      break; // found interval does not overlap with inserted one

    if (!Res.has_value()) // Update to be the first found intersection
      Res = Interval.value();

    if (Interval.start() <= LowerBound && UpperBound <= Interval.stop()) {
      // x <= a <= b <= y implies that [a;b] is covered by [x;y]
      //  -> so we don't need to insert this, report an overlap
      return Res;
    } else if (LowerBound <= Interval.start() &&
               Interval.stop() <= UpperBound) {
      // a <= x <= y <= b implies that [x;y] is covered by [a;b]
      //  -> so remove the existing interval that we will cover with the
      //  overwrite
      Interval.erase();
    } else if (LowerBound < Interval.start() && UpperBound <= Interval.stop()) {
      // a < x <= b <= y implies that [a; x] is not covered but [x;b] is
      //  -> so set b = x - 1 such that [a;x-1] is now the interval to insert
      UpperBound = Interval.start() - 1;
    } else if (Interval.start() <= LowerBound && Interval.stop() < UpperBound) {
      // a < x <= b <= y implies that [y; b] is not covered but [a;y] is
      //  -> so set a = y + 1 such that [y+1;b] is now the interval to insert
      LowerBound = Interval.stop() + 1;
    }
  }

  assert(LowerBound <= UpperBound && "Attempting to insert an empty interval");
  Intervals.insert(LowerBound, UpperBound, &Info);
  return Res;
}

} // namespace rootsig
} // namespace hlsl
} // namespace llvm
