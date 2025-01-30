//===- HLSLRootSignature.cpp - HLSL Root Signature helper objects
//----------===//
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
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"

namespace llvm {
namespace hlsl {
namespace root_signature {

// Static helper functions

static MDString *ClauseTypeToName(LLVMContext &Ctx, ClauseType Type) {
  StringRef Name;
  switch (Type) {
  case ClauseType::CBuffer:
    Name = "CBV";
    break;
  case ClauseType::SRV:
    Name = "SRV";
    break;
  case ClauseType::UAV:
    Name = "UAV";
    break;
  case ClauseType::Sampler:
    Name = "Sampler";
    break;
  }
  return MDString::get(Ctx, Name);
}

// Helper struct so that we can use the overloaded notation of std::visit
template <class... Ts> struct OverloadBuilds : Ts... {
  using Ts::operator()...;
};
template <class... Ts> OverloadBuilds(Ts...) -> OverloadBuilds<Ts...>;

MDNode *MetadataBuilder::BuildRootSignature() {
  for (const RootElement &Element : Elements) {
    MDNode *ElementMD =
        std::visit(OverloadBuilds{
                       [&](DescriptorTable Table) -> MDNode * {
                         return BuildDescriptorTable(Table);
                       },
                       [&](DescriptorTableClause Clause) -> MDNode * {
                         return BuildDescriptorTableClause(Clause);
                       },
                   },
                   Element);
    GeneratedMetadata.push_back(ElementMD);
  }

  return MDNode::get(Ctx, GeneratedMetadata);
}

MDNode *MetadataBuilder::BuildDescriptorTable(const DescriptorTable &Table) {
  IRBuilder<> B(Ctx);
  SmallVector<Metadata *> TableOperands;
  // Set the mandatory arguments
  TableOperands.push_back(MDString::get(Ctx, "DescriptorTable"));
  TableOperands.push_back(ConstantAsMetadata::get(
      B.getInt32(llvm::to_underlying(Table.Visibility))));

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
  IRBuilder<> B(Ctx);
  return MDNode::get(
      Ctx, {
               ClauseTypeToName(Ctx, Clause.Type),
               ConstantAsMetadata::get(B.getInt32(Clause.NumDescriptors)),
               ConstantAsMetadata::get(B.getInt32(Clause.Register.Number)),
               ConstantAsMetadata::get(B.getInt32(Clause.Space)),
               ConstantAsMetadata::get(
                   B.getInt32(llvm::to_underlying(Clause.Offset))),
               ConstantAsMetadata::get(
                   B.getInt32(llvm::to_underlying(Clause.Flags))),
           });
}

} // namespace root_signature
} // namespace hlsl
} // namespace llvm
