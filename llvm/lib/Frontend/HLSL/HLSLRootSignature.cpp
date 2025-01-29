//===- HLSLRootSignature.cpp - HLSL Root Signature helper objects ----------===//
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
      std::visit(
        OverloadBuilds{
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
  return MDNode::get(Ctx, {MDString::get(Ctx, "DescriptorTable")});
}

MDNode *MetadataBuilder::BuildDescriptorTableClause(const DescriptorTableClause &Clause) {
  return MDNode::get(Ctx, {ClauseTypeToName(Ctx, Clause.Type)});
}

} // namespace root_signature
} // namespace hlsl
} // namespace llvm

