//===-- RegisterTypeBuilderClang.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H
#define LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H

#include "lldb/Target/RegisterTypeBuilder.h"
#include "lldb/Target/Target.h"
#include "clang/AST/ExternalASTSource.h"

namespace lldb_private {
class RegisterTypeBuilderClang : public RegisterTypeBuilder {
public:
  RegisterTypeBuilderClang(Target &target);

  static void Initialize();
  static void Terminate();
  static llvm::StringRef GetPluginNameStatic() {
    return "register-types-clang";
  }
  llvm::StringRef GetPluginName() override { return GetPluginNameStatic(); }
  static llvm::StringRef GetPluginDescriptionStatic() {
    return "Create register types using TypeSystemClang";
  }
  static lldb::RegisterTypeBuilderSP CreateInstance(Target &target);

  CompilerType GetRegisterType(const std::string &name,
                               const lldb_private::RegisterFlags &flags,
                               uint32_t byte_size) override;

private:
  /// This external AST is used to override the layout of bitfield structs
  /// created from sets of register "flags".
  ///
  /// We have two goals with register display:
  /// 1. Most significant to least significant display order, to match
  /// architecure
  ///    manuals.
  /// 2. Correctly extracting field values.
  ///
  /// Goal 1 is achieved by building the struct with the most significant field
  /// as the first member and the least significant as the last member.
  ///
  /// The default bit position of those fields is that the first member is bit
  /// 0, and the last is bit N. This is LSB to MSB, so the replacement layouts
  /// in this external AST reverse that to be MSB to LSB. This achieves goal 2.
  class RegisterExternalASTSource : public clang::ExternalASTSource {
  public:
    struct LayoutInfo {
      uint64_t size_bytes = 0;
      llvm::DenseMap<const clang::FieldDecl *, uint64_t> field_offsets;
    };
    llvm::DenseMap<const clang::RecordDecl *, LayoutInfo> m_struct_layouts;

    bool layoutRecordType(
        const clang::RecordDecl *record, uint64_t &size, uint64_t &align,
        llvm::DenseMap<const clang::FieldDecl *, uint64_t> &field_offsets,
        llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
            &base_offsets,
        llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
            &vbase_offsets) override {
      auto it = m_struct_layouts.find(record);
      if (it == m_struct_layouts.end())
        return false;

      size = it->second.size_bytes * 8;
      align = size;
      field_offsets = it->second.field_offsets;
      base_offsets.clear();
      vbase_offsets.clear();
      return true;
    }
  };

  // This is created the first time a register type is requested, then handed
  // to the type system. We keep a reference to it so we can add more layouts
  // as more register types are requested.
  llvm::IntrusiveRefCntPtr<RegisterExternalASTSource> m_external_ast;
  Target &m_target;
};
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_REGISTERTYPEBUILDER_REGISTERTYPEBUILDERCLANG_H
