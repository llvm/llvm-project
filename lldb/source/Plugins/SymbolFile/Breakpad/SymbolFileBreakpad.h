//===-- SymbolFileBreakpad.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_PLUGINS_SYMBOLFILE_BREAKPAD_SYMBOLFILEBREAKPAD_H
#define LLDB_PLUGINS_SYMBOLFILE_BREAKPAD_SYMBOLFILEBREAKPAD_H

#include "Plugins/ObjectFile/Breakpad/BreakpadRecords.h"
#include "lldb/Core/FileSpecList.h"
#include "lldb/Symbol/LineTable.h"
#include "lldb/Symbol/SymbolFile.h"

namespace lldb_private {

namespace breakpad {

class SymbolFileBreakpad : public SymbolFile {
public:
  //------------------------------------------------------------------
  // Static Functions
  //------------------------------------------------------------------
  static void Initialize();
  static void Terminate();
  static void DebuggerInitialize(Debugger &debugger) {}
  static ConstString GetPluginNameStatic();

  static const char *GetPluginDescriptionStatic() {
    return "Breakpad debug symbol file reader.";
  }

  static SymbolFile *CreateInstance(ObjectFile *obj_file) {
    return new SymbolFileBreakpad(obj_file);
  }

  //------------------------------------------------------------------
  // Constructors and Destructors
  //------------------------------------------------------------------
  SymbolFileBreakpad(ObjectFile *object_file) : SymbolFile(object_file) {}

  ~SymbolFileBreakpad() override {}

  uint32_t CalculateAbilities() override;

  void InitializeObject() override {}

  //------------------------------------------------------------------
  // Compile Unit function calls
  //------------------------------------------------------------------

  uint32_t GetNumCompileUnits() override;

  lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index) override;

  lldb::LanguageType ParseLanguage(CompileUnit &comp_unit) override {
    return lldb::eLanguageTypeUnknown;
  }

  size_t ParseFunctions(CompileUnit &comp_unit) override;

  bool ParseLineTable(CompileUnit &comp_unit) override;

  bool ParseDebugMacros(CompileUnit &comp_unit) override { return false; }

  bool ParseSupportFiles(CompileUnit &comp_unit,
                         FileSpecList &support_files) override;
  size_t ParseTypes(CompileUnit &cu) override { return 0; }

  bool
  ParseImportedModules(const SymbolContext &sc,
                       std::vector<ConstString> &imported_modules) override {
    return false;
  }

  size_t ParseBlocksRecursive(Function &func) override { return 0; }

  uint32_t FindGlobalVariables(const ConstString &name,
                               const CompilerDeclContext *parent_decl_ctx,
                               uint32_t max_matches,
                               VariableList &variables) override {
    return 0;
  }

  size_t ParseVariablesForContext(const SymbolContext &sc) override {
    return 0;
  }
  Type *ResolveTypeUID(lldb::user_id_t type_uid) override { return nullptr; }
  llvm::Optional<ArrayInfo> GetDynamicArrayInfoForUID(
      lldb::user_id_t type_uid,
      const lldb_private::ExecutionContext *exe_ctx) override {
    return llvm::None;
  }

  bool CompleteType(CompilerType &compiler_type) override { return false; }
  uint32_t ResolveSymbolContext(const Address &so_addr,
                                lldb::SymbolContextItem resolve_scope,
                                SymbolContext &sc) override;

  uint32_t ResolveSymbolContext(const FileSpec &file_spec, uint32_t line,
                                bool check_inlines,
                                lldb::SymbolContextItem resolve_scope,
                                SymbolContextList &sc_list) override;

  size_t GetTypes(SymbolContextScope *sc_scope, lldb::TypeClass type_mask,
                  TypeList &type_list) override {
    return 0;
  }

  uint32_t FindFunctions(const ConstString &name,
                         const CompilerDeclContext *parent_decl_ctx,
                         lldb::FunctionNameType name_type_mask,
                         bool include_inlines, bool append,
                         SymbolContextList &sc_list) override;

  uint32_t FindFunctions(const RegularExpression &regex, bool include_inlines,
                         bool append, SymbolContextList &sc_list) override;

  uint32_t FindTypes(const ConstString &name,
                     const CompilerDeclContext *parent_decl_ctx, bool append,
                     uint32_t max_matches,
                     llvm::DenseSet<SymbolFile *> &searched_symbol_files,
                     TypeMap &types) override;

  size_t FindTypes(const std::vector<CompilerContext> &context, bool append,
                   TypeMap &types) override;

  TypeSystem *GetTypeSystemForLanguage(lldb::LanguageType language) override {
    return nullptr;
  }

  CompilerDeclContext
  FindNamespace(const ConstString &name,
                const CompilerDeclContext *parent_decl_ctx) override {
    return CompilerDeclContext();
  }

  void AddSymbols(Symtab &symtab) override;

  ConstString GetPluginName() override { return GetPluginNameStatic(); }
  uint32_t GetPluginVersion() override { return 1; }

private:
  // A class representing a position in the breakpad file. Useful for
  // remembering the position so we can go back to it later and parse more data.
  // Can be converted to/from a LineIterator, but it has a much smaller memory
  // footprint.
  struct Bookmark {
    uint32_t section;
    size_t offset;
  };

  // At iterator class for simplifying algorithms reading data from the breakpad
  // file. It iterates over all records (lines) in the sections of a given type.
  // It also supports saving a specific position (via the GetBookmark() method)
  // and then resuming from it afterwards.
  class LineIterator;

  // Return an iterator range for all records in the given object file of the
  // given type.
  llvm::iterator_range<LineIterator> lines(Record::Kind section_type);

  // Breakpad files do not contain sufficient information to correctly
  // reconstruct compile units. The approach chosen here is to treat each
  // function as a compile unit. The compile unit name is the name if the first
  // line entry belonging to this function.
  // This class is our internal representation of a compile unit. It stores the
  // CompileUnit object and a bookmark pointing to the FUNC record of the
  // compile unit function. It also lazily construct the list of support files
  // and line table entries for the compile unit, when these are needed.
  class CompUnitData {
  public:
    CompUnitData(Bookmark bookmark) : bookmark(bookmark) {}

    CompUnitData() = default;
    CompUnitData(const CompUnitData &rhs) : bookmark(rhs.bookmark) {}
    CompUnitData &operator=(const CompUnitData &rhs) {
      bookmark = rhs.bookmark;
      support_files.reset();
      line_table_up.reset();
      return *this;
    }
    friend bool operator<(const CompUnitData &lhs, const CompUnitData &rhs) {
      return std::tie(lhs.bookmark.section, lhs.bookmark.offset) <
             std::tie(rhs.bookmark.section, rhs.bookmark.offset);
    }

    Bookmark bookmark;
    llvm::Optional<FileSpecList> support_files;
    std::unique_ptr<LineTable> line_table_up;

  };

  SymbolVendor &GetSymbolVendor();
  lldb::addr_t GetBaseFileAddress();
  void ParseFileRecords();
  void ParseCUData();
  void ParseLineTableAndSupportFiles(CompileUnit &cu, CompUnitData &data);

  using CompUnitMap = RangeDataVector<lldb::addr_t, lldb::addr_t, CompUnitData>;

  llvm::Optional<std::vector<FileSpec>> m_files;
  llvm::Optional<CompUnitMap> m_cu_data;
};

} // namespace breakpad
} // namespace lldb_private

#endif
