//===-- DWARFDIE.h ----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_DWARFDIE_h_
#define SymbolFileDWARF_DWARFDIE_h_

#include "lldb/Core/dwarf.h"
#include "lldb/lldb-types.h"

struct DIERef;
class DWARFASTParser;
class DWARFAttributes;
class DWARFCompileUnit;
class DWARFDebugInfoEntry;
class DWARFDeclContext;
class DWARFDIECollection;
class SymbolFileDWARF;

class DWARFDIE {
public:
  DWARFDIE() : m_cu(nullptr), m_die(nullptr) {}

  DWARFDIE(DWARFCompileUnit *cu, DWARFDebugInfoEntry *die)
      : m_cu(cu), m_die(die) {}

  DWARFDIE(const DWARFCompileUnit *cu, DWARFDebugInfoEntry *die)
      : m_cu(const_cast<DWARFCompileUnit *>(cu)), m_die(die) {}

  DWARFDIE(DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
      : m_cu(cu), m_die(const_cast<DWARFDebugInfoEntry *>(die)) {}

  DWARFDIE(const DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die)
      : m_cu(const_cast<DWARFCompileUnit *>(cu)),
        m_die(const_cast<DWARFDebugInfoEntry *>(die)) {}

  //----------------------------------------------------------------------
  // Tests
  //----------------------------------------------------------------------
  explicit operator bool() const { return IsValid(); }

  bool IsValid() const { return m_cu && m_die; }

  bool IsStructOrClass() const;

  bool HasChildren() const;

  bool Supports_DW_AT_APPLE_objc_complete_type() const;

  //----------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------
  SymbolFileDWARF *GetDWARF() const;

  DWARFCompileUnit *GetCU() const { return m_cu; }

  DWARFDebugInfoEntry *GetDIE() const { return m_die; }

  DIERef GetDIERef() const;

  lldb_private::TypeSystem *GetTypeSystem() const;

  DWARFASTParser *GetDWARFParser() const;

  void Set(DWARFCompileUnit *cu, DWARFDebugInfoEntry *die) {
    if (cu && die) {
      m_cu = cu;
      m_die = die;
    } else {
      Clear();
    }
  }

  void Clear() {
    m_cu = nullptr;
    m_die = nullptr;
  }

  lldb::ModuleSP GetContainingDWOModule() const;

  DWARFDIE
  GetContainingDWOModuleDIE() const;

  //----------------------------------------------------------------------
  // Accessing information about a DIE
  //----------------------------------------------------------------------
  dw_tag_t Tag() const;

  const char *GetTagAsCString() const;

  dw_offset_t GetOffset() const;

  dw_offset_t GetCompileUnitRelativeOffset() const;

  //----------------------------------------------------------------------
  // Get the LLDB user ID for this DIE. This is often just the DIE offset,
  // but it might have a SymbolFileDWARF::GetID() in the high 32 bits if
  // we are doing Darwin DWARF in .o file, or DWARF stand alone debug
  // info.
  //----------------------------------------------------------------------
  lldb::user_id_t GetID() const;

  const char *GetName() const;

  const char *GetMangledName() const;

  const char *GetPubname() const;

  const char *GetQualifiedName(std::string &storage) const;

  lldb::LanguageType GetLanguage() const;

  lldb::ModuleSP GetModule() const;

  lldb_private::CompileUnit *GetLLDBCompileUnit() const;

  lldb_private::Type *ResolveType() const;

  //----------------------------------------------------------------------
  // Resolve a type by UID using this DIE's DWARF file
  //----------------------------------------------------------------------
  lldb_private::Type *ResolveTypeUID(const DIERef &die_ref) const;

  //----------------------------------------------------------------------
  // Functions for obtaining DIE relations and references
  //----------------------------------------------------------------------

  DWARFDIE
  GetParent() const;

  DWARFDIE
  GetFirstChild() const;

  DWARFDIE
  GetSibling() const;

  DWARFDIE
  GetReferencedDIE(const dw_attr_t attr) const;

  //----------------------------------------------------------------------
  // Get a another DIE from the same DWARF file as this DIE. This will
  // check the current DIE's compile unit first to see if "die_offset" is
  // in the same compile unit, and fall back to checking the DWARF file.
  //----------------------------------------------------------------------
  DWARFDIE
  GetDIE(dw_offset_t die_offset) const;

  DWARFDIE
  LookupDeepestBlock(lldb::addr_t file_addr) const;

  DWARFDIE
  GetParentDeclContextDIE() const;

  //----------------------------------------------------------------------
  // DeclContext related functions
  //----------------------------------------------------------------------
  void GetDeclContextDIEs(DWARFDIECollection &decl_context_dies) const;

  void GetDWARFDeclContext(DWARFDeclContext &dwarf_decl_ctx) const;

  void GetDWOContext(std::vector<lldb_private::CompilerContext> &context) const;

  //----------------------------------------------------------------------
  // Getting attribute values from the DIE.
  //
  // GetAttributeValueAsXXX() functions should only be used if you are
  // looking for one or two attributes on a DIE. If you are trying to
  // parse all attributes, use GetAttributes (...) instead
  //----------------------------------------------------------------------
  const char *GetAttributeValueAsString(const dw_attr_t attr,
                                        const char *fail_value) const;

  uint64_t GetAttributeValueAsUnsigned(const dw_attr_t attr,
                                       uint64_t fail_value) const;

  int64_t GetAttributeValueAsSigned(const dw_attr_t attr,
                                    int64_t fail_value) const;

  uint64_t GetAttributeValueAsReference(const dw_attr_t attr,
                                        uint64_t fail_value) const;

  DWARFDIE
  GetAttributeValueAsReferenceDIE(const dw_attr_t attr) const;

  uint64_t GetAttributeValueAsAddress(const dw_attr_t attr,
                                      uint64_t fail_value) const;

  size_t GetAttributes(DWARFAttributes &attributes, uint32_t depth = 0) const;

  bool GetDIENamesAndRanges(const char *&name, const char *&mangled,
                            DWARFRangeList &ranges, int &decl_file,
                            int &decl_line, int &decl_column, int &call_file,
                            int &call_line, int &call_column,
                            lldb_private::DWARFExpression *frame_base) const;

  //----------------------------------------------------------------------
  // Pretty printing
  //----------------------------------------------------------------------

  void Dump(lldb_private::Stream *s, const uint32_t recurse_depth) const;

  lldb_private::CompilerDecl GetDecl() const;

  lldb_private::CompilerDeclContext GetDeclContext() const;

  lldb_private::CompilerDeclContext GetContainingDeclContext() const;

protected:
  DWARFCompileUnit *m_cu;
  DWARFDebugInfoEntry *m_die;
};

bool operator==(const DWARFDIE &lhs, const DWARFDIE &rhs);
bool operator!=(const DWARFDIE &lhs, const DWARFDIE &rhs);

#endif // SymbolFileDWARF_DWARFDIE_h_
