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

#include "DWARFBaseDIE.h"

class DWARFDIE : public DWARFBaseDIE {
public:
  using DWARFBaseDIE::DWARFBaseDIE;

  //----------------------------------------------------------------------
  // Tests
  //----------------------------------------------------------------------
  bool IsStructClassOrUnion() const;

  //----------------------------------------------------------------------
  // Accessors
  //----------------------------------------------------------------------
  lldb::ModuleSP GetContainingDWOModule() const;

  DWARFDIE
  GetContainingDWOModuleDIE() const;

  //----------------------------------------------------------------------
  // Accessing information about a DIE
  //----------------------------------------------------------------------
  const char *GetMangledName() const;

  const char *GetPubname() const;

  const char *GetQualifiedName(std::string &storage) const;

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
  using DWARFBaseDIE::GetDIE;

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
  DWARFDIE
  GetAttributeValueAsReferenceDIE(const dw_attr_t attr) const;

  bool GetDIENamesAndRanges(const char *&name, const char *&mangled,
                            DWARFRangeList &ranges, int &decl_file,
                            int &decl_line, int &decl_column, int &call_file,
                            int &call_line, int &call_column,
                            lldb_private::DWARFExpression *frame_base) const;

  //----------------------------------------------------------------------
  // CompilerDecl related functions
  //----------------------------------------------------------------------

  lldb_private::CompilerDecl GetDecl() const;

  lldb_private::CompilerDeclContext GetDeclContext() const;

  lldb_private::CompilerDeclContext GetContainingDeclContext() const;
};

#endif // SymbolFileDWARF_DWARFDIE_h_
