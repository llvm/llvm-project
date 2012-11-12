//===-- SymbolFileDWARF.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARF_SymbolFileDWARF_h_
#define SymbolFileDWARF_SymbolFileDWARF_h_

// C Includes
// C++ Includes
#include <list>
#include <memory>
#include <map>
#include <vector>

// Other libraries and framework includes
#include "clang/AST/CharUnits.h"
#include "clang/AST/ExternalASTSource.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "lldb/lldb-private.h"
#include "lldb/Core/ClangForward.h"
#include "lldb/Core/ConstString.h"
#include "lldb/Core/dwarf.h"
#include "lldb/Core/DataExtractor.h"
#include "lldb/Core/Flags.h"
#include "lldb/Core/UniqueCStringMap.h"
#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Symbol/SymbolContext.h"

// Project includes
#include "DWARFDefines.h"
#include "HashedNameToDIE.h"
#include "NameToDIE.h"
#include "UniqueDWARFASTType.h"

//----------------------------------------------------------------------
// Forward Declarations for this DWARF plugin
//----------------------------------------------------------------------
class DebugMapModule;
class DWARFAbbreviationDeclaration;
class DWARFAbbreviationDeclarationSet;
class DWARFileUnit;
class DWARFDebugAbbrev;
class DWARFDebugAranges;
class DWARFDebugInfo;
class DWARFDebugInfoEntry;
class DWARFDebugLine;
class DWARFDebugPubnames;
class DWARFDebugRanges;
class DWARFDeclContext;
class DWARFDIECollection;
class DWARFFormValue;
class SymbolFileDWARFDebugMap;

struct BitfieldInfo
{
    uint32_t bit_size;
    uint32_t bit_offset;
};

typedef std::map<int64_t, BitfieldInfo> BitfieldMap;

class SymbolFileDWARF : public lldb_private::SymbolFile, public lldb_private::UserID
{
public:
    friend class SymbolFileDWARFDebugMap;    
    friend class DebugMapModule;
    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();

    static void
    Terminate();

    static const char *
    GetPluginNameStatic();

    static const char *
    GetPluginDescriptionStatic();

    static lldb_private::SymbolFile*
    CreateInstance (lldb_private::ObjectFile* obj_file);
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
                            SymbolFileDWARF(lldb_private::ObjectFile* ofile);
    virtual                 ~SymbolFileDWARF();

    virtual uint32_t        CalculateAbilities ();
    virtual void            InitializeObject();

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    virtual uint32_t        GetNumCompileUnits();
    virtual lldb::CompUnitSP ParseCompileUnitAtIndex(uint32_t index);

    virtual lldb::LanguageType ParseCompileUnitLanguage (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseCompileUnitFunctions (const lldb_private::SymbolContext& sc);
    virtual bool            ParseCompileUnitLineTable (const lldb_private::SymbolContext& sc);
    virtual bool            ParseCompileUnitSupportFiles (const lldb_private::SymbolContext& sc, lldb_private::FileSpecList& support_files);
    virtual size_t          ParseFunctionBlocks (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseTypes (const lldb_private::SymbolContext& sc);
    virtual size_t          ParseVariablesForContext (const lldb_private::SymbolContext& sc);

    virtual lldb_private::Type* ResolveTypeUID(lldb::user_id_t type_uid);
    virtual lldb::clang_type_t ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_opaque_type);

    virtual lldb_private::Type* ResolveType (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* type_die, bool assert_not_being_parsed = true);
    virtual clang::DeclContext* GetClangDeclContextContainingTypeUID (lldb::user_id_t type_uid);
    virtual clang::DeclContext* GetClangDeclContextForTypeUID (const lldb_private::SymbolContext &sc, lldb::user_id_t type_uid);

    virtual uint32_t        ResolveSymbolContext (const lldb_private::Address& so_addr, uint32_t resolve_scope, lldb_private::SymbolContext& sc);
    virtual uint32_t        ResolveSymbolContext (const lldb_private::FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindGlobalVariables(const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindGlobalVariables(const lldb_private::RegularExpression& regex, bool append, uint32_t max_matches, lldb_private::VariableList& variables);
    virtual uint32_t        FindFunctions(const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, uint32_t name_type_mask, bool include_inlines, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindFunctions(const lldb_private::RegularExpression& regex, bool include_inlines, bool append, lldb_private::SymbolContextList& sc_list);
    virtual uint32_t        FindTypes (const lldb_private::SymbolContext& sc, const lldb_private::ConstString &name, const lldb_private::ClangNamespaceDecl *namespace_decl, bool append, uint32_t max_matches, lldb_private::TypeList& types);
    virtual lldb_private::TypeList *
                            GetTypeList ();
    virtual lldb_private::ClangASTContext &
                            GetClangASTContext ();

    virtual lldb_private::ClangNamespaceDecl
            FindNamespace (const lldb_private::SymbolContext& sc, 
                           const lldb_private::ConstString &name, 
                           const lldb_private::ClangNamespaceDecl *parent_namespace_decl);


    //------------------------------------------------------------------
    // ClangASTContext callbacks for external source lookups.
    //------------------------------------------------------------------
    static void
    CompleteTagDecl (void *baton, clang::TagDecl *);
    
    static void
    CompleteObjCInterfaceDecl (void *baton, clang::ObjCInterfaceDecl *);
    
    static void
    FindExternalVisibleDeclsByName (void *baton,
                                    const clang::DeclContext *DC,
                                    clang::DeclarationName Name,
                                    llvm::SmallVectorImpl <clang::NamedDecl *> *results);

    static bool 
    LayoutRecordType (void *baton, 
                      const clang::RecordDecl *record_decl,
                      uint64_t &size, 
                      uint64_t &alignment,
                      llvm::DenseMap <const clang::FieldDecl *, uint64_t> &field_offsets,
                      llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                      llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets);

    bool 
    LayoutRecordType (const clang::RecordDecl *record_decl,
                      uint64_t &size, 
                      uint64_t &alignment,
                      llvm::DenseMap <const clang::FieldDecl *, uint64_t> &field_offsets,
                      llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &base_offsets,
                      llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> &vbase_offsets);

    struct LayoutInfo
    {
        LayoutInfo () :
            bit_size(0),
            alignment(0),
            field_offsets(),
            base_offsets(),
            vbase_offsets()
        {
        }
        uint64_t bit_size;
        uint64_t alignment;
        llvm::DenseMap <const clang::FieldDecl *, uint64_t> field_offsets;
        llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> base_offsets;
        llvm::DenseMap <const clang::CXXRecordDecl *, clang::CharUnits> vbase_offsets;
    };
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();

    virtual const char *
    GetShortPluginName();

    virtual uint32_t
    GetPluginVersion();

    // Approach 2 - count + accessor
    // Index compile units would scan the initial compile units and register
    // them with the module. This would only be done on demand if and only if
    // the compile units were needed.
    //virtual size_t        GetCompUnitCount() = 0;
    //virtual CompUnitSP    GetCompUnitAtIndex(size_t cu_idx) = 0;

    const lldb_private::DataExtractor&      get_debug_abbrev_data ();
    const lldb_private::DataExtractor&      get_debug_aranges_data ();
    const lldb_private::DataExtractor&      get_debug_frame_data ();
    const lldb_private::DataExtractor&      get_debug_info_data ();
    const lldb_private::DataExtractor&      get_debug_line_data ();
    const lldb_private::DataExtractor&      get_debug_loc_data ();
    const lldb_private::DataExtractor&      get_debug_ranges_data ();
    const lldb_private::DataExtractor&      get_debug_str_data ();
    const lldb_private::DataExtractor&      get_apple_names_data ();
    const lldb_private::DataExtractor&      get_apple_types_data ();
    const lldb_private::DataExtractor&      get_apple_namespaces_data ();
    const lldb_private::DataExtractor&      get_apple_objc_data ();


    DWARFDebugAbbrev*       DebugAbbrev();
    const DWARFDebugAbbrev* DebugAbbrev() const;

    DWARFDebugInfo*         DebugInfo();
    const DWARFDebugInfo*   DebugInfo() const;

    DWARFDebugRanges*       DebugRanges();
    const DWARFDebugRanges* DebugRanges() const;

    const lldb_private::DataExtractor&
    GetCachedSectionData (uint32_t got_flag, 
                          lldb::SectionType sect_type, 
                          lldb_private::DataExtractor &data);

    static bool
    SupportedVersion(uint16_t version);

    clang::DeclContext *
    GetCachedClangDeclContextForDIE (const DWARFDebugInfoEntry *die)
    {
        DIEToDeclContextMap::iterator pos = m_die_to_decl_ctx.find(die);
        if (pos != m_die_to_decl_ctx.end())
            return pos->second;
        else
            return NULL;
    }

    clang::DeclContext *
    GetClangDeclContextForDIE (const lldb_private::SymbolContext &sc, DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die);
    
    clang::DeclContext *
    GetClangDeclContextForDIEOffset (const lldb_private::SymbolContext &sc, dw_offset_t die_offset);
    
    clang::DeclContext *
    GetClangDeclContextContainingDIE (DWARFCompileUnit *cu, 
                                      const DWARFDebugInfoEntry *die,
                                      const DWARFDebugInfoEntry **decl_ctx_die);

    clang::DeclContext *
    GetClangDeclContextContainingDIEOffset (dw_offset_t die_offset);

    const DWARFDebugInfoEntry *
    GetDeclContextDIEContainingDIE (DWARFCompileUnit *cu, const DWARFDebugInfoEntry *die);

    void
    SearchDeclContext (const clang::DeclContext *decl_context, 
                       const char *name, 
                       llvm::SmallVectorImpl <clang::NamedDecl *> *results);
    
    lldb_private::Flags&
    GetFlags ()
    {
        return m_flags;
    }

    const lldb_private::Flags&
    GetFlags () const
    {
        return m_flags;
    }

    bool
    HasForwardDeclForClangType (lldb::clang_type_t clang_type);

protected:

    enum
    {
        flagsGotDebugAbbrevData     = (1 << 0),
        flagsGotDebugArangesData    = (1 << 1),
        flagsGotDebugFrameData      = (1 << 2),
        flagsGotDebugInfoData       = (1 << 3),
        flagsGotDebugLineData       = (1 << 4),
        flagsGotDebugLocData        = (1 << 5),
        flagsGotDebugMacInfoData    = (1 << 6),
        flagsGotDebugPubNamesData   = (1 << 7),
        flagsGotDebugPubTypesData   = (1 << 8),
        flagsGotDebugRangesData     = (1 << 9),
        flagsGotDebugStrData        = (1 << 10),
        flagsGotAppleNamesData      = (1 << 11),
        flagsGotAppleTypesData      = (1 << 12),
        flagsGotAppleNamespacesData = (1 << 13),
        flagsGotAppleObjCData       = (1 << 14)
    };
    
    bool                    NamespaceDeclMatchesThisSymbolFile (const lldb_private::ClangNamespaceDecl *namespace_decl);

    bool                    DIEIsInNamespace (const lldb_private::ClangNamespaceDecl *namespace_decl, 
                                              DWARFCompileUnit* dwarf_cu, 
                                              const DWARFDebugInfoEntry* die);

    DISALLOW_COPY_AND_ASSIGN (SymbolFileDWARF);
    lldb::CompUnitSP        ParseCompileUnit (DWARFCompileUnit* dwarf_cu, uint32_t cu_idx);
    DWARFCompileUnit*       GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit);
    DWARFCompileUnit*       GetNextUnparsedDWARFCompileUnit(DWARFCompileUnit* prev_cu);
    lldb_private::CompileUnit*      GetCompUnitForDWARFCompUnit(DWARFCompileUnit* dwarf_cu, uint32_t cu_idx = UINT32_MAX);
    bool                    GetFunction (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* func_die, lldb_private::SymbolContext& sc);
    lldb_private::Function *        ParseCompileUnitFunction (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die);
    size_t                  ParseFunctionBlocks (const lldb_private::SymbolContext& sc,
                                                 lldb_private::Block *parent_block,
                                                 DWARFCompileUnit* dwarf_cu,
                                                 const DWARFDebugInfoEntry *die,
                                                 lldb::addr_t subprogram_low_pc,
                                                 uint32_t depth);
    size_t                  ParseTypes (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool parse_siblings, bool parse_children);
    lldb::TypeSP            ParseType (const lldb_private::SymbolContext& sc, DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry *die, bool *type_is_new);
    lldb_private::Type*     ResolveTypeUID (DWARFCompileUnit* dwarf_cu, const DWARFDebugInfoEntry* die, bool assert_not_being_parsed);

    lldb::VariableSP        ParseVariableDIE(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die,
                                const lldb::addr_t func_low_pc);

    size_t                  ParseVariables(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const lldb::addr_t func_low_pc,
                                const DWARFDebugInfoEntry *die,
                                bool parse_siblings,
                                bool parse_children,
                                lldb_private::VariableList* cc_variable_list = NULL);

    class DelayedAddObjCClassProperty;
    typedef std::vector <DelayedAddObjCClassProperty> DelayedPropertyList;
    
    size_t                  ParseChildMembers(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *die,
                                lldb::clang_type_t class_clang_type,
                                const lldb::LanguageType class_language,
                                std::vector<clang::CXXBaseSpecifier *>& base_classes,
                                std::vector<int>& member_accessibilities,
                                DWARFDIECollection& member_function_dies,
                                BitfieldMap &bitfield_map,
                                DelayedPropertyList& delayed_properties,
                                lldb::AccessType &default_accessibility,
                                bool &is_a_class,
                                LayoutInfo &layout_info);

    size_t                  ParseChildParameters(
                                const lldb_private::SymbolContext& sc,
                                clang::DeclContext *containing_decl_ctx,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die,
                                bool skip_artificial,
                                bool &is_static,
                                lldb_private::TypeList* type_list,
                                std::vector<lldb::clang_type_t>& function_args,
                                std::vector<clang::ParmVarDecl*>& function_param_decls,
                                unsigned &type_quals,
                                lldb_private::ClangASTContext::TemplateParameterInfos &template_param_infos);

    size_t                  ParseChildEnumerators(
                                const lldb_private::SymbolContext& sc,
                                lldb::clang_type_t enumerator_qual_type,
                                uint32_t enumerator_byte_size,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *enum_die);

    void                    ParseChildArrayInfo(
                                const lldb_private::SymbolContext& sc,
                                DWARFCompileUnit* dwarf_cu,
                                const DWARFDebugInfoEntry *parent_die,
                                int64_t& first_index,
                                std::vector<uint64_t>& element_orders,
                                uint32_t& byte_stride,
                                uint32_t& bit_stride);

                            // Given a die_offset, figure out the symbol context representing that die.
    bool                    ResolveFunction (dw_offset_t offset,
                                             DWARFCompileUnit *&dwarf_cu,
                                             lldb_private::SymbolContextList& sc_list);
                            
    bool                    ResolveFunction (DWARFCompileUnit *cu,
                                             const DWARFDebugInfoEntry *die,
                                             lldb_private::SymbolContextList& sc_list);

    bool                    FunctionDieMatchesPartialName (
                                const DWARFDebugInfoEntry* die,
                                const DWARFCompileUnit *dwarf_cu,
                                uint32_t name_type_mask,
                                const char *partial_name,
                                const char *base_name_start,
                                const char *base_name_end);

    void                    FindFunctions(
                                const lldb_private::ConstString &name, 
                                const NameToDIE &name_to_die,
                                lldb_private::SymbolContextList& sc_list);

    void                    FindFunctions (
                                const lldb_private::RegularExpression &regex, 
                                const NameToDIE &name_to_die,
                                lldb_private::SymbolContextList& sc_list);

    void                    FindFunctions (
                                const lldb_private::RegularExpression &regex, 
                                const DWARFMappedHash::MemoryTable &memory_table,
                                lldb_private::SymbolContextList& sc_list);

    lldb::TypeSP            FindDefinitionTypeForDIE (
                                DWARFCompileUnit* dwarf_cu, 
                                const DWARFDebugInfoEntry *die, 
                                const lldb_private::ConstString &type_name);
    
    lldb::TypeSP            FindDefinitionTypeForDWARFDeclContext (
                                const DWARFDeclContext &die_decl_ctx);

    lldb::TypeSP            FindCompleteObjCDefinitionTypeForDIE (
                                const DWARFDebugInfoEntry *die, 
                                const lldb_private::ConstString &type_name,
                                bool must_be_implementation);

    bool                    Supports_DW_AT_APPLE_objc_complete_type (DWARFCompileUnit *cu);

    lldb::TypeSP            FindCompleteObjCDefinitionType (const lldb_private::ConstString &type_name,
                                                            bool header_definition_ok);

    lldb_private::Symbol *  GetObjCClassSymbol (const lldb_private::ConstString &objc_class_name);

    void                    ParseFunctions (const DIEArray &die_offsets,
                                            lldb_private::SymbolContextList& sc_list);
    lldb::TypeSP            GetTypeForDIE (DWARFCompileUnit *cu, 
                                           const DWARFDebugInfoEntry* die);

    uint32_t                FindTypes(std::vector<dw_offset_t> die_offsets, uint32_t max_matches, lldb_private::TypeList& types);

    void                    Index();
    
    void                    DumpIndexes();

    void                    SetDebugMapModule (const lldb::ModuleSP &module_sp)
                            {
                                m_debug_map_module_wp = module_sp;
                            }
    
    SymbolFileDWARFDebugMap *
                            GetDebugMapSymfile ();

    const DWARFDebugInfoEntry *
                            FindBlockContainingSpecification (dw_offset_t func_die_offset, 
                                                              dw_offset_t spec_block_die_offset,
                                                              DWARFCompileUnit **dwarf_cu_handle);

    const DWARFDebugInfoEntry *
                            FindBlockContainingSpecification (DWARFCompileUnit* dwarf_cu,
                                                              const DWARFDebugInfoEntry *die,
                                                              dw_offset_t spec_block_die_offset,
                                                              DWARFCompileUnit **dwarf_cu_handle);

    clang::NamespaceDecl *
    ResolveNamespaceDIE (DWARFCompileUnit *curr_cu, const DWARFDebugInfoEntry *die);
    
    UniqueDWARFASTTypeMap &
    GetUniqueDWARFASTTypeMap ();

    void                    LinkDeclContextToDIE (clang::DeclContext *decl_ctx,
                                                  const DWARFDebugInfoEntry *die)
                            {
                                m_die_to_decl_ctx[die] = decl_ctx;
                                // There can be many DIEs for a single decl context
                                m_decl_ctx_to_die[decl_ctx].insert(die);
                            }
    
    bool
    UserIDMatches (lldb::user_id_t uid) const
    {
        const lldb::user_id_t high_uid = uid & 0xffffffff00000000ull;
        if (high_uid)
            return high_uid == GetID();
        return true;
    }
    
    lldb::user_id_t
    MakeUserID (dw_offset_t die_offset) const
    {
        return GetID() | die_offset;
    }

    static bool
    DeclKindIsCXXClass (clang::Decl::Kind decl_kind)
    {
        switch (decl_kind)
        {
            case clang::Decl::CXXRecord:
            case clang::Decl::ClassTemplateSpecialization:
                return true;
            default:
                break;
        }
        return false;
    }
    
    bool
    ParseTemplateParameterInfos (DWARFCompileUnit* dwarf_cu,
                                 const DWARFDebugInfoEntry *parent_die,
                                 lldb_private::ClangASTContext::TemplateParameterInfos &template_param_infos);

    bool
    ParseTemplateDIE (DWARFCompileUnit* dwarf_cu,
                      const DWARFDebugInfoEntry *die,
                      lldb_private::ClangASTContext::TemplateParameterInfos &template_param_infos);

    clang::ClassTemplateDecl *
    ParseClassTemplateDecl (clang::DeclContext *decl_ctx,
                            lldb::AccessType access_type,
                            const char *parent_name,
                            int tag_decl_kind,
                            const lldb_private::ClangASTContext::TemplateParameterInfos &template_param_infos);

    bool
    DIEDeclContextsMatch (DWARFCompileUnit* cu1, const DWARFDebugInfoEntry *die1,
                          DWARFCompileUnit* cu2, const DWARFDebugInfoEntry *die2);
    
    bool
    ClassContainsSelector (DWARFCompileUnit *dwarf_cu,
                           const DWARFDebugInfoEntry *class_die,
                           const lldb_private::ConstString &selector);

    bool
    CopyUniqueClassMethodTypes (lldb_private::Type *class_type,
                                DWARFCompileUnit* src_cu,
                                const DWARFDebugInfoEntry *src_class_die,
                                DWARFCompileUnit* dst_cu,
                                const DWARFDebugInfoEntry *dst_class_die);

    lldb::ModuleWP                  m_debug_map_module_wp;
    SymbolFileDWARFDebugMap *       m_debug_map_symfile;
    clang::TranslationUnitDecl *    m_clang_tu_decl;
    lldb_private::Flags             m_flags;
    lldb_private::DataExtractor     m_dwarf_data; 
    lldb_private::DataExtractor     m_data_debug_abbrev;
    lldb_private::DataExtractor     m_data_debug_aranges;
    lldb_private::DataExtractor     m_data_debug_frame;
    lldb_private::DataExtractor     m_data_debug_info;
    lldb_private::DataExtractor     m_data_debug_line;
    lldb_private::DataExtractor     m_data_debug_loc;
    lldb_private::DataExtractor     m_data_debug_ranges;
    lldb_private::DataExtractor     m_data_debug_str;
    lldb_private::DataExtractor     m_data_apple_names;
    lldb_private::DataExtractor     m_data_apple_types;
    lldb_private::DataExtractor     m_data_apple_namespaces;
    lldb_private::DataExtractor     m_data_apple_objc;

    // The auto_ptr items below are generated on demand if and when someone accesses
    // them through a non const version of this class.
    std::auto_ptr<DWARFDebugAbbrev>     m_abbr;
    std::auto_ptr<DWARFDebugInfo>       m_info;
    std::auto_ptr<DWARFDebugLine>       m_line;
    std::auto_ptr<DWARFMappedHash::MemoryTable> m_apple_names_ap;
    std::auto_ptr<DWARFMappedHash::MemoryTable> m_apple_types_ap;
    std::auto_ptr<DWARFMappedHash::MemoryTable> m_apple_namespaces_ap;
    std::auto_ptr<DWARFMappedHash::MemoryTable> m_apple_objc_ap;
    NameToDIE                           m_function_basename_index;  // All concrete functions
    NameToDIE                           m_function_fullname_index;  // All concrete functions
    NameToDIE                           m_function_method_index;    // All inlined functions
    NameToDIE                           m_function_selector_index;  // All method names for functions of classes
    NameToDIE                           m_objc_class_selectors_index; // Given a class name, find all selectors for the class
    NameToDIE                           m_global_index;             // Global and static variables
    NameToDIE                           m_type_index;               // All type DIE offsets
    NameToDIE                           m_namespace_index;          // All type DIE offsets
    bool                                m_indexed:1,
                                        m_is_external_ast_source:1,
                                        m_using_apple_tables:1;
    lldb_private::LazyBool              m_supports_DW_AT_APPLE_objc_complete_type;

    std::auto_ptr<DWARFDebugRanges>     m_ranges;
    UniqueDWARFASTTypeMap m_unique_ast_type_map;
    typedef llvm::SmallPtrSet<const DWARFDebugInfoEntry *, 4> DIEPointerSet;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, clang::DeclContext *> DIEToDeclContextMap;
    typedef llvm::DenseMap<const clang::DeclContext *, DIEPointerSet> DeclContextToDIEMap;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb_private::Type *> DIEToTypePtr;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::VariableSP> DIEToVariableSP;
    typedef llvm::DenseMap<const DWARFDebugInfoEntry *, lldb::clang_type_t> DIEToClangType;
    typedef llvm::DenseMap<lldb::clang_type_t, const DWARFDebugInfoEntry *> ClangTypeToDIE;
    typedef llvm::DenseMap<const clang::RecordDecl *, LayoutInfo> RecordDeclToLayoutMap;
    DIEToDeclContextMap m_die_to_decl_ctx;
    DeclContextToDIEMap m_decl_ctx_to_die;
    DIEToTypePtr m_die_to_type;
    DIEToVariableSP m_die_to_variable_sp;
    DIEToClangType m_forward_decl_die_to_clang_type;
    ClangTypeToDIE m_forward_decl_clang_type_to_die;
    RecordDeclToLayoutMap m_record_decl_to_layout_map;
};

#endif  // SymbolFileDWARF_SymbolFileDWARF_h_
