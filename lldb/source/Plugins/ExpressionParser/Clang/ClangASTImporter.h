//===-- ClangASTImporter.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_EXPRESSIONPARSER_CLANG_CLANGASTIMPORTER_H
#define LLDB_SOURCE_PLUGINS_EXPRESSIONPARSER_CLANG_CLANGASTIMPORTER_H

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "clang/AST/ASTImporter.h"
#include "clang/AST/CharUnits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/FileSystemOptions.h"

#include "lldb/Host/FileSystem.h"
#include "lldb/Symbol/CompilerDeclContext.h"
#include "lldb/lldb-types.h"

#include "Plugins/ExpressionParser/Clang/CxxModuleHandler.h"

#include "llvm/ADT/DenseMap.h"

namespace lldb_private {

class ClangASTMetadata;
class TypeSystemClang;

class ClangASTImporter {
public:
  struct LayoutInfo {
    LayoutInfo() = default;
    typedef llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
        OffsetMap;

    uint64_t bit_size = 0;
    uint64_t alignment = 0;
    llvm::DenseMap<const clang::FieldDecl *, uint64_t> field_offsets;
    OffsetMap base_offsets;
    OffsetMap vbase_offsets;
  };

  ClangASTImporter()
      : m_file_manager(clang::FileSystemOptions(),
                       FileSystem::Instance().GetVirtualFileSystem()) {}

  CompilerType CopyType(TypeSystemClang &dst, const CompilerType &src_type);

  clang::Decl *CopyDecl(clang::ASTContext *dst_ctx, clang::Decl *decl);

  CompilerType DeportType(TypeSystemClang &dst, const CompilerType &src_type);

  clang::Decl *DeportDecl(clang::ASTContext *dst_ctx, clang::Decl *decl);

  /// Sets the layout for the given RecordDecl. The layout will later be
  /// used by Clang's during code generation. Not calling this function for
  /// a RecordDecl will cause that Clang's codegen tries to layout the
  /// record by itself.
  ///
  /// \param decl The RecordDecl to set the layout for.
  /// \param layout The layout for the record.
  void SetRecordLayout(clang::RecordDecl *decl, const LayoutInfo &layout);

  bool LayoutRecordType(
      const clang::RecordDecl *record_decl, uint64_t &bit_size,
      uint64_t &alignment,
      llvm::DenseMap<const clang::FieldDecl *, uint64_t> &field_offsets,
      llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
          &base_offsets,
      llvm::DenseMap<const clang::CXXRecordDecl *, clang::CharUnits>
          &vbase_offsets);

  bool CanImport(const CompilerType &type);

  bool Import(const CompilerType &type);

  bool CompleteType(const CompilerType &compiler_type);

  bool CompleteTagDecl(clang::TagDecl *decl);

  bool CompleteTagDeclWithOrigin(clang::TagDecl *decl, clang::TagDecl *origin);

  bool CompleteObjCInterfaceDecl(clang::ObjCInterfaceDecl *interface_decl);

  bool CompleteAndFetchChildren(clang::QualType type);

  bool RequireCompleteType(clang::QualType type);

  void SetDeclOrigin(const clang::Decl *decl, clang::Decl *original_decl);

  ClangASTMetadata *GetDeclMetadata(const clang::Decl *decl);

  //
  // Namespace maps
  //

  typedef std::pair<lldb::ModuleSP, CompilerDeclContext> NamespaceMapItem;
  typedef std::vector<NamespaceMapItem> NamespaceMap;
  typedef std::shared_ptr<NamespaceMap> NamespaceMapSP;

  void RegisterNamespaceMap(const clang::NamespaceDecl *decl,
                            NamespaceMapSP &namespace_map);

  NamespaceMapSP GetNamespaceMap(const clang::NamespaceDecl *decl);

  void BuildNamespaceMap(const clang::NamespaceDecl *decl);

  //
  // Completers for maps
  //

  class MapCompleter {
  public:
    virtual ~MapCompleter();

    virtual void CompleteNamespaceMap(NamespaceMapSP &namespace_map,
                                      ConstString name,
                                      NamespaceMapSP &parent_map) const = 0;
  };

  void InstallMapCompleter(clang::ASTContext *dst_ctx,
                           MapCompleter &completer) {
    ASTContextMetadataSP context_md;
    ContextMetadataMap::iterator context_md_iter = m_metadata_map.find(dst_ctx);

    if (context_md_iter == m_metadata_map.end()) {
      context_md = ASTContextMetadataSP(new ASTContextMetadata(dst_ctx));
      m_metadata_map[dst_ctx] = context_md;
    } else {
      context_md = context_md_iter->second;
    }

    context_md->m_map_completer = &completer;
  }

  void ForgetDestination(clang::ASTContext *dst_ctx);
  void ForgetSource(clang::ASTContext *dst_ctx, clang::ASTContext *src_ctx);

  struct DeclOrigin {
    DeclOrigin() : ctx(nullptr), decl(nullptr) {}

    DeclOrigin(clang::ASTContext *_ctx, clang::Decl *_decl)
        : ctx(_ctx), decl(_decl) {}

    DeclOrigin(const DeclOrigin &rhs) {
      ctx = rhs.ctx;
      decl = rhs.decl;
    }

    void operator=(const DeclOrigin &rhs) {
      ctx = rhs.ctx;
      decl = rhs.decl;
    }

    bool Valid() const { return (ctx != nullptr || decl != nullptr); }

    clang::ASTContext *ctx;
    clang::Decl *decl;
  };

  /// Listener interface used by the ASTImporterDelegate to inform other code
  /// about decls that have been imported the first time.
  struct NewDeclListener {
    virtual ~NewDeclListener() = default;
    /// A decl has been imported for the first time.
    virtual void NewDeclImported(clang::Decl *from, clang::Decl *to) = 0;
  };

  /// ASTImporter that intercepts and records the import process of the
  /// underlying ASTImporter.
  ///
  /// This class updates the map from declarations to their original
  /// declarations and can record declarations that have been imported in a
  /// certain interval.
  ///
  /// When intercepting a declaration import, the ASTImporterDelegate uses the
  /// CxxModuleHandler to replace any missing or malformed declarations with
  /// their counterpart from a C++ module.
  struct ASTImporterDelegate : public clang::ASTImporter {
    ASTImporterDelegate(ClangASTImporter &master, clang::ASTContext *target_ctx,
                        clang::ASTContext *source_ctx)
        : clang::ASTImporter(*target_ctx, master.m_file_manager, *source_ctx,
                             master.m_file_manager, true /*minimal*/),
          m_master(master), m_source_ctx(source_ctx) {
      setODRHandling(clang::ASTImporter::ODRHandlingType::Liberal);
    }

    /// Scope guard that attaches a CxxModuleHandler to an ASTImporterDelegate
    /// and deattaches it at the end of the scope. Supports being used multiple
    /// times on the same ASTImporterDelegate instance in nested scopes.
    class CxxModuleScope {
      /// The handler we attach to the ASTImporterDelegate.
      CxxModuleHandler m_handler;
      /// The ASTImporterDelegate we are supposed to attach the handler to.
      ASTImporterDelegate &m_delegate;
      /// True iff we attached the handler to the ASTImporterDelegate.
      bool m_valid = false;

    public:
      CxxModuleScope(ASTImporterDelegate &delegate, clang::ASTContext *dst_ctx)
          : m_delegate(delegate) {
        // If the delegate doesn't have a CxxModuleHandler yet, create one
        // and attach it.
        if (!delegate.m_std_handler) {
          m_handler = CxxModuleHandler(delegate, dst_ctx);
          m_valid = true;
          delegate.m_std_handler = &m_handler;
        }
      }
      ~CxxModuleScope() {
        if (m_valid) {
          // Make sure no one messed with the handler we placed.
          assert(m_delegate.m_std_handler == &m_handler);
          m_delegate.m_std_handler = nullptr;
        }
      }
    };

    void ImportDefinitionTo(clang::Decl *to, clang::Decl *from);

    void Imported(clang::Decl *from, clang::Decl *to) override;

    clang::Decl *GetOriginalDecl(clang::Decl *To) override;

    void SetImportListener(NewDeclListener *listener) {
      assert(m_new_decl_listener == nullptr && "Already attached a listener?");
      m_new_decl_listener = listener;
    }
    void RemoveImportListener() { m_new_decl_listener = nullptr; }

  protected:
    llvm::Expected<clang::Decl *> ImportImpl(clang::Decl *From) override;

  private:
    /// Decls we should ignore when mapping decls back to their original
    /// ASTContext. Used by the CxxModuleHandler to mark declarations that
    /// were created from the 'std' C++ module to prevent that the Importer
    /// tries to sync them with the broken equivalent in the debug info AST.
    llvm::SmallPtrSet<clang::Decl *, 16> m_decls_to_ignore;
    ClangASTImporter &m_master;
    clang::ASTContext *m_source_ctx;
    CxxModuleHandler *m_std_handler = nullptr;
    /// The currently attached listener.
    NewDeclListener *m_new_decl_listener = nullptr;
  };

  typedef std::shared_ptr<ASTImporterDelegate> ImporterDelegateSP;
  typedef llvm::DenseMap<clang::ASTContext *, ImporterDelegateSP> DelegateMap;
  typedef llvm::DenseMap<const clang::NamespaceDecl *, NamespaceMapSP>
      NamespaceMetaMap;

  class ASTContextMetadata {
    typedef llvm::DenseMap<const clang::Decl *, DeclOrigin> OriginMap;

  public:
    ASTContextMetadata(clang::ASTContext *dst_ctx) : m_dst_ctx(dst_ctx) {}

    clang::ASTContext *m_dst_ctx;
    DelegateMap m_delegates;

    NamespaceMetaMap m_namespace_maps;
    MapCompleter *m_map_completer = nullptr;

    /// Sets the DeclOrigin for the given Decl and overwrites any existing
    /// DeclOrigin.
    void setOrigin(const clang::Decl *decl, DeclOrigin origin) {
      m_origins[decl] = origin;
    }

    /// Removes any tracked DeclOrigin for the given decl.
    void removeOrigin(const clang::Decl *decl) { m_origins.erase(decl); }

    /// Remove all DeclOrigin entries that point to the given ASTContext.
    /// Useful when an ASTContext is about to be deleted and all the dangling
    /// pointers to it need to be removed.
    void removeOriginsWithContext(clang::ASTContext *ctx) {
      for (OriginMap::iterator iter = m_origins.begin();
           iter != m_origins.end();) {
        if (iter->second.ctx == ctx)
          m_origins.erase(iter++);
        else
          ++iter;
      }
    }

    /// Returns the DeclOrigin for the given Decl or an invalid DeclOrigin
    /// instance if there no known DeclOrigin for the given Decl.
    DeclOrigin getOrigin(const clang::Decl *decl) const {
      auto iter = m_origins.find(decl);
      if (iter == m_origins.end())
        return DeclOrigin();
      return iter->second;
    }

    /// Returns true there is a known DeclOrigin for the given Decl.
    bool hasOrigin(const clang::Decl *decl) const {
      return getOrigin(decl).Valid();
    }

  private:
    /// Maps declarations to the ASTContext/Decl from which they were imported
    /// from. If a declaration is from an ASTContext which has been deleted
    /// since the declaration was imported or the declaration wasn't created by
    /// the ASTImporter, then it doesn't have a DeclOrigin and will not be
    /// tracked here.
    OriginMap m_origins;
  };

  typedef std::shared_ptr<ASTContextMetadata> ASTContextMetadataSP;
  typedef llvm::DenseMap<const clang::ASTContext *, ASTContextMetadataSP>
      ContextMetadataMap;

  ContextMetadataMap m_metadata_map;

  ASTContextMetadataSP GetContextMetadata(clang::ASTContext *dst_ctx) {
    ContextMetadataMap::iterator context_md_iter = m_metadata_map.find(dst_ctx);

    if (context_md_iter == m_metadata_map.end()) {
      ASTContextMetadataSP context_md =
          ASTContextMetadataSP(new ASTContextMetadata(dst_ctx));
      m_metadata_map[dst_ctx] = context_md;
      return context_md;
    }
    return context_md_iter->second;
  }

  ASTContextMetadataSP MaybeGetContextMetadata(clang::ASTContext *dst_ctx) {
    ContextMetadataMap::iterator context_md_iter = m_metadata_map.find(dst_ctx);

    if (context_md_iter != m_metadata_map.end())
      return context_md_iter->second;
    return ASTContextMetadataSP();
  }

  ImporterDelegateSP GetDelegate(clang::ASTContext *dst_ctx,
                                 clang::ASTContext *src_ctx) {
    ASTContextMetadataSP context_md = GetContextMetadata(dst_ctx);

    DelegateMap &delegates = context_md->m_delegates;
    DelegateMap::iterator delegate_iter = delegates.find(src_ctx);

    if (delegate_iter == delegates.end()) {
      ImporterDelegateSP delegate =
          ImporterDelegateSP(new ASTImporterDelegate(*this, dst_ctx, src_ctx));
      delegates[src_ctx] = delegate;
      return delegate;
    }
    return delegate_iter->second;
  }

  DeclOrigin GetDeclOrigin(const clang::Decl *decl);

  clang::FileManager m_file_manager;
  typedef llvm::DenseMap<const clang::RecordDecl *, LayoutInfo>
      RecordDeclToLayoutMap;

  RecordDeclToLayoutMap m_record_decl_to_layout_map;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_EXPRESSIONPARSER_CLANG_CLANGASTIMPORTER_H
