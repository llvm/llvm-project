//===-- GNUstepObjCDeclVendor.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCDECLVENDOR_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCDECLVENDOR_H

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "lldb/Symbol/DeclVendor.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

#include "clang/AST/DeclObjC.h"

namespace lldb_private {

class GNUstepObjCExternalASTSource;

/// Splits a runtime method type encoding into its component types.
///
/// The encoding interleaves types with the byte offsets of the arguments in
/// the (long obsolete) argument frame - "v16@0:8" is a void return, then
/// `self` at 0 and `_cmd` at 8. Only the types are wanted, so the digits are
/// dropped; digits that belong to a type rather than to an offset are kept,
/// which is what the depth and quote tracking are for.
///
/// The resulting order is fixed: [0] return, [1] self, [2] _cmd, [3...]
/// the declared parameters.
class MethodTypeSplitter {
public:
  explicit MethodTypeSplitter(llvm::StringRef types) { Parse(types); }

  bool IsValid() const { return m_valid && m_types.size() >= 3; }
  llvm::StringRef GetReturnType() const { return m_types[0]; }
  size_t GetNumArguments() const { return m_types.size() - 3; }
  llvm::StringRef GetArgumentType(size_t idx) const { return m_types[idx + 3]; }

private:
  void Parse(llvm::StringRef types);

  std::vector<std::string> m_types;
  bool m_valid = false;
};

/// Builds Objective-C interface declarations out of libobjc2's runtime
/// metadata, so that a class the debug info does not describe is still
/// usable: `frame variable` can show its ivars, `expr` can name its type,
/// and `type lookup` can print it.
///
/// Declarations are created lazily. FindDecls returns a forward declaration;
/// the ivars are filled in only when clang asks for the definition, through
/// an ExternalASTSource. That laziness is what keeps a lookup for one class
/// from dragging in its whole superclass chain and every type they mention.
class GNUstepObjCDeclVendor : public DeclVendor {
public:
  explicit GNUstepObjCDeclVendor(ObjCLanguageRuntime &runtime);

  ~GNUstepObjCDeclVendor() override = default;

  static bool classof(const DeclVendor *vendor) {
    return vendor->GetKind() == eGNUstepObjCDeclVendor;
  }

  uint32_t FindDecls(ConstString name, bool append, uint32_t max_matches,
                     std::vector<CompilerDecl> &decls) override;

  /// Fills in \p interface_decl's superclass and ivars. Called by the AST
  /// source when clang first needs the definition, and directly for a
  /// superclass, which has to be complete before it can be attached.
  ///
  /// Returns false if the runtime could not describe the class, in which
  /// case the interface is left as a definition with no members.
  bool FinishDecl(clang::ObjCInterfaceDecl *interface_decl);

private:
  /// Builds an ObjCMethodDecl from a runtime method name and type
  /// encoding, or nullptr if the encoding cannot be realized.
  clang::ObjCMethodDecl *
  BuildMethodDecl(clang::ObjCInterfaceDecl *interface_decl,
                  llvm::StringRef name, llvm::StringRef types,
                  bool is_instance_method);

  /// Returns a forward declaration for \p isa, creating it if needed.
  clang::ObjCInterfaceDecl *GetDeclForISA(ObjCLanguageRuntime::ObjCISA isa);

  ObjCLanguageRuntime &m_runtime;

  /// The vendor owns its own AST. Decls are copied into the expression
  /// parser's AST by ClangASTImporter when they are used, so they must not
  /// be created in a context that could outlive or conflict with it.
  std::shared_ptr<TypeSystemClang> m_ast_ctx_sp;

  /// Not owned; the AST context holds the reference that keeps it alive.
  GNUstepObjCExternalASTSource *m_external_source = nullptr;

  llvm::DenseMap<ObjCLanguageRuntime::ObjCISA, clang::ObjCInterfaceDecl *>
      m_isa_to_interface;

  // The AST source completes decls on demand and needs the vendor's AST.
  friend class GNUstepObjCExternalASTSource;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCDECLVENDOR_H
