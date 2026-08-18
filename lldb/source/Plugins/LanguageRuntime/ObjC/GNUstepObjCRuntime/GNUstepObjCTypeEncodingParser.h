//===-- GNUstepObjCTypeEncodingParser.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCTYPEENCODINGPARSER_H
#define LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCTYPEENCODINGPARSER_H

#include "Plugins/LanguageRuntime/ObjC/ObjCLanguageRuntime.h"
#include "lldb/lldb-private.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/Triple.h"

#include "clang/AST/ASTContext.h"

#include <string>

namespace lldb_private {

/// Turns an Objective-C type encoding into a CompilerType, for the encodings
/// libobjc2 stores alongside its ivars, methods and properties.
///
/// The grammar is the standard one (documented normatively in libobjc2's
/// ABIDoc/abi.tex), so this is close to AppleObjCTypeEncodingParser. It
/// differs in four ways that matter for libobjc2 input:
///
///   - Leading method qualifiers ("nNoORVA", libobjc2's encoding2.c) are
///     skipped. `A` for _Atomic is a libobjc2 addition. `r` is deliberately
///     not in that set: it is _C_CONST, and is handled as a real qualifier.
///   - `@"Name"` is always a class name. Apple has to disambiguate it from a
///     field named "Name", because its runtime emits quoted field names
///     inside struct encodings; clang's GNUstep output never does, so the
///     disambiguation would actively mis-parse `{Foo=@"NSString"i}`.
///   - A class name the runtime cannot resolve yields `id` rather than
///     tripping an assertion. libobjc2 encodings routinely name classes that
///     are not realized.
///   - Record types are cached, so a struct appearing in many encodings does
///     not mint a fresh (and separately laid out) decl each time.
///
/// The parser holds no target state, so it can be constructed from a triple
/// alone; a runtime is only needed to resolve `@"Name"` for expressions.
class GNUstepObjCTypeEncodingParser
    : public ObjCLanguageRuntime::EncodingToType {
public:
  /// \param triple selects the data model, which decides how wide `long`
  ///        and friends are.
  /// \param runtime resolves `@"Name"` when realizing types for expressions;
  ///        without one such a name degrades to `id`.
  explicit GNUstepObjCTypeEncodingParser(
      const llvm::Triple &triple, ObjCLanguageRuntime *runtime = nullptr);

  ~GNUstepObjCTypeEncodingParser() override = default;

  CompilerType RealizeType(TypeSystemClang &ast_ctx, const char *name,
                           bool for_expression) override;

private:
  struct StructElement {
    std::string name;
    clang::QualType type;
    uint32_t bitfield = 0;
  };

  clang::QualType BuildType(TypeSystemClang &ast_ctx, llvm::StringRef &type,
                            bool for_expression,
                            uint32_t *bitfield_bit_size = nullptr);

  clang::QualType BuildAggregate(TypeSystemClang &ast_ctx,
                                 llvm::StringRef &type, bool for_expression,
                                 char opener, char closer, uint32_t kind);

  clang::QualType BuildArray(TypeSystemClang &ast_ctx, llvm::StringRef &type,
                             bool for_expression);

  clang::QualType BuildObjCObjectPointerType(TypeSystemClang &ast_ctx,
                                             llvm::StringRef &type,
                                             bool for_expression);

  StructElement ReadStructElement(TypeSystemClang &ast_ctx,
                                  llvm::StringRef &type, bool for_expression);

  ObjCLanguageRuntime *m_runtime;

  /// Recursion depth of the current BuildType call chain.
  unsigned m_depth = 0;

  /// Records already built, keyed by the AST they belong to and then by the
  /// exact encoding text they were built from. Both halves are needed:
  /// RealizeType is called with different ASTContexts, and a QualType is only
  /// valid in the one that created it.
  llvm::DenseMap<TypeSystemClang *, llvm::StringMap<clang::QualType>>
      m_record_cache;
};

} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_LANGUAGERUNTIME_OBJC_GNUSTEPOBJCRUNTIME_GNUSTEPOBJCTYPEENCODINGPARSER_H
