//===-- DialectBase.td - Base Dialect definition file ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the base set of constructs for defining Dialect classes.
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTBASE_TD
#define DIALECTBASE_TD

//===----------------------------------------------------------------------===//
// Dialect definitions
//===----------------------------------------------------------------------===//

class Dialect {
  // The name of the dialect.
  string name = ?;

  // Short summary of the dialect.
  string summary = ?;

  // The description of the dialect.
  string description = ?;

  // A list of dialects this dialect will load on construction as dependencies.
  // These are dialects that this dialect may involve in canonicalization
  // pattern or interfaces.
  list<string> dependentDialects = [];

  // The C++ namespace that ops of this dialect should be placed into.
  //
  // By default, uses the name of the dialect as the only namespace. To avoid
  // placing in any namespace, use "". To specify nested namespaces, use "::"
  // as the delimiter, e.g., given "A::B", ops will be placed in
  // `namespace A { namespace B { <ops> } }`.
  //
  // Note that this works in conjunction with dialect C++ code. Depending on how
  // the generated files are included into the dialect, you may want to specify
  // a full namespace path or a partial one.
  string cppNamespace = name;

  // An optional code block containing extra declarations to place in the
  // dialect declaration.
  code extraClassDeclaration = "";

  // If this dialect overrides the hook for materializing constants.
  bit hasConstantMaterializer = 0;

  /// If the dialect definition provides a non-default destructor.
  /// If false, a default destructor implementation will be generated.
  bit hasNonDefaultDestructor = 0;

  // If this dialect overrides the hook for verifying operation attributes.
  bit hasOperationAttrVerify = 0;

  // If this dialect overrides the hook for verifying region argument
  // attributes.
  bit hasRegionArgAttrVerify = 0;

  // If this dialect overrides the hook for verifying region result attributes.
  bit hasRegionResultAttrVerify = 0;

  // If this dialect overrides the hook for op interface fallback.
  bit hasOperationInterfaceFallback = 0;

  // If this dialect should use default generated attribute parser boilerplate.
  // When set, ODS will generate declarations for the attribute parsing and
  // printing hooks in the dialect and default implementations that dispatch to
  // each individual attribute directly.
  bit useDefaultAttributePrinterParser = 0;

  // If this dialect should use default generated type parser boilerplate:
  // When set, ODS will generate declarations for the type parsing and printing
  // hooks in the dialect and default implementations that dispatch to each
  // individual type directly.
  bit useDefaultTypePrinterParser = 0;

  // If this dialect overrides the hook for canonicalization patterns.
  bit hasCanonicalizer = 0;

  // If this dialect can be extended at runtime with new operations or types.
  bit isExtensible = 0;
}

#endif // DIALECTBASE_TD
