//===- OpenMPOpBase.td - OpenMP dialect shared definitions -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains shared definitions for the OpenMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_OP_BASE
#define OPENMP_OP_BASE

include "mlir/Dialect/OpenMP/OpenMPAttrDefs.td"
include "mlir/Dialect/OpenMP/OpenMPDialect.td"
include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.td"
include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// OpenMP dialect type constraints.
//===----------------------------------------------------------------------===//

class OpenMP_Type<string name, string typeMnemonic> :
      TypeDef<OpenMP_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Type which can be constraint accepting standard integers and indices.
def IntLikeType : AnyTypeOf<[AnyInteger, Index]>;

def OpenMP_PointerLikeType : TypeAlias<OpenMP_PointerLikeTypeInterface,
	"OpenMP-compatible variable type">;

def OpenMP_MapBoundsType : OpenMP_Type<"MapBounds", "map_bounds_ty"> {
  let summary = "Type for representing omp map clause bounds information";
}

//===----------------------------------------------------------------------===//
// Base classes for OpenMP dialect operations.
//===----------------------------------------------------------------------===//

// Base class for representing OpenMP clauses.
//
// Clauses are meant to be used in a mixin-style pattern to help define OpenMP
// operations in a scalable way, since often the same clause can be applied to
// multiple different operations.
//
// To keep the representation of clauses consistent across different operations,
// each clause must define a set of arguments (values and attributes) which will
// become input arguments of each OpenMP operation that accepts that clause.
//
// It is also recommended that an assembly format and description are defined
// for each clause wherever posible, to make sure they are always printed,
// parsed and described in the same way.
//
// Optionally, operation traits and extra class declarations might be attached
// to clauses, which will be forwarded to all operations that include them.
//
// Each clause must specify whether it's required or optional. This impacts how
// the `assemblyFormat` for operations including it get generated.
//
// An `OpenMP_Op` can inhibit the inheritance of `traits`, `arguments`,
// `assemblyFormat`, `description` and `extraClassDeclaration` fields from any
// given `OpenMP_Clause` by setting to 1 the corresponding "skip" template
// argument bit.
class OpenMP_Clause<bit isRequired, bit skipTraits, bit skipArguments,
                    bit skipAssemblyFormat, bit skipDescription,
                    bit skipExtraClassDeclaration> {
  bit required = isRequired;

  bit ignoreTraits = skipTraits;
  list<Trait> traits = [];

  bit ignoreArgs = skipArguments;
  dag arguments;

  bit ignoreAsmFormat = skipAssemblyFormat;
  string assemblyFormat = "";

  bit ignoreDesc = skipDescription;
  string description = "";

  bit ignoreExtraDecl = skipExtraClassDeclaration;
  string extraClassDeclaration = "";
}

// Base class for representing OpenMP operations.
//
// This is a subclass of the builtin `Op` for the OpenMP dialect. By default,
// some of its fields are initialized according to the list of OpenMP clauses
// passed as template argument:
//   - `traits`: It is a union of the traits list passed as template argument
//     and those inherited from the `traits` field of all clauses.
//   - `arguments`: They are a concatenation of clause-inherited arguments. They
//     are saved to a `clausesArgs` field to allow overriding the arguments
//     field in the definition of the operation and still being able to include
//     those inherited from clauses.
//   - `assemblyFormat`: It is a concatenation of the `assemblyFormat` of
//     all required clauses followed by an `oilist()` containing the
//     `assemblyFormat` of all optional clauses. The format string is completed
//     with $region (if `singleRegion = true`) followed by attr-dict. This field
//     remains uninitialized if no non-empty `assemblyFormat` strings are
//     inherited from clauses. The `clausesAssemblyFormat` field holds
//     all the format except for "$region attr-dict", so that an operation
//     overriding `assemblyFormat` can still benefit from the auto-generated
//     format for its clauses.
//   - `description`: This is still required to be defined by the operation.
//     However, a `clausesDescription` field is provided containing a
//     concatenation of descriptions of all clauses, to be appended to the
//     operation's `description` field.
//   - `extraClassDeclaration`: It contains a concatenation of the
//     `extraClassDeclaration` of all clauses. This string is also stored in
//     `clausesExtraClassDeclaration`, so that an operation overriding this
//     field can append the clause-inherited ones as well.
//
// The `regions` field will contain a single `AnyRegion:$region` element if the
// `singleRegion` bit template argument is set to 1. Otherwise, it will be
// empty.
class OpenMP_Op<string mnemonic, list<Trait> traits = [],
                list<OpenMP_Clause> clauses = [], bit singleRegion = false> :
    Op<OpenMP_Dialect, mnemonic,
    // The resulting operation's traits list will be the concatenation of
    // explicit operation traits and all traits attached to the clauses of the
    // operation. Repetitions are skipped.
    !listconcat(traits,
      !listremove(
        !foldl([]<Trait>,
               !foreach(clause,
                        !filter(fClause, clauses, !not(fClause.ignoreTraits)),
                        clause.traits),
               acc, traitList, !listconcat(acc, !listremove(traitList, acc))),
        traits
      )
    )> {
  // Aggregate `arguments` fields of all clauses into a single dag, to be used
  // by operations to populate their `arguments` field.
  defvar argsFilteredClauses =
    !filter(clause, clauses, !not(clause.ignoreArgs));

  dag clausesArgs =
    !foldl((ins), !foreach(clause, argsFilteredClauses, clause.arguments),
           acc, argList, !con(acc, argList));

  // Create assembly format string by concatenating format strings separately
  // for required and optional clauses. Then, required clauses format strings
  // are joined with spaces in between. Optional clauses format strings are
  // wrapped into an unsorted list of optional values and separated by "|"
  // characters.

  // Required clauses.
  defvar reqClauses = !filter(clause, clauses, clause.required);
  defvar asmFormatFilteredReqClauses =
    !filter(clause, reqClauses, !not(!or(clause.ignoreAsmFormat,
                                     !empty(clause.assemblyFormat))));

  defvar asmFormatReqClauseStrings =
    !foreach(clause, asmFormatFilteredReqClauses, clause.assemblyFormat);

  defvar asmFormatReqClauseBody = !interleave(asmFormatReqClauseStrings, " ");

  // Optional clauses.
  defvar optClauses = !filter(clause, clauses, !not(clause.required));
  defvar asmFormatFilteredOptClauses =
    !filter(clause, optClauses, !not(!or(clause.ignoreAsmFormat,
                                     !empty(clause.assemblyFormat))));

  defvar asmFormatOptClauseStrings =
    !foreach(clause, asmFormatFilteredOptClauses, clause.assemblyFormat);

  defvar asmFormatOptClauseBody = !interleave(asmFormatOptClauseStrings, "|");

  string clausesAssemblyFormat =
    !if(!empty(asmFormatReqClauseStrings), "", asmFormatReqClauseBody # " ") #
    !if(!empty(asmFormatOptClauseStrings), "",
        "oilist(" # asmFormatOptClauseBody # ")");

  // Put together descriptions of all clauses into a single string.
  defvar descFilteredClauses =
    !filter(clause, clauses, !not(clause.ignoreDesc));

  string clausesDescription =
    !interleave(!foreach(clause, descFilteredClauses, clause.description), "");
  
  // Aggregate `extraClassDeclaration` of all clauses that define it.
  defvar extraDeclFilteredClauses =
    !filter(clause, clauses, !not(clause.ignoreExtraDecl));

  string clausesExtraClassDeclaration =
    !interleave(!foreach(clause, extraDeclFilteredClauses,
                         clause.extraClassDeclaration), "\n");

  // The default arguments, assembly format and extra class declarations for
  // OpenMP operations are those defined by their args and clauses.
  let arguments = clausesArgs;
  let assemblyFormat =
    !if(!empty(clausesAssemblyFormat), ?,
        clausesAssemblyFormat # !if(singleRegion, " $region", "") #
        " attr-dict");
  let extraClassDeclaration = clausesExtraClassDeclaration;

  // By default, the op will have zero regions. Setting `singleRegion = true`
  // will result in a single region named `$region`.
  let regions = !if(singleRegion, (region AnyRegion:$region), (region));
}

#endif  // OPENMP_OP_BASE
