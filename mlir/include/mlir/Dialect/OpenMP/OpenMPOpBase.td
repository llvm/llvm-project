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

//===---------------------------------------------------------------------===//
// OpenMP Canonical Loop Info Type
//===---------------------------------------------------------------------===//

def CanonicalLoopInfoType : OpenMP_Type<"CanonicalLoopInfo", "cli"> {
  let summary = "Type for representing a reference to a canonical loop";
  let description = [{
    A variable of type CanonicalLoopInfo refers to an OpenMP-compatible
    canonical loop in the same function. Values of this type are not
    available at runtime and therefore cannot be used by the program itself,
    i.e. an opaque type. It is similar to the transform dialect's
    `!transform.interface` type, but instead of implementing an interface
    for each transformation, the OpenMP dialect itself defines possible
    operations on this type.

    A value of type CanonicalLoopInfoType (in the following: CLI) value can be

    1. created by omp.new_cli.
    2. passed to omp.canonical_loop to associate the loop to that CLI. A CLI
       can only be associated once.
    3. passed to an omp loop transformation operation that modifies the loop
       associated with the CLI. The CLI is the "applyee" and the operation is
       the consumer. A CLI can only be consumed once.
    4. passed to an omp loop transformation operation to associate the cli with
       a result of that transformation. The CLI is the "generatee" and the
       operation is the generator.

    A CLI cannot

    1. be returned from a function.
    2. be passed to operations that are not specifically designed to take a
       CanonicalLoopInfoType, including AnyType.

    A CLI directly corresponds to an object of
    OpenMPIRBuilder's CanonicalLoopInfo struct when lowering to LLVM-IR.
  }];
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
// An `OpenMP_Op` can inhibit the inheritance of `traits`, `arguments`,
// `description` and `extraClassDeclaration` fields from any given
// `OpenMP_Clause` by setting to 1 the corresponding "skip" template argument
// bit. Additionally, the `skipAssemblyFormat` bit inhibits the inheritance of
// fields used to populate `OpenMP_Op`'s `assemblyFormat` (i.e.
// `reqAssemblyFormat` and `optAssemblyFormat`).
class OpenMP_Clause<bit skipTraits, bit skipArguments, bit skipAssemblyFormat,
                    bit skipDescription, bit skipExtraClassDeclaration> {
  bit ignoreTraits = skipTraits;
  list<Trait> traits = [];

  bit ignoreArgs = skipArguments;
  dag arguments;

  bit ignoreAsmFormat = skipAssemblyFormat;
  string reqAssemblyFormat = "";
  string optAssemblyFormat = "";

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
//   - `assemblyFormat`: It is a concatenation of the `reqAssemblyFormat`
//     followed by an `oilist()` containing the `optAssemblyFormat` of all
//     clauses. The format string is completed with $region (if `singleRegion =
//     true`) followed by `attr-dict`. This field remains uninitialized if no
//     non-empty `{req,opt}AssemblyFormat` strings are inherited from clauses.
//     The `clausesAssemblyFormat` field holds all the format string except for
//     "$region attr-dict", and the `clauses{Req,Opt}AssemblyFormat` fields
//     hold the required and optional parts of the format string separately, so
//     that an operation overriding `assemblyFormat` can still benefit from the
//     auto-generated format for its clauses.
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
  list<OpenMP_Clause> clauseList = clauses;

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
  defvar asmFormatFilteredReqClauses =
    !filter(clause, clauses, !not(!or(clause.ignoreAsmFormat,
                                     !empty(clause.reqAssemblyFormat))));

  defvar asmFormatReqClauseStrings =
    !foreach(clause, asmFormatFilteredReqClauses, clause.reqAssemblyFormat);

  string clausesReqAssemblyFormat = !interleave(asmFormatReqClauseStrings, " ");

  // Optional clauses.
  defvar asmFormatFilteredOptClauses =
    !filter(clause, clauses, !not(!or(clause.ignoreAsmFormat,
                                     !empty(clause.optAssemblyFormat))));

  defvar asmFormatOptClauseStrings =
    !foreach(clause, asmFormatFilteredOptClauses, clause.optAssemblyFormat);

  string clausesOptAssemblyFormat = !interleave(asmFormatOptClauseStrings, "|");

  string clausesAssemblyFormat =
    !if(!empty(asmFormatReqClauseStrings), "", clausesReqAssemblyFormat # " ") #
    !if(!empty(asmFormatOptClauseStrings), "",
        "oilist(" # clausesOptAssemblyFormat # ")");

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


// Base class for OpenMP loop transformations (that either consume or generate
// loops)
//
// Doesn't actually create a C++ base class (only defines default values for
// tablegen classes that derive from this). Use LoopTransformationInterface
// instead for common operations.
class OpenMPTransform_Op<string mnemonic,
                         list<Trait> traits = [],
                         list<OpenMP_Clause> clauses = []> :
      OpenMP_Op<mnemonic,
                traits = !listconcat([DeclareOpInterfaceMethods<LoopTransformationInterface>], traits),
                clauses = clauses> {
}

// Base clause for loop transformations using the standard syntax.
//
//     omp.opname ($generatees) <- ($applyees) clause(...) clause(...) ... <attr-dicr>
//     omp.opname                  ($applyees) clause(...) clause(...) ... <attr-dict>
//
// $generatees is optional and is assumed to be empty if omitted
class OpenMPTransformBase_Op<string mnemonic,
                         list<Trait> traits = [],
                         list<OpenMP_Clause> clauses = []> :
      OpenMPTransform_Op<mnemonic,
                         traits = !listconcat(traits, [AttrSizedOperandSegments]),
                         clauses = clauses> {

  let arguments = !con(
                       (ins Variadic<CanonicalLoopInfoType>:$generatees,
                            Variadic<CanonicalLoopInfoType>:$applyees
                      ), clausesArgs);

  let assemblyFormat = [{ custom<LoopTransformClis>($generatees, $applyees) }]
                         # clausesAssemblyFormat
                         # [{ attr-dict }];
}

#endif  // OPENMP_OP_BASE
