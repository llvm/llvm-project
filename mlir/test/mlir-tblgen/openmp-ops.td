// Tablegen tests for the automatic population of OpenMP dialect operation
// fields from clauses. Arguments and extra class declarations are checked from
// the output of tablegen declarations. Summary, description, assembly format
// and traits are checked from the output of tablegen docs.

// Run tablegen to generate OmpCommon.td in temp directory first.
// RUN: mkdir -p %t/mlir/Dialect/OpenMP
// RUN: mlir-tblgen --gen-directive-decl --directives-dialect=OpenMP \
// RUN:   %S/../../../llvm/include/llvm/Frontend/OpenMP/OMP.td \
// RUN:   -I %S/../../../llvm/include > %t/mlir/Dialect/OpenMP/OmpCommon.td

// RUN: mlir-tblgen -gen-op-decls -I %S/../../include -I %t %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-op-doc -I %S/../../include -I %t %s | FileCheck %s --check-prefix=DOC

include "mlir/Dialect/OpenMP/OpenMPOpBase.td"

// Dummy traits.

def TraitOne : NativeOpTrait<"TraitOne">;
def TraitTwo : NativeOpTrait<"TraitTwo">;
def TraitThree : NativeOpTrait<"TraitThree">;

// Test clauses.

class OptClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause</*isRequired=*/false, traits, arguments, assemblyFormat,
                    description, extraClassDeclaration> {
  let traits = [
    TraitOne
  ];

  let arguments = (ins
    Optional<AnyInteger>:$opt_simple_val
  );

  let assemblyFormat = [{
    `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
  }];

  let description = [{
    Optional clause description.
  }];

  let extraClassDeclaration = [{
    void optClauseExtraClassDecl();
  }];
}

def OptClause : OptClauseSkip<>;

def OptClauseTwo : OpenMP_Clause<
    /*isRequired=*/false, /*skipTraits=*/false, /*skipArguments=*/false,
    /*skipAssemblyFormat=*/false, /*skipDescription=*/false,
    /*skipExtraClassDeclaration=*/false> {
  let traits = [
    TraitOne, TraitTwo
  ];

  let arguments = (ins
    Optional<AnyInteger>:$opt_two_val
  );

  let assemblyFormat = [{
    `opt_two` `(` $opt_two_val `:` type($opt_two_val) `)`
  }];

  let description = [{
    Optional clause two description.
  }];

  let extraClassDeclaration = [{
    void optClauseTwoExtraClassDecl();
  }];
}

class ReqClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause</*isRequired=*/true, traits, arguments, assemblyFormat,
                    description, extraClassDeclaration> {
  let traits = [
    TraitTwo
  ];

  let arguments = (ins
    AnyInteger:$req_simple_val
  );

  let assemblyFormat = [{
    `reqsimple` `(` $req_simple_val `:` type($req_simple_val) `)`
  }];

  let description = [{
    Required clause description.
  }];

  let extraClassDeclaration = [{
    void reqClauseExtraClassDecl();
  }];
}

def ReqClause : ReqClauseSkip<>;

def ReqClauseTwo : OpenMP_Clause<
    /*isRequired=*/true, /*skipTraits=*/false, /*skipArguments=*/false,
    /*skipAssemblyFormat=*/false, /*skipDescription=*/false,
    /*skipExtraClassDeclaration=*/false> {
  let traits = [
    TraitTwo, TraitThree
  ];

  let arguments = (ins
    AnyInteger:$req_two_val
  );

  let assemblyFormat = [{
    `req_two` `(` $req_two_val `:` type($req_two_val) `)`
  }];

  let description = [{
    Required clause two description.
  }];

  let extraClassDeclaration = [{
    void reqClauseTwoExtraClassDecl();
  }];
}

// Clause-based operation definitions.

def OpOptClause : OpenMP_Op<"op_with_opt",
    traits=[TraitThree], clauses=[OptClause]> {
  let summary = "operation with optional clause";
  let description = [{
    Description of operation with optional clause.
  }] # clausesDescription;
}

// DECL-LABEL: class OpOptClause : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpOptClause
// DOC: _Operation with optional clause_
// DOC: operation ::= `omp.op_with_opt`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with optional clause.
// DOC: Optional clause description.
// DOC: Traits: `TraitOne`, `TraitThree`

def OpReqClause : OpenMP_Op<"op_with_req",
    traits=[TraitThree], clauses=[ReqClause]> {
  let summary = "operation with required clause";
  let description = [{
    Description of operation with required clause.
  }] # clausesDescription;
}

// DECL-LABEL: class OpReqClause : public ::mlir::Op<
// DECL: getReqSimpleVal() {
// DECL: void reqClauseExtraClassDecl();

// DOC-LABEL: omp::OpReqClause
// DOC: _Operation with required clause_
// DOC: operation ::= `omp.op_with_req`
// DOC-NOT: oilist(
// DOC: `reqsimple` `(` $req_simple_val `:` type($req_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with required clause.
// DOC: Required clause description.
// DOC: Traits: `TraitThree`, `TraitTwo`

def OpReqOptClause : OpenMP_Op<"op_with_req_and_opt",
    traits=[TraitThree], clauses=[ReqClause, OptClause]> {
  let summary = "operation with required and optional clauses";
  let description = [{
    Description of operation with required and optional clauses.
  }] # clausesDescription;
}

// DECL-LABEL: class OpReqOptClause : public ::mlir::Op<
// DECL: getReqSimpleVal() {
// DECL: getOptSimpleVal() {
// DECL: void reqClauseExtraClassDecl();
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpReqOptClause
// DOC: _Operation with required and optional clauses_
// DOC: operation ::= `omp.op_with_req_and_opt`
// DOC: `reqsimple` `(` $req_simple_val `:` type($req_simple_val) `)`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with required and optional clauses.
// DOC: Required clause description.
// DOC: Optional clause description.
// DOC: Traits: `TraitOne`, `TraitThree`, `TraitTwo`

def OpSingleRegion : OpenMP_Op<"op_single_region",
    clauses=[OptClause], singleRegion=true> {
  let summary = "operation with a single region";
  let description = [{
    Description of operation with a single region.
  }] # clausesDescription;
}

// DECL-LABEL: class OpSingleRegion : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL: getRegion() {
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpSingleRegion
// DOC: _Operation with a single region_
// DOC: operation ::= `omp.op_single_region`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC: ) $region attr-dict
// DOC: Description of operation with a single region.
// DOC: Optional clause description.
// DOC: Traits: `TraitOne`

def OpSkipArguments : OpenMP_Op<"op_skip_arguments",
    traits=[TraitThree], clauses=[OptClauseSkip<arguments=true>]> {
  let summary = "operation with clause - arguments skipped";
  let description = [{
    Description of operation with clause - arguments skipped.
  }] # clausesDescription;
  let arguments = !con(clausesArgs, (ins Optional<AnyInteger>:$opt_simple_val));
}

// DECL-LABEL: class OpSkipArguments : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpSkipArguments
// DOC: _Operation with clause - arguments skipped_
// DOC: operation ::= `omp.op_skip_arguments`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with clause - arguments skipped.
// DOC: Optional clause description.
// DOC: Traits: `TraitOne`, `TraitThree`

def OpSkipAssemblyFormat : OpenMP_Op<"op_skip_assembly_format",
    traits=[TraitThree], clauses=[ReqClauseSkip<assemblyFormat=true>,
                                  OptClause]> {
  let summary = "operation with clauses - assemblyFormat skipped";
  let description = [{
    Description of operation with clauses - assemblyFormat skipped.
  }] # clausesDescription;
  let assemblyFormat = [{
    `alt_assembly_format` `(` $req_simple_val `:` type($req_simple_val) `)`
  }] # clausesAssemblyFormat # "attr-dict";
}

// DECL-LABEL: class OpSkipAssemblyFormat : public ::mlir::Op<
// DECL: getReqSimpleVal() {
// DECL: getOptSimpleVal() {
// DECL: void reqClauseExtraClassDecl();
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpSkipAssemblyFormat
// DOC: _Operation with clauses - assemblyFormat skipped_
// DOC: operation ::= `omp.op_skip_assembly_format`
// DOC: `alt_assembly_format` `(` $req_simple_val `:` type($req_simple_val) `)`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with clauses - assemblyFormat skipped.
// DOC: Required clause description.
// DOC: Optional clause description.
// DOC: Traits: `TraitOne`, `TraitThree`

def OpSkipDescription : OpenMP_Op<"op_skip_description",
    traits=[TraitThree], clauses=[OptClauseSkip<description=true>]> {
  let summary = "operation with clause - description skipped";
  let description = [{
    Description of operation with clause - description skipped.
  }] # clausesDescription;
}

// DECL-LABEL: class OpSkipDescription : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpSkipDescription
// DOC: _Operation with clause - description skipped_
// DOC: operation ::= `omp.op_skip_description`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with clause - description skipped.
// DOC-NOT: Optional clause description.
// DOC: Traits: `TraitOne`, `TraitThree`

def OpSkipExtraClassDeclaration : OpenMP_Op<"op_skip_extra_class_declaration",
    traits=[TraitThree], clauses=[OptClauseSkip<extraClassDeclaration=true>]> {
  let summary = "operation with clause - extraClassDeclaration skipped";
  let description = [{
    Description of operation with clause - extraClassDeclaration skipped.
  }] # clausesDescription;
}

// DECL-LABEL: class OpSkipExtraClassDeclaration : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL-NOT: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpSkipExtraClassDeclaration
// DOC: _Operation with clause - extraClassDeclaration skipped_
// DOC: operation ::= `omp.op_skip_extra_class_declaration`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with clause - extraClassDeclaration skipped.
// DOC: Optional clause description.
// DOC: Traits: `TraitOne`, `TraitThree`

def OpSkipTraits : OpenMP_Op<"op_skip_traits",
    traits=[TraitThree], clauses=[OptClauseSkip<traits=true>]> {
  let summary = "operation with clause - traits skipped";
  let description = [{
    Description of operation with clause - traits skipped.
  }] # clausesDescription;
}

// DECL-LABEL: class OpSkipTraits : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL: void optClauseExtraClassDecl();

// DOC-LABEL: omp::OpSkipTraits
// DOC: _Operation with clause - traits skipped_
// DOC: operation ::= `omp.op_skip_traits`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with clause - traits skipped.
// DOC: Optional clause description.
// DOC: Traits: `TraitThree`

def OpTwoOptionalClauses : OpenMP_Op<"op_two_opt_clauses",
    traits=[AttrSizedOperandSegments], clauses=[OptClause, OptClauseTwo]> {
  let summary = "operation with two optional clauses";
  let description = [{
    Description of operation with two optional clauses.
  }] # clausesDescription;
}

// DECL-LABEL: class OpTwoOptionalClauses : public ::mlir::Op<
// DECL: getOptSimpleVal() {
// DECL: getOptTwoVal() {
// DECL: void optClauseExtraClassDecl();
// DECL: void optClauseTwoExtraClassDecl();

// DOC-LABEL: omp::OpTwoOptionalClauses
// DOC: _Operation with two optional clauses_
// DOC: operation ::= `omp.op_two_opt_clauses`
// DOC: oilist(
// DOC: `optsimple` `(` $opt_simple_val `:` type($opt_simple_val) `)`
// DOC: |
// DOC: `opt_two` `(` $opt_two_val `:` type($opt_two_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with two optional clauses.
// DOC: Optional clause description.
// DOC: Optional clause two description.
// DOC: Traits: `AttrSizedOperandSegments`, `TraitOne`, `TraitTwo`

def OpTwoRequiredClauses : OpenMP_Op<"op_two_req_clauses",
    clauses=[ReqClause, ReqClauseTwo]> {
  let summary = "operation with two required clauses";
  let description = [{
    Description of operation with two required clauses.
  }] # clausesDescription;
}

// DECL-LABEL: class OpTwoRequiredClauses : public ::mlir::Op<
// DECL: getReqSimpleVal() {
// DECL: getReqTwoVal() {
// DECL: void reqClauseExtraClassDecl();
// DECL: void reqClauseTwoExtraClassDecl();

// DOC-LABEL: omp::OpTwoRequiredClauses
// DOC: _Operation with two required clauses_
// DOC: operation ::= `omp.op_two_req_clauses`
// DOC-NOT: oilist(
// DOC: `reqsimple` `(` $req_simple_val `:` type($req_simple_val) `)`
// DOC-NOT: |
// DOC: `req_two` `(` $req_two_val `:` type($req_two_val) `)`
// DOC-NOT: $region
// DOC: attr-dict
// DOC: Description of operation with two required clauses.
// DOC: Required clause description.
// DOC: Required clause two description.
// DOC: Traits: `TraitThree`, `TraitTwo`

def OpZeroClauses : OpenMP_Op<"op_zero_clauses"> {
  let summary = "operation with no clauses";
  let description = [{
    Description of operation with no clauses.
  }] # clausesDescription;
}

// DECL-LABEL: class OpZeroClauses : public ::mlir::Op<
// DECL: static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);

// DOC-LABEL: omp::OpZeroClauses
// DOC: _Operation with no clauses_
// DOC-NOT: operation ::=
// DOC: Description of operation with no clauses.
// DOC-NOT: Traits:
