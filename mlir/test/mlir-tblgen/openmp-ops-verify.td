// Tablegen tests for the verification of clause-based OpenMP dialect operation
// definitions.

// Run tablegen to generate OmpCommon.td in temp directory first.
// RUN: mkdir -p %t/mlir/Dialect/OpenMP
// RUN: mlir-tblgen --gen-directive-decl --directives-dialect=OpenMP \
// RUN:   %S/../../../llvm/include/llvm/Frontend/OpenMP/OMP.td \
// RUN:   -I %S/../../../llvm/include > %t/mlir/Dialect/OpenMP/OmpCommon.td

// RUN: not mlir-tblgen -verify-openmp-ops -I %S/../../include -I %t %s 2>&1 | FileCheck %s

include "mlir/Dialect/OpenMP/OpenMPOpBase.td"


def OpenMP_SimpleClause : OpenMP_Clause<
    /*traits=*/false, /*arguments=*/false, /*assemblyFormat=*/false,
    /*description=*/false, /*extraClassDeclaration=*/false> {
  let arguments = (ins I32:$val1);
  let reqAssemblyFormat = "`val1` `(` $val1 `)`";
  let description = "Simple clause description.";
  let extraClassDeclaration = "void simpleClauseExtraClassDecl();";
}


// -----------------------------------------------------------------------------
// Verify errors / warnings for overriding each field.
// -----------------------------------------------------------------------------

def 1OverrideArgsOp : OpenMP_Op<"op", clauses=[OpenMP_SimpleClause]> {
  let description = "Description of operation." # clausesDescription;
  dag arguments = (ins I32:$myval);
}
// CHECK: warning: 'Simple' clause-defined argument 'I32:$val1' not present in operation.
// CHECK-SAME: Consider `dag arguments = !con(clausesArgs, ...)` or explicitly skipping this field.
// CHECK-NEXT: def 1OverrideArgsOp

def 2OverrideAssemblyFormatOp : OpenMP_Op<"op", clauses=[OpenMP_SimpleClause]> {
  let description = "Description of operation." # clausesDescription;
  string assemblyFormat = "`alt_repr` `(` $val1 `)`";
}
// CHECK: warning: 'Simple' clause-defined `reqAssemblyFormat` not present in operation.
// CHECK-SAME: Consider concatenating `clauses[{Req,Opt}]AssemblyFormat` or explicitly skipping this field.
// CHECK-NEXT: def 2OverrideAssemblyFormatOp

def 3OverrideDescriptionOp : OpenMP_Op<"op", clauses=[OpenMP_SimpleClause]> {
  let description = "Description of operation.";
}
// CHECK: error: 'Simple' clause-defined `description` not present in operation.
// CHECK-SAME: Consider concatenating `clausesDescription` or explicitly skipping this field.
// CHECK-NEXT: def 3OverrideDescriptionOp

def 4OverrideExtraClassDeclarationOp : OpenMP_Op<"op", clauses=[OpenMP_SimpleClause]> {
  let description = "Description of operation." # clausesDescription;
  string extraClassDeclaration = "";
}
// CHECK: warning: 'Simple' clause-defined `extraClassDeclaration` not present in operation.
// CHECK-SAME: Consider concatenating `clausesExtraClassDeclaration` or explicitly skipping this field.
// CHECK-NEXT: def 4OverrideExtraClassDeclarationOp


// -----------------------------------------------------------------------------
// Verify that reporting is correct when OpenMP_Clause is inherited indirectly.
// -----------------------------------------------------------------------------

class OpenMP_IndirectClauseSkip<
    bit traits = false, bit arguments = false, bit assemblyFormat = false,
    bit description = false, bit extraClassDeclaration = false
  > : OpenMP_Clause<traits, arguments, assemblyFormat, description,
                    extraClassDeclaration> {
  let arguments = (ins I32:$val2);
  let reqAssemblyFormat = "`val2` `(` $val2 `)`";
  let description = "Indirectly-inherited clause description.";
  let extraClassDeclaration = "void indirectClauseExtraClassDecl();";
}

def IndirectClause : OpenMP_IndirectClauseSkip<>;

def 5IndirectClauseOp : OpenMP_Op<"op", clauses=[IndirectClause]> {
  let description = "Description of operation." # clausesDescription;
  dag arguments = (ins I32:$myval);
}
// CHECK: warning: 'Indirect' clause-defined argument 'I32:$val2' not present in operation.
// CHECK-NEXT: def 5IndirectClauseOp


// -----------------------------------------------------------------------------
// Verify that multiple clauses are taken into account.
// -----------------------------------------------------------------------------

def 6MultiClauseOp : OpenMP_Op<"op", clauses=[OpenMP_SimpleClause, IndirectClause]> {
  let description = "Description of operation." # clausesDescription;
  let arguments = (ins I32:$val1);
  let assemblyFormat = "`val2` `(` $val2 `)`";
}
// CHECK: warning: 'Simple' clause-defined `reqAssemblyFormat` not present in operation.
// CHECK-NEXT: def 6MultiClauseOp
// CHECK: warning: 'Indirect' clause-defined argument 'I32:$val2' not present in operation.
// CHECK-NEXT: def 6MultiClauseOp


// -----------------------------------------------------------------------------
// Verify that reporting is correct when clause definitions have other
// superclasses in addition to OpenMP_Clause.
// -----------------------------------------------------------------------------

class Placeholder {}
def MultiSuperClassClause : Placeholder, OpenMP_IndirectClauseSkip<>;

def 7MultiSuperClassClauseOp : OpenMP_Op<"op", clauses=[IndirectClause]> {
  let description = "Description of operation." # clausesDescription;
  dag arguments = (ins I32:$myval);
}
// CHECK: warning: 'Indirect' clause-defined argument 'I32:$val2' not present in operation.
// CHECK-NEXT: def 7MultiSuperClassClauseOp


// -----------------------------------------------------------------------------
// Verify that no errors are produced if the field being overriden is also
// skipped for the clause.
// -----------------------------------------------------------------------------

def SkipArgsOp : OpenMP_Op<"op",
    clauses=[OpenMP_IndirectClauseSkip<arguments=true>]> {
  let description = "Description of operation." # clausesDescription;
  dag arguments = (ins I32:$myval);
}
def SkipAssemblyFormatOp : OpenMP_Op<"op",
    clauses=[OpenMP_IndirectClauseSkip<assemblyFormat=true>]> {
  let description = "Description of operation." # clausesDescription;
  string assemblyFormat = "`alt_repr` `(` $val1 `)`";
}
def SkipDescriptionOp : OpenMP_Op<"op",
    clauses=[OpenMP_IndirectClauseSkip<description=true>]> {
  let description = "Description of operation.";
}
def SkipExtraClassDeclarationOp : OpenMP_Op<"op",
    clauses=[OpenMP_IndirectClauseSkip<extraClassDeclaration=true>]> {
  let description = "Description of operation." # clausesDescription;
  string extraClassDeclaration = "";
}
// CHECK-NOT: error:
// CHECK-NOT: warning:
