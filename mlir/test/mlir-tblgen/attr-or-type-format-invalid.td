// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include -asmformat-error-is-fatal=false %s 2>&1 | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";
}

class InvalidType<string name, string asm> : TypeDef<Test_Dialect, name> {
  let mnemonic = asm;
}

/// Test format is missing a parameter capture.
def InvalidTypeA : InvalidType<"InvalidTypeA", "invalid_a"> {
  let parameters = (ins "int":$v0, "int":$v1);
  // CHECK: format is missing reference to parameter: v1
  let assemblyFormat = "`<` $v0 `>`";
}

/// Test format has duplicate parameter captures.
def InvalidTypeB : InvalidType<"InvalidTypeB", "invalid_b"> {
  let parameters = (ins "int":$v0, "int":$v1);
  // CHECK: duplicate parameter 'v0'
  let assemblyFormat = "`<` $v0 `,` $v1 `,` $v0 `>`";
}

/// Test format has invalid syntax.
def InvalidTypeC : InvalidType<"InvalidTypeC", "invalid_c"> {
  let parameters = (ins "int":$v0, "int":$v1);
  // CHECK: expected literal, directive, or variable
  let assemblyFormat = "`<` $v0, $v1 `>`";
}

/// Test struct directive has invalid syntax.
def InvalidTypeD : InvalidType<"InvalidTypeD", "invalid_d"> {
  let parameters = (ins "int":$v0);
  // CHECK: literals may only be used in the top-level section of the format
  // CHECK: expected a variable in `struct` argument list
  let assemblyFormat = "`<` struct($v0, `,`) `>`";
}

/// Test struct directive cannot capture zero parameters.
def InvalidTypeE : InvalidType<"InvalidTypeE", "invalid_e"> {
  let parameters = (ins "int":$v0);
  // CHECK: `struct` argument list expected a variable or directive
  let assemblyFormat = "`<` struct() $v0 `>`";
}

/// Test capture parameter that does not exist.
def InvalidTypeF : InvalidType<"InvalidTypeF", "invalid_f"> {
  let parameters = (ins "int":$v0);
  // CHECK: InvalidTypeF has no parameter named 'v1'
  let assemblyFormat = "`<` $v0 $v1 `>`";
}

/// Test duplicate capture of parameter in capture-all struct.
def InvalidTypeG : InvalidType<"InvalidTypeG", "invalid_g"> {
  let parameters = (ins "int":$v0, "int":$v1, "int":$v2);
  // CHECK: duplicate parameter 'v0'
  let assemblyFormat = "`<` struct(params) $v0 `>`";
}

/// Test capture-all struct duplicate capture.
def InvalidTypeH : InvalidType<"InvalidTypeH", "invalid_h"> {
  let parameters = (ins "int":$v0, "int":$v1, "int":$v2);
  // CHECK: `params` captures duplicate parameter: v0
  let assemblyFormat = "`<` $v0 struct(params) `>`";
}

/// Test capture of parameter after `params` directive.
def InvalidTypeI : InvalidType<"InvalidTypeI", "invalid_i"> {
  let parameters = (ins "int":$v0);
  // CHECK: duplicate parameter 'v0'
  let assemblyFormat = "`<` params $v0 `>`";
}
