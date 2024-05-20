// RUN: mlir-tblgen -gen-typedef-defs -I %S/../../include -asmformat-error-is-fatal=false %s 2>&1 | FileCheck %s

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "TestDialect";
  let cppNamespace = "::test";
}

class InvalidType<string name, string asm> : TypeDef<Test_Dialect, name> {
  let mnemonic = asm;
}

// Test format is missing a parameter capture.
def InvalidTypeA : InvalidType<"InvalidTypeA", "invalid_a"> {
  let parameters = (ins "int":$v0, "int":$v1);
  // CHECK: format is missing reference to parameter: v1
  let assemblyFormat = "`<` $v0 `>`";
}

// Test format has duplicate parameter captures.
def InvalidTypeB : InvalidType<"InvalidTypeB", "invalid_b"> {
  let parameters = (ins "int":$v0, "int":$v1);
  // CHECK: duplicate parameter 'v0'
  let assemblyFormat = "`<` $v0 `,` $v1 `,` $v0 `>`";
}

// Test format has invalid syntax.
def InvalidTypeC : InvalidType<"InvalidTypeC", "invalid_c"> {
  let parameters = (ins "int":$v0, "int":$v1);
  // CHECK: expected literal, variable, directive, or optional group
  let assemblyFormat = "`<` $v0, $v1 `>`";
}

// Test struct directive has invalid syntax.
def InvalidTypeD : InvalidType<"InvalidTypeD", "invalid_d"> {
  let parameters = (ins "int":$v0);
  // CHECK: literals may only be used in the top-level section of the format
  // CHECK: expected a variable in `struct` argument list
  let assemblyFormat = "`<` struct($v0, `,`) `>`";
}

// Test struct directive cannot capture zero parameters.
def InvalidTypeE : InvalidType<"InvalidTypeE", "invalid_e"> {
  let parameters = (ins "int":$v0);
  // CHECK: `struct` argument list expected a variable or directive
  let assemblyFormat = "`<` struct() $v0 `>`";
}

// Test capture parameter that does not exist.
def InvalidTypeF : InvalidType<"InvalidTypeF", "invalid_f"> {
  let parameters = (ins "int":$v0);
  // CHECK: InvalidTypeF has no parameter named 'v1'
  let assemblyFormat = "`<` $v0 $v1 `>`";
}

// Test duplicate capture of parameter in capture-all struct.
def InvalidTypeG : InvalidType<"InvalidTypeG", "invalid_g"> {
  let parameters = (ins "int":$v0, "int":$v1, "int":$v2);
  // CHECK: duplicate parameter 'v0'
  let assemblyFormat = "`<` struct(params) $v0 `>`";
}

// Test capture-all struct duplicate capture.
def InvalidTypeH : InvalidType<"InvalidTypeH", "invalid_h"> {
  let parameters = (ins "int":$v0, "int":$v1, "int":$v2);
  // CHECK: `params` captures duplicate parameter: v0
  let assemblyFormat = "`<` $v0 struct(params) `>`";
}

// Test capture of parameter after `params` directive.
def InvalidTypeI : InvalidType<"InvalidTypeI", "invalid_i"> {
  let parameters = (ins "int":$v0);
  // CHECK: duplicate parameter 'v0'
  let assemblyFormat = "`<` params $v0 `>`";
}

// Test `struct` with optional parameter followed by comma.
def InvalidTypeJ : InvalidType<"InvalidTypeJ", "invalid_j"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  // CHECK: directive with optional parameters cannot be followed by a comma literal
  let assemblyFormat = "struct($a) `,` $b";
}

// Test `struct` in optional group must have all optional parameters.
def InvalidTypeK : InvalidType<"InvalidTypeK", "invalid_k"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  // CHECK: is only allowed in an optional group if all captured parameters are optional
  let assemblyFormat = "(`(` struct(params)^ `)`)?";
}

// Test `struct` in optional group must have all optional parameters.
def InvalidTypeL : InvalidType<"InvalidTypeL", "invalid_l"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  // CHECK: directive allowed in optional group only if all parameters are optional
  let assemblyFormat = "(`(` params^ `)`)?";
}

def InvalidTypeM : InvalidType<"InvalidTypeM", "invalid_m"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  // CHECK: parameters in an optional group must be optional
  let assemblyFormat = "(`(` $a^ `,` $b `)`)?";
}

def InvalidTypeN : InvalidType<"InvalidTypeN", "invalid_n"> {
  let parameters = (ins OptionalParameter<"int">:$a);
  // CHECK: optional group anchor must be a parameter or directive
  let assemblyFormat = "(`(` $a `)`^)?";
}

def InvalidTypeO : InvalidType<"InvalidTypeO", "invalid_o"> {
  let parameters = (ins "int":$a);
  // CHECK: 'ref' is only valid within a `custom` directive
  let assemblyFormat = "$a ref($a)";
}

def InvalidTypeP : InvalidType<"InvalidTypeP", "invalid_p"> {
  let parameters = (ins "int":$a);
  // CHECK: parameter 'a' must be bound before it is referenced
  let assemblyFormat = "custom<Foo>(ref($a)) $a";
}

def InvalidTypeQ : InvalidType<"InvalidTypeQ", "invalid_q"> {
  let parameters = (ins "int":$a);
  // CHECK: `params` can only be used at the top-level context or within a `struct` directive
  let assemblyFormat = "custom<Foo>(params)";
}

def InvalidTypeR : InvalidType<"InvalidTypeR", "invalid_r"> {
  let parameters = (ins "int":$a);
  // CHECK: `struct` can only be used at the top-level context
  let assemblyFormat = "custom<Foo>(struct(params))";
}

def InvalidTypeS : InvalidType<"InvalidTypeS", "invalid_s"> {
  let parameters = (ins OptionalParameter<"int">:$a, "int":$b);
  // CHECK: `custom` is only allowed in an optional group if all captured parameters are optional
  let assemblyFormat = "(`(` custom<Foo>($a, $b)^ `)`)?";
}

def InvalidTypeT : InvalidType<"InvalidTypeT", "invalid_t"> {
  let parameters = (ins OptionalParameter<"int">:$a);
  // CHECK: `custom` directive with no bound parameters cannot be used as optional group anchor
  let assemblyFormat = "$a (`(` custom<Foo>(ref($a))^ `)`)?";
}
