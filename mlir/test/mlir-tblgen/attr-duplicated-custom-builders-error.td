// RUN: not mlir-tblgen -gen-attrdef-decls -I %S/../../include %s 2>&1 | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

class TestAttr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Test_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def TestAttr : TestAttr<"Test", "test"> {
  let summary = "Test attrubute";
  let description = "Test attribute";

  let parameters = (ins AttrParameter<"std::int64_t", "arg">:$arg);
  let builders = [AttrBuilder<(ins "std::int64_t":$arg), [{
            return $_get($_ctxt, arg);
        }]>,
                  AttrBuilder<(ins "std::int64_t":$arg), [{
            // Duplicated builder
            return $_get($_ctxt, arg);
        }]>];

  let assemblyFormat = "`<` $arg `>`";

  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let genMnemonicAlias = 1;
}

def Test_TestAttrOp : Op<Test_Dialect, "test", []> {
  let summary = "test operation with attribute";
  let description = "test operation with attribute";

  let arguments = (ins TestAttr:$testAttr);
  let assemblyFormat = "$testAttr attr-dict";
}

// CHECK: attr-duplicated-custom-builders-error.td:20:7: error: builder `get` conflicts with an existing builder.
// CHECK-NEXT:   let builders = [AttrBuilder<(ins "std::int64_t":$arg), [{
// CHECK-NEXT:   ^
// CHECK-NEXT: note: A new builder with signature:
// CHECK-NEXT: static TestAttr get(::mlir::MLIRContext *context, std::int64_t arg);
// CHECK-EMPTY:
// CHECK-NEXT: is shadowed by an existing builder with signature:
// CHECK-NEXT: static TestAttr get(::mlir::MLIRContext *context, std::int64_t arg);
// CHECK-EMPTY:
// CHECK-NEXT: Please remove one of the conflicting definitions.
