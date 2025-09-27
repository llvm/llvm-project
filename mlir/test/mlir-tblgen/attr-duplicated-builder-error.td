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
        }]>];

  let assemblyFormat = "`<` $arg `>`";

  let skipDefaultBuilders = 0;
  let genVerifyDecl = 1;
  let genMnemonicAlias = 1;
}

def Test_TestAttrOp : Op<Test_Dialect, "test", []> {
  let summary = "test operation with attribute";
  let description = "test operation with attribute";

  let arguments = (ins TestAttr:$testAttr);
  let assemblyFormat = "$testAttr attr-dict";
}

// CHECK: attr-duplicated-builder-error.td:15:5: error: Unexpected overlap when generating `get` for TestAttr (from line 537)
