// RUN: mlir-tblgen -gen-op-decls -op-include-regex='^test\.wrappers$' -I %S/../../include %s | FileCheck %s --check-prefix=WRAPPERS
// RUN: mlir-tblgen -gen-op-decls -op-include-regex='^test\.any_of$' -I %S/../../include %s | FileCheck %s --check-prefix=ANYOF
// RUN: mlir-tblgen -gen-op-decls -op-include-regex='^test\.attrs$' -I %S/../../include %s | FileCheck %s --check-prefix=ATTRS
// RUN: mlir-tblgen -gen-attrdef-decls -I %S/../../include %s | FileCheck %s --check-prefix=PARAMS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
}

// Dialect types and attributes set their header in their base class.
class Test_Type<string name> : TypeDef<Test_Dialect, name> {
  let cppIncludes = ["test/TestTypes.h"];
}
class Test_Attr<string name> : AttrDef<Test_Dialect, name> {
  let cppIncludes = ["test/TestAttrs.h"];
}

def Test_FooType : Test_Type<"Foo"> { let mnemonic = "foo"; }
def Test_FooAttr : Test_Attr<"Foo"> { let mnemonic = "foo"; }
def Test_OtherType : TypeDef<Test_Dialect, "Other"> {
  let mnemonic = "other";
  let cppIncludes = ["test/OtherTypes.h", "test/OtherTypeUtils.h"];
}

//===----------------------------------------------------------------------===//
// Ops
//===----------------------------------------------------------------------===//

// Type constraint wrappers keep the headers of the wrapped constraint.

// WRAPPERS-LABEL: #undef GET_OP_CLASSES
// WRAPPERS-EMPTY:
// WRAPPERS-NEXT: #include "mlir/IR/OpDefinition.h"
// WRAPPERS-NEXT: #include "test/OtherTypeUtils.h"
// WRAPPERS-NEXT: #include "test/OtherTypes.h"
// WRAPPERS-NEXT: #include "test/TestTypes.h"
// WRAPPERS-EMPTY:
def WrappersOp : Op<Test_Dialect, "wrappers"> {
  let arguments = (ins Variadic<Test_OtherType>:$others);
  let results = (outs Optional<Test_FooType>:$foo);
}

// `AnyTypeOf` keeps the headers of all allowed types.

// ANYOF-LABEL: #undef GET_OP_CLASSES
// ANYOF-EMPTY:
// ANYOF-NEXT: #include "mlir/IR/OpDefinition.h"
// ANYOF-NEXT: #include "test/OtherTypeUtils.h"
// ANYOF-NEXT: #include "test/OtherTypes.h"
// ANYOF-NEXT: #include "test/TestTypes.h"
// ANYOF-EMPTY:
def AnyOfOp : Op<Test_Dialect, "any_of"> {
  let arguments = (ins AnyTypeOf<[Test_FooType, Test_OtherType]>:$value);
}

// Attribute wrappers keep the headers of the wrapped attribute. An attribute
// needs headers for both its attribute class and its return type, e.g.
// `AffineMapAttr` and `AffineMap`.

// ATTRS-LABEL: #undef GET_OP_CLASSES
// ATTRS-EMPTY:
// ATTRS-NEXT: #include "mlir/Bytecode/BytecodeOpInterface.h"
// ATTRS-NEXT: #include "mlir/IR/AffineMap.h"
// ATTRS-NEXT: #include "mlir/IR/BuiltinAttributes.h"
// ATTRS-NEXT: #include "mlir/IR/OpDefinition.h"
// ATTRS-NEXT: #include "test/OtherTypeUtils.h"
// ATTRS-NEXT: #include "test/OtherTypes.h"
// ATTRS-NEXT: #include "test/TestAttrs.h"
// ATTRS-EMPTY:
def AttrsOp : Op<Test_Dialect, "attrs"> {
  let arguments = (ins OptionalAttr<Test_FooAttr>:$foo,
                       AffineMapAttr:$map,
                       TypeAttrOf<Test_OtherType>:$type);
}

//===----------------------------------------------------------------------===//
// Attribute and type parameters
//===----------------------------------------------------------------------===//

// Parameters that are types or attributes include their headers.

// PARAMS-LABEL: #undef GET_ATTRDEF_CLASSES
// PARAMS-EMPTY:
// PARAMS-NEXT: #include "mlir/IR/Attributes.h"
// PARAMS-NEXT: #include "test/OtherTypeUtils.h"
// PARAMS-NEXT: #include "test/OtherTypes.h"
// PARAMS-EMPTY:
def Test_ParamAttr : AttrDef<Test_Dialect, "Param"> {
  let mnemonic = "param";
  let parameters = (ins Test_OtherType:$type, "int":$value);
  let assemblyFormat = "`<` $type `,` $value `>`";
}
