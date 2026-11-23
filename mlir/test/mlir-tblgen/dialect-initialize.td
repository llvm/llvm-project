// RUN: mlir-tblgen -gen-dialect-decls -dialect=both -I %S/../../include %s | FileCheck %s --check-prefix=BOTH-DECL
// RUN: mlir-tblgen -gen-dialect-defs -dialect=both -I %S/../../include %s | FileCheck %s --check-prefix=BOTH
// RUN: mlir-tblgen -gen-dialect-decls -dialect=attrs -I %S/../../include %s | FileCheck %s --check-prefix=ATTRS-DECL
// RUN: mlir-tblgen -gen-dialect-defs -dialect=attrs -I %S/../../include %s | FileCheck %s --check-prefix=ATTRS
// RUN: mlir-tblgen -gen-dialect-decls -dialect=custom -I %S/../../include %s | FileCheck %s --check-prefix=CUSTOM-DECL
// RUN: mlir-tblgen -gen-dialect-defs -dialect=custom -I %S/../../include %s | FileCheck %s --check-prefix=CUSTOM
// RUN: mlir-tblgen -gen-attrdef-defs -attrdefs-dialect=custom -I %S/../../include %s | FileCheck %s --check-prefix=CUSTOM-ATTRS
// RUN: mlir-tblgen -gen-typedef-decls -typedefs-dialect=custom -I %S/../../include %s | FileCheck %s --check-prefix=CUSTOM-TYPES
// RUN: mlir-tblgen -gen-typedef-defs -typedefs-dialect=custom -I %S/../../include %s | FileCheck %s --check-prefix=CUSTOM-TYPES
// RUN: mlir-tblgen -gen-dialect-decls -dialect=none -I %S/../../include %s | FileCheck %s --check-prefix=NONE-DECL
// RUN: mlir-tblgen -gen-dialect-defs -dialect=none -I %S/../../include %s | FileCheck %s --check-prefix=NONE
// RUN: mlir-tblgen -gen-attrdef-decls -attrdefs-dialect=none -I %S/../../include %s | FileCheck %s --check-prefix=NONE-HOOK
// RUN: mlir-tblgen -gen-attrdef-defs -attrdefs-dialect=none -I %S/../../include %s | FileCheck %s --check-prefix=NONE-HOOK

include "mlir/IR/AttrTypeBase.td"

// `hasAttributes` (or `hasTypes`) provides the default of
// `useDefaultAttributeRegistration` and `useDefaultAttributePrinterParser` (or
// their type counterparts). The constructor registers the attributes and types
// independently of whether the dialect has a custom destructor.

def Both_Dialect : Dialect {
  let name = "both";
  let cppNamespace = "::both";
  let hasAttributes = 1;
  let hasTypes = 1;
  let hasNonDefaultDestructor = 1;
  let dependentDialects = ["::foo::FooDialect"];
}

// BOTH-DECL: /// Parse an attribute registered to this dialect.
// BOTH-DECL: /// Parse a type registered to this dialect.

// BOTH: void registerBothDialectAttributes(BothDialect *dialect);
// BOTH-NEXT: void registerBothDialectTypes(BothDialect *dialect);
// BOTH: BothDialect::BothDialect(::mlir::MLIRContext *context)
// BOTH: getContext()->loadDialect<::foo::FooDialect>();
// BOTH-NEXT: registerBothDialectAttributes(this);
// BOTH-NEXT: registerBothDialectTypes(this);
// BOTH-NEXT: initialize();
// BOTH-NEXT: }
// BOTH-NOT: ~BothDialect

def Attrs_Dialect : Dialect {
  let name = "attrs";
  let cppNamespace = "::attrs";
  let hasAttributes = 1;
}

// ATTRS-DECL: /// Parse an attribute registered to this dialect.
// ATTRS-DECL-NOT: /// Parse a type registered to this dialect.

// ATTRS-NOT: Types
// ATTRS: void registerAttrsDialectAttributes(AttrsDialect *dialect);
// ATTRS: AttrsDialect::AttrsDialect(::mlir::MLIRContext *context)
// ATTRS: registerAttrsDialectAttributes(this);
// ATTRS-NEXT: initialize();
// ATTRS-NEXT: }
// ATTRS: AttrsDialect::~AttrsDialect() = default;
// ATTRS-NOT: Types

// The `use` flags override the default provided by `hasAttributes` and
// `hasTypes`. The registration hook is only generated for the dialect's default
// registration.

def Custom_Dialect : Dialect {
  let name = "custom";
  let cppNamespace = "::custom";
  let hasAttributes = 1;
  let hasTypes = 1;
  let useDefaultAttributePrinterParser = 0;
  let useDefaultTypeRegistration = 0;
}
def Custom_Attr : AttrDef<Custom_Dialect, "Foo"> { let mnemonic = "foo"; }
def Custom_Type : TypeDef<Custom_Dialect, "Bar"> { let mnemonic = "bar"; }

// CUSTOM-DECL-NOT: /// Parse an attribute registered to this dialect.
// CUSTOM-DECL: /// Parse a type registered to this dialect.

// CUSTOM-NOT: Types
// CUSTOM: registerCustomDialectAttributes(this);
// CUSTOM-NEXT: initialize();
// CUSTOM-NOT: Types

// CUSTOM-ATTRS: void registerCustomDialectAttributes(CustomDialect *dialect) {
// CUSTOM-TYPES-NOT: registerCustomDialectTypes

// By default, nothing is registered and no hook is generated, even if the
// definitions are visible.

def None_Dialect : Dialect {
  let name = "none";
  let cppNamespace = "::none";
}
def None_Attr : AttrDef<None_Dialect, "Foo"> { let mnemonic = "foo"; }
def None_Type : TypeDef<None_Dialect, "Bar"> { let mnemonic = "bar"; }

// NONE-DECL-NOT: registered to this dialect
// NONE-NOT: register
// NONE-HOOK-NOT: registerNoneDialectAttributes
