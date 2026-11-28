// RUN: mlir-tblgen -gen-dialect-decls -dialect=test -I %S/../../include %s | FileCheck %s --check-prefix=DECL
// RUN: mlir-tblgen -gen-dialect-defs -dialect=test -I %S/../../include %s | FileCheck %s --check-prefix=DEF
// RUN: mlir-tblgen -gen-dialect-decls -dialect=ext -I %S/../../include %s | FileCheck %s --check-prefix=EXT-DECL
// RUN: mlir-tblgen -gen-dialect-defs -dialect=ext -I %S/../../include %s | FileCheck %s --check-prefix=EXT-DEF

include "mlir/IR/DialectBase.td"

// Dialect declarations include the header declaring the dialect base class.
// The discardable attribute helpers additionally need the headers declaring
// `StringAttr` and `Operation`. The dialect definitions include the context
// header when loading dependent dialects.

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "::test";
  let dependentDialects = ["::test::OtherDialect"];
  let discardableAttrs = (ins "::mlir::UnitAttr":$discardable);
}

// DECL: #include "mlir/IR/BuiltinAttributes.h"
// DECL-NEXT: #include "mlir/IR/Dialect.h"
// DECL-NEXT: #include "mlir/IR/Operation.h"
// DECL-EMPTY:
// DECL-NOT: #include

// DEF: #include "mlir/IR/MLIRContext.h"
// DEF-EMPTY:
// DEF-NOT: #include

def Ext_Dialect : Dialect {
  let name = "ext";
  let cppNamespace = "::ext";
  let isExtensible = 1;
}

// EXT-DECL: #include "mlir/IR/ExtensibleDialect.h"
// EXT-DECL-EMPTY:
// EXT-DECL-NOT: #include

// EXT-DEF-NOT: #include
