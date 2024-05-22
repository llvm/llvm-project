// RUN: mlir-tblgen -gen-op-decls -I %S/../../include %s | FileCheck %s

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
  let cppNamespace = "foobar";
}
class NS_Op<string mnemonic, list<Trait> traits> :
    Op<Test_Dialect, mnemonic, traits>;

def SomeAttr : Attr<CPred<"some-condition">, "some attribute kind"> {
  let storageType = "some-attr-kind";
  let returnType = "some-return-type";
  let convertFromStorage = "$_self.some-convert-from-storage()";
  let constBuilderCall = "some-const-builder-call($_builder, $0)";
}

def AOp : NS_Op<"a_op", []> {
  let arguments = (ins
      FloatLike:$lhs,
      SomeAttr:$aAttr,
      DefaultValuedAttr<SomeAttr, "4.2">:$bAttr,
      OptionalAttr<SomeAttr>:$cAttr,
      DefaultValuedOptionalAttr<SomeAttr, "7.2">:$dAttr
  );
}

// CHECK-LABEL: AOp declarations
// Note: `cAttr` below could be conditionally optional and so the generation is
// currently conservative.
// CHECK-DAG: ::mlir::Value lhs, some-attr-kind aAttr, some-attr-kind bAttr, /*optional*/some-attr-kind cAttr, /*optional*/some-attr-kind dAttr);
// CHECK-DAG: ::mlir::Value lhs, some-return-type aAttr, some-return-type bAttr, /*optional*/some-attr-kind cAttr, /*optional*/some-return-type dAttr = 7.2);
// CHECK-DAG: ::mlir::TypeRange resultTypes, ::mlir::Value lhs, some-attr-kind aAttr, some-attr-kind bAttr, /*optional*/some-attr-kind cAttr, /*optional*/some-attr-kind dAttr);
// CHECK-DAG: ::mlir::TypeRange resultTypes, ::mlir::Value lhs, some-return-type aAttr, some-return-type bAttr, /*optional*/some-attr-kind cAttr, /*optional*/some-return-type dAttr = 7.2);

def BOp : NS_Op<"b_op", []> {
  let arguments = (ins
      DefaultValuedAttr<SomeAttr, "6.2">:$aAttr,
      DefaultValuedAttr<SomeAttr, "4.2">:$bAttr
  );
}

// Verify that non-overlapping builders created where all could be elided.
// CHECK-LABEL: BOp declarations
// CHECK-DAG: some-attr-kind aAttr, some-attr-kind bAttr = nullptr);
// CHECK-DAG: some-return-type aAttr = 6.2, some-return-type bAttr = 4.2);
// CHECK-DAG: ::mlir::TypeRange resultTypes, some-attr-kind aAttr, some-attr-kind bAttr = nullptr);
// CHECK-DAG: ::mlir::TypeRange resultTypes, some-return-type aAttr = 6.2, some-return-type bAttr = 4.2);

def COp : NS_Op<"c_op", []> {
  let arguments = (ins
    FloatLike:$value,
    OptionalAttr<SymbolRefArrayAttr>:$ag,
    OptionalAttr<SymbolRefArrayAttr>:$as,
    OptionalAttr<SymbolRefArrayAttr>:$nos,
    OptionalAttr<I64Attr>:$al,
    UnitAttr:$vo,
    UnitAttr:$non
  );
}

// CHECK-LABEL: COp declarations
// Note: `al` below could be conditionally optional and so the generation is
// currently conservative.
// CHECK-DAG: ::mlir::Value value, /*optional*/::mlir::ArrayAttr ag, /*optional*/::mlir::ArrayAttr as, /*optional*/::mlir::ArrayAttr nos, /*optional*/::mlir::IntegerAttr al, /*optional*/::mlir::UnitAttr vo, /*optional*/::mlir::UnitAttr non = nullptr);
// CHECK-DAG: ::mlir::Value value, /*optional*/::mlir::ArrayAttr ag, /*optional*/::mlir::ArrayAttr as, /*optional*/::mlir::ArrayAttr nos, /*optional*/::mlir::IntegerAttr al, /*optional*/bool vo = false, /*optional*/bool non = false);
// CHECK-DAG: ::mlir::TypeRange resultTypes, ::mlir::Value value, /*optional*/::mlir::ArrayAttr ag, /*optional*/::mlir::ArrayAttr as, /*optional*/::mlir::ArrayAttr nos, /*optional*/::mlir::IntegerAttr al, /*optional*/::mlir::UnitAttr vo, /*optional*/::mlir::UnitAttr non = nullptr);
// CHECK-DAG: ::mlir::TypeRange resultTypes, ::mlir::Value value, /*optional*/::mlir::ArrayAttr ag, /*optional*/::mlir::ArrayAttr as, /*optional*/::mlir::ArrayAttr nos, /*optional*/::mlir::IntegerAttr al, /*optional*/bool vo = false, /*optional*/bool non = false);
