// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";
}

def TopLevelTypeInterface : TypeInterface<"TopLevelTypeInterface">;

def TypeInterfaceInNamespace : TypeInterface<"TypeInterfaceInNamespace"> {
	let cppNamespace = "test";
}

def TopLevelAttrInterface : AttrInterface<"TopLevelAttrInterface">;

def AttrInterfaceInNamespace : AttrInterface<"AttrInterfaceInNamespace"> {
	let cppNamespace = "test";
}

def OpUsingAllOfThose : Op<Test_Dialect, "OpUsingAllOfThose"> {
	let arguments = (ins TopLevelAttrInterface:$attr1, AttrInterfaceInNamespace:$attr2);
	let results = (outs TopLevelTypeInterface:$res1, TypeInterfaceInNamespace:$res2);
}

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_type_constraint.*}}(
// CHECK:   if (!((::llvm::isa<TopLevelTypeInterface>(type)))) {
// CHECK-NEXT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:        << " must be TopLevelTypeInterface instance, but got " << type;

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_type_constraint.*}}(
// CHECK:   if (!((::llvm::isa<test::TypeInterfaceInNamespace>(type)))) {
// CHECK-NEXT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:        << " must be TypeInterfaceInNamespace instance, but got " << type;

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_attr_constraint.*}}(
// CHECK:   if (attr && !((::llvm::isa<TopLevelAttrInterface>(attr))))
// CHECK-NEXT:     return getDiag() << "attribute '" << attrName
// CHECK-NEXT:        << "' failed to satisfy constraint: TopLevelAttrInterface instance";

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_attr_constraint.*}}(
// CHECK:   if (attr && !((::llvm::isa<test::AttrInterfaceInNamespace>(attr))))
// CHECK-NEXT:    return getDiag() << "attribute '" << attrName
// CHECK-NEXT:        << "' failed to satisfy constraint: AttrInterfaceInNamespace instance";

// CHECK: TopLevelAttrInterface OpUsingAllOfThose::getAttr1()
// CHECK: test::AttrInterfaceInNamespace OpUsingAllOfThose::getAttr2()
