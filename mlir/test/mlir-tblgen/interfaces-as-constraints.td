// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/IR/OpBase.td"

def Test_Dialect : Dialect {
  let name = "test";

  // TODO: Flip to _Prefixed
  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
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
// CHECK:   if (!((type.isa<TopLevelTypeInterface>()))) {
// CHECK-NEXT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:        << " must be TopLevelTypeInterface instance, but got " << type;

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_type_constraint.*}}(
// CHECK:   if (!((type.isa<test::TypeInterfaceInNamespace>()))) {
// CHECK-NEXT:    return op->emitOpError(valueKind) << " #" << valueIndex
// CHECK-NEXT:        << " must be TypeInterfaceInNamespace instance, but got " << type;

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_attr_constraint.*}}(
// CHECK:   if (attr && !((attr.isa<TopLevelAttrInterface>()))) {
// CHECK-NEXT:    return op->emitOpError("attribute '") << attrName
// CHECK-NEXT:        << "' failed to satisfy constraint: TopLevelAttrInterface instance";

// CHECK: static ::mlir::LogicalResult {{__mlir_ods_local_attr_constraint.*}}(
// CHECK:   if (attr && !((attr.isa<test::AttrInterfaceInNamespace>()))) {
// CHECK-NEXT:    return op->emitOpError("attribute '") << attrName
// CHECK-NEXT:        << "' failed to satisfy constraint: AttrInterfaceInNamespace instance";

// CHECK: TopLevelAttrInterface OpUsingAllOfThose::attr1()
// CHECK: test::AttrInterfaceInNamespace OpUsingAllOfThose::attr2()
