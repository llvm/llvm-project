// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include -DTEST_INTRINSICS_SUPPRESS_DEFS %s | FileCheck %s
// XFAIL: vg_leak

include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"

def DummyTarget : Target;

def int_abc : Intrinsic<[llvm_anyint_ty]>;
def int_xyz : Intrinsic<[llvm_anyint_ty]>;

let TargetPrefix = "gtarget" in {
  def int_gtarget_def : Intrinsic<[llvm_anyint_ty]>;
  def int_gtarget_defg : Intrinsic<[llvm_anyint_ty]>;
  def int_gtarget_uvw : Intrinsic<[llvm_anyint_ty]>;
}

let TargetPrefix = "ftarget" in {
  def int_ftarget_ghi : Intrinsic<[llvm_anyint_ty]>;
  def int_ftarget_ghi_x : Intrinsic<[llvm_anyint_ty]>;
  def int_ftarget_rst : Intrinsic<[llvm_anyint_ty]>;
}

class Table<Intrinsic intr, int payload> : SearchableTable {
  let SearchableFields = ["Intr"];
  let EnumNameField = ?;

  Intrinsic Intr = !cast<Intrinsic>(intr);
  bits<16> Payload = payload;
}

// CHECK-LABEL: TablesList[] = {
// CHECK-DAG: { Intrinsic::abc, 0x0 },
// CHECK-DAG: { Intrinsic::xyz, 0x1 },
// CHECK-DAG: { Intrinsic::gtarget_def, 0x10 },
// CHECK-DAG: { Intrinsic::gtarget_defg, 0x11 },
// CHECK-DAG: { Intrinsic::gtarget_uvw, 0x12 },
// CHECK-DAG: { Intrinsic::ftarget_ghi, 0x20 },
// CHECK-DAG: { Intrinsic::ftarget_ghi_x, 0x21 },
// CHECK-DAG: { Intrinsic::ftarget_rst, 0x22 },

// Check that the index is in the correct order, consistent with the ordering
// of enums: alphabetically, but target intrinsics after generic intrinsics
//
// CHECK-LABEL: lookupTableByIntr(unsigned Intr) {
// CHECK: Index[] = {
// CHECK-NEXT: Intrinsic::abc
// CHECK-NEXT: Intrinsic::xyz
// CHECK-NEXT: Intrinsic::ftarget_ghi
// CHECK-NEXT: Intrinsic::ftarget_ghi_x
// CHECK-NEXT: Intrinsic::ftarget_rst
// CHECK-NEXT: Intrinsic::gtarget_def
// CHECK-NEXT: Intrinsic::gtarget_defg
// CHECK-NEXT: Intrinsic::gtarget_uvw

def : Table<int_abc, 0x0>;
def : Table<int_xyz, 0x1>;
def : Table<int_gtarget_def, 0x10>;
def : Table<int_gtarget_defg, 0x11>;
def : Table<int_gtarget_uvw, 0x12>;
def : Table<int_ftarget_ghi, 0x20>;
def : Table<int_ftarget_ghi_x, 0x21>;
def : Table<int_ftarget_rst, 0x22>;
