//===- NVPTXIntrinsicInfo.td - NVPTX intrinsic mappings ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"

// Emit the address-space list as a reference to a constexpr C++ array.
class IntrinsicAddressSpaces<
    Intrinsic generic, Intrinsic specific = generic,
    list<int> addressSpaces = [3],
    bit trapOnInvalidAS = 1, bits<8> pointerOperand = 0> {
  Intrinsic GenericID = generic;
  Intrinsic SpecificID = specific;
  code AddressSpaces = [{AddressSpaceList<}] #
      !interleave(!foreach(as, addressSpaces, !cast<string>(as)), ", ") # [{>}];
  bit TrapOnInvalidAS = trapOnInvalidAS;
  bits<8> PointerOperand = pointerOperand;
}

def : IntrinsicAddressSpaces<int_nvvm_mbarrier_init>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_inval,
                            int_nvvm_mbarrier_inval_shared>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_check_layout>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_arrive,
                            int_nvvm_mbarrier_arrive_shared>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_arrive_noComplete,
                            int_nvvm_mbarrier_arrive_noComplete_shared>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_arrive_drop,
                            int_nvvm_mbarrier_arrive_drop_shared>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_arrive_drop_noComplete,
                            int_nvvm_mbarrier_arrive_drop_noComplete_shared>;
def : IntrinsicAddressSpaces<int_nvvm_mbarrier_test_wait,
                            int_nvvm_mbarrier_test_wait_shared>;
def : IntrinsicAddressSpaces<int_nvvm_cp_async_mbarrier_arrive,
                            int_nvvm_cp_async_mbarrier_arrive_shared>;
def : IntrinsicAddressSpaces<int_nvvm_cp_async_mbarrier_arrive_noinc,
                            int_nvvm_cp_async_mbarrier_arrive_noinc_shared>;
// Constant (AS 4) and entry-parameter (AS 101) variants. Other address spaces
// cannot be specialized but do not imply undefined behavior.
def : IntrinsicAddressSpaces<int_nvvm_prefetch_tensormap,
                            addressSpaces = [4, 101], trapOnInvalidAS = 0>;

def IntrinsicAddressSpaceTable : GenericTable {
  let FilterClass = "IntrinsicAddressSpaces";
  let CppTypeName = "IntrinsicAddressSpaceInfo";
  let Fields = ["GenericID", "SpecificID", "AddressSpaces", "TrapOnInvalidAS",
                "PointerOperand"];
  let PrimaryKey = ["GenericID"];
  let PrimaryKeyName = "getIntrinsicAddressSpaceInfo";
}
