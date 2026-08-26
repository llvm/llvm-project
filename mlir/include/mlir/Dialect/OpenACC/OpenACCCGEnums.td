//===- OpenACCCGEnums.td - OpenACC codegen enums -----------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines OpenACC codegen enums used by intermediate operations that do not
// map directly to OpenACC language constructs.
//
//===----------------------------------------------------------------------===//

#ifndef OPENACCCG_ENUMS
#define OPENACCCG_ENUMS

// Descriptor layouts. Bit positions match the runtime TGT_ACC_DESC_* values.
def OpenACC_DataDescKindNone : I32BitEnumAttrCaseNone<"none">;
def OpenACC_DataDescKindCFI : I32BitEnumAttrCaseBit<"cfi", 0>;
def OpenACC_DataDescKindMemref : I32BitEnumAttrCaseBit<"memref", 1>;
// Overlay of OpenACC bounds nested on a base descriptor (cfi / memref /
// generic). Combines with the cases above; matches TGT_ACC_DESC_OPENACC.
def OpenACC_DataDescKindOpenACC : I32BitEnumAttrCaseBit<"openacc", 12>;

def OpenACC_DataDescKindEnum : I32BitEnumAttr<
    "DataDescKind",
    "OpenACC offload descriptor kind (may be combined for nested descriptors)",
    [OpenACC_DataDescKindNone, OpenACC_DataDescKindCFI,
     OpenACC_DataDescKindMemref, OpenACC_DataDescKindOpenACC]> {
  let separator = ",";
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

def OpenACC_DataDescKindAttr
    : EnumAttr<OpenACC_Dialect, OpenACC_DataDescKindEnum, "desc_kind">;

// Offload map-type flags. Bit positions match the runtime TGT_ACC_MAPTYPE_*
// values.
def OpenACC_MapFlagsNone : I32BitEnumAttrCaseNone<"none">;
def OpenACC_MapFlagsTo : I32BitEnumAttrCaseBit<"to", 0>;
def OpenACC_MapFlagsFrom : I32BitEnumAttrCaseBit<"from", 1>;
def OpenACC_MapFlagsDelete : I32BitEnumAttrCaseBit<"delete_", 3, "delete">;
def OpenACC_MapFlagsPtrAndObj : I32BitEnumAttrCaseBit<"ptr_and_obj", 4>;
def OpenACC_MapFlagsPrivate : I32BitEnumAttrCaseBit<"private_", 7, "private">;
def OpenACC_MapFlagsLiteral : I32BitEnumAttrCaseBit<"literal", 8>;
def OpenACC_MapFlagsImplicit : I32BitEnumAttrCaseBit<"implicit", 9>;
def OpenACC_MapFlagsDevptr : I32BitEnumAttrCaseBit<"devptr", 10>;
def OpenACC_MapFlagsManagedDevptr : I32BitEnumAttrCaseBit<"managed_devptr", 11>;
def OpenACC_MapFlagsNoCreate : I32BitEnumAttrCaseBit<"no_create", 13>;
def OpenACC_MapFlagsGangPrivate : I32BitEnumAttrCaseBit<"gang_private", 14>;
def OpenACC_MapFlagsWorkerPrivate : I32BitEnumAttrCaseBit<"worker_private", 15>;
def OpenACC_MapFlagsVectorPrivate : I32BitEnumAttrCaseBit<"vector_private", 16>;
def OpenACC_MapFlagsInitZero : I32BitEnumAttrCaseBit<"init_zero", 17>;
def OpenACC_MapFlagsDeviceResident
    : I32BitEnumAttrCaseBit<"device_resident", 18>;
def OpenACC_MapFlagsIfPresent : I32BitEnumAttrCaseBit<"if_present", 19>;
def OpenACC_MapFlagsPresent : I32BitEnumAttrCaseBit<"present", 20>;
def OpenACC_MapFlagsDescriptor : I32BitEnumAttrCaseBit<"descriptor", 21>;
def OpenACC_MapFlagsReduction : I32BitEnumAttrCaseBit<"reduction", 22>;

def OpenACC_MapFlagsEnum : I32BitEnumAttr<
    "MapFlags", "OpenACC offload map-type flags",
    [OpenACC_MapFlagsNone, OpenACC_MapFlagsTo, OpenACC_MapFlagsFrom,
     OpenACC_MapFlagsDelete, OpenACC_MapFlagsPtrAndObj, OpenACC_MapFlagsPrivate,
     OpenACC_MapFlagsLiteral, OpenACC_MapFlagsImplicit, OpenACC_MapFlagsDevptr,
     OpenACC_MapFlagsManagedDevptr, OpenACC_MapFlagsNoCreate,
     OpenACC_MapFlagsGangPrivate, OpenACC_MapFlagsWorkerPrivate,
     OpenACC_MapFlagsVectorPrivate, OpenACC_MapFlagsInitZero,
     OpenACC_MapFlagsDeviceResident, OpenACC_MapFlagsIfPresent,
     OpenACC_MapFlagsPresent, OpenACC_MapFlagsDescriptor,
     OpenACC_MapFlagsReduction]> {
  let separator = ",";
  let cppNamespace = "::mlir::acc";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
  let printBitEnumQuoted = 0;
}

def OpenACC_MapFlagsAttr
    : EnumAttr<OpenACC_Dialect, OpenACC_MapFlagsEnum, "map_flags">;

#endif // OPENACCCG_ENUMS
