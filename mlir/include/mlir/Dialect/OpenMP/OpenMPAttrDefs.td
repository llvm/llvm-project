//=== OpenMPAttrDefs.td - OpenMP Attributes definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef OPENMP_ATTR_DEFS
#define OPENMP_ATTR_DEFS

include "mlir/Dialect/OpenMP/OpenMPDialect.td"
include "mlir/Dialect/OpenMP/OpenMPEnums.td"
include "mlir/Dialect/OpenMP/OpenMPOpsInterfaces.td"
include "mlir/Dialect/OpenMP/OpenMPTypeInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"

class OpenMP_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                  string baseCppClass = "::mlir::Attribute">
    : AttrDef<OpenMP_Dialect, name, traits, baseCppClass> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// DeclareTargetAttr
//===----------------------------------------------------------------------===//

def DeclareTargetAttr : OpenMP_Attr<"DeclareTarget", "declaretarget"> {
  let parameters = (ins
    OptionalParameter<"DeclareTargetDeviceTypeAttr">:$device_type,
    OptionalParameter<"DeclareTargetCaptureClauseAttr">:$capture_clause
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// FlagsAttr
//===----------------------------------------------------------------------===//

// Runtime library flags attribute that holds information for lowering to LLVM.
def FlagsAttr : OpenMP_Attr<"Flags", "flags"> {
  let parameters = (ins
    DefaultValuedParameter<"uint32_t", "0">:$debug_kind,
    DefaultValuedParameter<"bool", "false">:$assume_teams_oversubscription,
    DefaultValuedParameter<"bool", "false">:$assume_threads_oversubscription,
    DefaultValuedParameter<"bool", "false">:$assume_no_thread_state,
    DefaultValuedParameter<"bool", "false">:$assume_no_nested_parallelism,
    DefaultValuedParameter<"bool", "false">:$no_gpu_lib,
    DefaultValuedParameter<"uint32_t", "50">:$openmp_device_version
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

//===----------------------------------------------------------------------===//
// TaskDependArrayAttr
//===----------------------------------------------------------------------===//

def TaskDependArrayAttr
    : TypedArrayAttrBase<ClauseTaskDependAttr,
                         ClauseTaskDependAttr.summary # " array"> {
  let constBuilderCall = ?;
}

//===----------------------------------------------------------------------===//
// VersionAttr
//===----------------------------------------------------------------------===//

def VersionAttr : OpenMP_Attr<"Version", "version"> {
  let parameters = (ins
    "uint32_t":$version
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // OPENMP_ATTR_DEFS
