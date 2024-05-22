//===-- OpenMPOpsInterfaces.td - OpenMP op interfaces ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the OpenMP Dialect interfaces definition file.
//
//===----------------------------------------------------------------------===//

#ifndef OpenMP_OPS_INTERFACES
#define OpenMP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def OutlineableOpenMPOpInterface : OpInterface<"OutlineableOpenMPOpInterface"> {
  let description = [{
    OpenMP operations whose region will be outlined will implement this
    interface. These operations will
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<"Get alloca block", "::mlir::Block*", "getAllocaBlock",
      (ins), [{
      return &$_op.getRegion().front();
      }]>,
  ];
}

def ReductionClauseInterface : OpInterface<"ReductionClauseInterface"> {
  let description = [{
    OpenMP operations that support reduction clause have this interface.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      "Get reduction vars", "::mlir::SmallVector<::mlir::Value>",
      "getAllReductionVars", (ins), [{}], [{
        return $_op.getReductionVars();
      }]>,
  ];
}

def DeclareTargetInterface : OpInterface<"DeclareTargetInterface"> {
  let description = [{
    OpenMP operations that support declare target have this interface.
    For example, FuncOp's and llvm.GlobalOp/fir.GlobalOp's. This
    interface allows simple manipulation and introspection of the
    declare target attribute that can be applied to these operations.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Set the declare target attribute on the current operation with the
        specified attribute arguments.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setDeclareTarget",
      (ins "mlir::omp::DeclareTargetDeviceType":$deviceType,
            "mlir::omp::DeclareTargetCaptureClause":$captureClause), [{}], [{
        $_op->setAttr("omp.declare_target",
                  mlir::omp::DeclareTargetAttr::get(
                      $_op->getContext(),
                      mlir::omp::DeclareTargetDeviceTypeAttr::get(
                          $_op->getContext(), deviceType),
                      mlir::omp::DeclareTargetCaptureClauseAttr::get(
                          $_op->getContext(), captureClause)));
      }]>,
      InterfaceMethod<
      /*description=*/[{
        Checks if the declare target attribute has been applied and exists on the
        current operation. Returns true if it exists on it, otherwise returns
        false.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isDeclareTarget",
      (ins), [{}], [{
        return $_op->hasAttr("omp.declare_target");
      }]>,
      InterfaceMethod<
      /*description=*/[{
        Returns the DeclareTargetDeviceType segment of the DeclareTarget attribute if it
        exists on the current operation. Otherwise it returns null.
      }],
      /*retTy=*/"mlir::omp::DeclareTargetDeviceType",
      /*methodName=*/"getDeclareTargetDeviceType",
      (ins), [{}], [{
        if (mlir::Attribute dTar = $_op->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            return dAttr.getDeviceType().getValue();
        return {};
      }]>,
      InterfaceMethod<
      /*description=*/[{
        Returns the DeclareTargetCaptureClause segment of the DeclareTarget attribute if it
        exists on the current operation. Otherwise it returns null.
      }],
      /*retTy=*/"mlir::omp::DeclareTargetCaptureClause",
      /*methodName=*/"getDeclareTargetCaptureClause",
      (ins), [{}], [{
        if (mlir::Attribute dTar = $_op->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            return dAttr.getCaptureClause().getValue();
        return {};
      }]>
  ];
}

def OffloadModuleInterface : OpInterface<"OffloadModuleInterface"> {
  let description = [{
    Operations that represent a module for offloading (host or device)
    should have this interface.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      /*description=*/[{
      Set the attribute on the current module with the specified boolean
      argument.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setIsTargetDevice",
      (ins "bool":$isTargetDevice), [{}], [{
        $_op->setAttr(
          mlir::StringAttr::get($_op->getContext(), llvm::Twine{"omp.is_target_device"}),
            mlir::BoolAttr::get($_op->getContext(), isTargetDevice));
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Get the attribute on the current module if it exists and
        return its value, if it doesn't exist it returns false by default.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getIsTargetDevice",
      (ins), [{}], [{
        if (Attribute isTargetDevice = $_op->getAttr("omp.is_target_device"))
          if (isTargetDevice.isa<mlir::BoolAttr>())
           return isTargetDevice.dyn_cast<BoolAttr>().getValue();
        return false;
      }]>,
    InterfaceMethod<
      /*description=*/[{
      Set the attribute on the current module with the specified boolean
      argument.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setIsGPU",
      (ins "bool":$isGPU), [{}], [{
        $_op->setAttr(
          mlir::StringAttr::get($_op->getContext(), "omp.is_gpu"),
            mlir::BoolAttr::get($_op->getContext(), isGPU));
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Get the attribute on the current module if it exists and
        return its value, if it doesn't exist it returns false by default.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getIsGPU",
      (ins), [{}], [{
        if (Attribute isTargetCGAttr = $_op->getAttr("omp.is_gpu"))
          if (auto isTargetCGVal = isTargetCGAttr.dyn_cast<BoolAttr>())
           return isTargetCGVal.getValue();
        return false;
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Get the FlagsAttr attribute on the current module if it exists
        and return the attribute, if it doesn't exit it returns a nullptr
      }],
      /*retTy=*/"mlir::omp::FlagsAttr",
      /*methodName=*/"getFlags",
      (ins), [{}], [{
        if (Attribute flags = $_op->getAttr("omp.flags"))
          return ::llvm::dyn_cast_or_null<mlir::omp::FlagsAttr>(flags);
        return nullptr;
      }]>,
      InterfaceMethod<
      /*description=*/[{
        Apply an omp.FlagsAttr to a module with the specified values
        for the flags
      }],
      /*retTy=*/"void",
      /*methodName=*/"setFlags",
      (ins "uint32_t":$debugKind,
            "bool":$assumeTeamsOversubscription,
            "bool":$assumeThreadsOversubscription,
            "bool":$assumeNoThreadState,
            "bool":$assumeNoNestedParallelism,
            "uint32_t":$openmpDeviceVersion), [{}], [{
        $_op->setAttr(("omp." + mlir::omp::FlagsAttr::getMnemonic()).str(),
                  mlir::omp::FlagsAttr::get($_op->getContext(), debugKind,
                      assumeTeamsOversubscription, assumeThreadsOversubscription,
                      assumeNoThreadState, assumeNoNestedParallelism, openmpDeviceVersion));
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Get the Target attribute on the current module if it exists
        and return the attribute, if it doesn't exist it returns a nullptr.
      }],
      /*retTy=*/"mlir::omp::TargetAttr",
      /*methodName=*/"getTarget",
      (ins), [{}], [{
        if (Attribute flags = $_op->getAttr("omp.target"))
          return ::llvm::dyn_cast_or_null<mlir::omp::TargetAttr>(flags);
        return nullptr;
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Set the attribute target on the current module with the
        specified string arguments - name of cpu and corresponding features.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setTarget",
      (ins "llvm::StringRef":$targetCPU,
           "llvm::StringRef":$targetFeatures), [{}], [{
        if (targetCPU.empty())
          return;
        $_op->setAttr(("omp." + mlir::omp::TargetAttr::getMnemonic()).str(),
                  mlir::omp::TargetAttr::get($_op->getContext(),
                                             targetCPU.str(),
                                             targetFeatures.str()));
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Set a StringAttr on the current module containing the host IR file path. This
        file path is used in two-phase compilation during the device phase to generate
        device side LLVM IR when lowering MLIR.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setHostIRFilePath",
      (ins "std::string":$hostIRFilePath), [{}], [{
        $_op->setAttr(
          mlir::StringAttr::get($_op->getContext(), llvm::Twine{"omp.host_ir_filepath"}),
            mlir::StringAttr::get($_op->getContext(), hostIRFilePath));
       }]>,
    InterfaceMethod<
      /*description=*/[{
        Find the host-ir file path StringAttr from the current module if it exists and
        return its contained value, if it doesn't exist it returns an empty string. This
        file path is used in two-phase compilation during the device phase to generate
        device side LLVM IR when lowering MLIR.
      }],
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getHostIRFilePath",
      (ins), [{}], [{
        if (Attribute filepath = $_op->getAttr("omp.host_ir_filepath"))
          if (::llvm::isa<mlir::StringAttr>(filepath))
            return ::llvm::dyn_cast<mlir::StringAttr>(filepath).getValue();
        return {};
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Get the omp.requires attribute on the operator if it's present and
        return its value. If it doesn't exist, return `ClauseRequires::none` by
        default.
      }],
      /*retTy=*/"::mlir::omp::ClauseRequires",
      /*methodName=*/"getRequires",
      (ins), [{}], [{
        if (Attribute requiresAttr = $_op->getAttr("omp.requires"))
          if (auto requiresVal = requiresAttr.dyn_cast<mlir::omp::ClauseRequiresAttr>())
            return requiresVal.getValue();
        return mlir::omp::ClauseRequires::none;
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Set the omp.requires attribute on the operator to the specified clauses.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setRequires",
      (ins "::mlir::omp::ClauseRequires":$clauses), [{}], [{
        $_op->setAttr(mlir::StringAttr::get($_op->getContext(), "omp.requires"),
          mlir::omp::ClauseRequiresAttr::get($_op->getContext(), clauses));
      }]>
  ];
}

def EarlyOutliningInterface : OpInterface<"EarlyOutliningInterface"> {
  let description = [{
    FuncOps that are a result of early outlining should have this interface.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Set a StringAttr on an outlined target region function containing the
        name of the parent function where the target region was outlined
        from. The parent name is used to construct the kernel names for target
        regions.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setParentName",
      (ins "std::string":$parentName), [{}], [{
        $_op->setAttr(
          mlir::StringAttr::get($_op->getContext(),
                                llvm::Twine{"omp.outline_parent_name"}),
          mlir::StringAttr::get($_op->getContext(), parentName));
       }]>,

    InterfaceMethod<
      /*description=*/[{
        Returns the parent function name from where the target op was outlined
        from. If it doesn't exist it returns an empty string.
      }],
      /*retTy=*/"llvm::StringRef",
      /*methodName=*/"getParentName",
      (ins), [{}], [{
        if (Attribute parentName = $_op->getAttr("omp.outline_parent_name"))
          if (::llvm::isa<mlir::StringAttr>(parentName))
            return ::llvm::dyn_cast<mlir::StringAttr>(parentName).getValue();
        return {};
      }]>
  ];
}

#endif // OpenMP_OPS_INTERFACES
