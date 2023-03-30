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

def OffloadModuleInterface : OpInterface<"OffloadModuleInterface"> {
  let description = [{
    Operations that represent a module for offloading (host or device) 
    should have this interface.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Set the attribute IsDeviceAttr on the current module with the 
        specified boolean argument. 
      }],
      /*retTy=*/"void",
      /*methodName=*/"setIsDevice", 
      (ins "bool":$isDevice), [{}], [{
        $_op->setAttr(
          mlir::StringAttr::get($_op->getContext(), llvm::Twine{"omp.is_device"}),
            mlir::omp::IsDeviceAttr::get($_op->getContext(), isDevice));
      }]>,
      InterfaceMethod<
      /*description=*/[{
        Get the IsDeviceAttr attribute on the current module if it exists and return
        its value, if it doesn't exit it returns false by default.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getIsDevice", 
      (ins), [{}], [{
        if (Attribute isDevice = $_op->getAttr("omp.is_device"))
          if (isDevice.isa<mlir::omp::IsDeviceAttr>())
            return isDevice.dyn_cast<IsDeviceAttr>().getIsDevice();
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
          return flags.dyn_cast_or_null<mlir::omp::FlagsAttr>();
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
            "bool":$assumeNoNestedParallelism), [{}], [{
        $_op->setAttr(("omp." + mlir::omp::FlagsAttr::getMnemonic()).str(),
                  mlir::omp::FlagsAttr::get($_op->getContext(), debugKind,
                      assumeTeamsOversubscription, assumeThreadsOversubscription, 
                      assumeNoThreadState, assumeNoNestedParallelism));
      }]>,
  ];
}

#endif // OpenMP_OPS_INTERFACES
