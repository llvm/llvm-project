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

#ifndef OPENMP_OPS_INTERFACES
#define OPENMP_OPS_INTERFACES

include "mlir/IR/OpBase.td"

def BlockArgOpenMPOpInterface : OpInterface<"BlockArgOpenMPOpInterface"> {
  let description = [{
    OpenMP operations that define entry block arguments as part of the
    representation of its clauses.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    // Default-implemented methods to be overriden by the corresponding clauses.
    InterfaceMethod<"Get number of block arguments defined by `host_eval`.",
                    "unsigned", "numHostEvalBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `in_reduction`.",
                    "unsigned", "numInReductionBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `map`.",
                    "unsigned", "numMapBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `private`.",
                    "unsigned", "numPrivateBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `reduction`.",
                    "unsigned", "numReductionBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `task_reduction`.",
                    "unsigned", "numTaskReductionBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `use_device_addr`.",
                    "unsigned", "numUseDeviceAddrBlockArgs", (ins), [{}], [{
      return 0;
    }]>,
    InterfaceMethod<"Get number of block arguments defined by `use_device_ptr`.",
                    "unsigned", "numUseDevicePtrBlockArgs", (ins), [{}], [{
      return 0;
    }]>,

    // Unified access methods for start indices of clause-associated entry block
    // arguments.
    InterfaceMethod<"Get start index of block arguments defined by `host_eval`.",
                    "unsigned", "getHostEvalBlockArgsStart", (ins), [{
      return 0;
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `in_reduction`.",
                    "unsigned", "getInReductionBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getHostEvalBlockArgsStart() + $_op.numHostEvalBlockArgs();
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `map`.",
                    "unsigned", "getMapBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getInReductionBlockArgsStart() +
             $_op.numInReductionBlockArgs();
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `private`.",
                    "unsigned", "getPrivateBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getMapBlockArgsStart() + $_op.numMapBlockArgs();
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `reduction`.",
                    "unsigned", "getReductionBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getPrivateBlockArgsStart() + $_op.numPrivateBlockArgs();
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `task_reduction`.",
                    "unsigned", "getTaskReductionBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getReductionBlockArgsStart() + $_op.numReductionBlockArgs();
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `use_device_addr`.",
                    "unsigned", "getUseDeviceAddrBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getTaskReductionBlockArgsStart() + $_op.numTaskReductionBlockArgs();
    }]>,
    InterfaceMethod<"Get start index of block arguments defined by `use_device_ptr`.",
                    "unsigned", "getUseDevicePtrBlockArgsStart", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return iface.getUseDeviceAddrBlockArgsStart() + $_op.numUseDeviceAddrBlockArgs();
    }]>,

    // Unified access methods for clause-associated entry block arguments.
    InterfaceMethod<"Get block arguments defined by `host_eval`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getHostEvalBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getHostEvalBlockArgsStart(), $_op.numHostEvalBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `in_reduction`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getInReductionBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getInReductionBlockArgsStart(), $_op.numInReductionBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `map`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getMapBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getMapBlockArgsStart(), $_op.numMapBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `private`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getPrivateBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getPrivateBlockArgsStart(), $_op.numPrivateBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `reduction`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getReductionBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getReductionBlockArgsStart(), $_op.numReductionBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `task_reduction`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getTaskReductionBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getTaskReductionBlockArgsStart(),
          $_op.numTaskReductionBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `use_device_addr`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getUseDeviceAddrBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getUseDeviceAddrBlockArgsStart(),
          $_op.numUseDeviceAddrBlockArgs());
    }]>,
    InterfaceMethod<"Get block arguments defined by `use_device_ptr`.",
                    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
                    "getUseDevicePtrBlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
          iface.getUseDevicePtrBlockArgsStart(),
          $_op.numUseDevicePtrBlockArgs());
    }]>,
  ];

  let verify = [{
    auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>($_op);
    unsigned expectedArgs = iface.numHostEvalBlockArgs() +
        iface.numInReductionBlockArgs() + iface.numMapBlockArgs() +
        iface.numPrivateBlockArgs() + iface.numReductionBlockArgs() +
        iface.numTaskReductionBlockArgs() + iface.numUseDeviceAddrBlockArgs() +
        iface.numUseDevicePtrBlockArgs();
    if ($_op->getRegion(0).getNumArguments() < expectedArgs)
      return $_op->emitOpError() << "expected at least " << expectedArgs
                                 << " entry block argument(s)";
    return ::mlir::success();
  }];
}

def OutlineableOpenMPOpInterface : OpInterface<"OutlineableOpenMPOpInterface"> {
  let description = [{
    OpenMP operations whose region will be outlined will implement this
    interface.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<"Get alloca block", "::mlir::Block*", "getAllocaBlock",
      (ins), [{
      return &$_op.getRegion().front();
      }]>,
  ];
}

def MapClauseOwningOpInterface : OpInterface<"MapClauseOwningOpInterface"> {
  let description = [{
    OpenMP operations which own a list of omp::MapInfoOp's implement this interface
    to allow generic access to deal with map operands to more easily manipulate
    this class of operations.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<"Get map operands", "::mlir::OperandRange", "getMapVars",
      (ins), [{
        return $_op.getMapVars();
      }]>,
      InterfaceMethod<"Get mutable map operands", "::mlir::MutableOperandRange",
                      "getMapVarsMutable",
      (ins), [{
        return $_op.getMapVarsMutable();
      }]>,
      InterfaceMethod<"Get operand index for a map clause",
                      "int64_t",
                      "getOperandIndexForMap",
      (ins "::mlir::Value":$map), [{
         return std::distance($_op.getMapVars().begin(),
                              llvm::find($_op.getMapVars(), map));
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

def LoopWrapperInterface : OpInterface<"LoopWrapperInterface"> {
  let description = [{
    OpenMP operations that wrap a single loop nest. They must only contain a
    single region with a single block in which there's a single operation and a
    terminator. That nested operation must be another loop wrapper or an
    `omp.loop_nest`.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      /*description=*/[{
        If there is another loop wrapper immediately nested inside, return that
        operation. Assumes this operation is a valid loop wrapper.
      }],
      /*retTy=*/"::mlir::omp::LoopWrapperInterface",
      /*methodName=*/"getNestedWrapper",
      (ins), [{}], [{
        Operation *nested = &*$_op->getRegion(0).op_begin();
        return ::llvm::dyn_cast<LoopWrapperInterface>(nested);
      }]
    >,
    InterfaceMethod<
      /*description=*/[{
        Return the loop nest nested directly or indirectly inside of this loop
        wrapper. Assumes this operation is a valid loop wrapper.
      }],
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"getWrappedLoop",
      (ins), [{}], [{
        if (LoopWrapperInterface nested = $_op.getNestedWrapper())
          return nested.getWrappedLoop();
        return &*$_op->getRegion(0).op_begin();
      }]
    >
  ];

  let extraClassDeclaration = [{
    /// Interface verifier imlementation.
    llvm::LogicalResult verifyImpl();
  }];

  let verify = [{
    return ::llvm::cast<::mlir::omp::LoopWrapperInterface>($_op).verifyImpl();
  }];
  let verifyWithRegions = 1;
}

def ComposableOpInterface : OpInterface<"ComposableOpInterface"> {
  let description = [{
    OpenMP operations that can represent a single leaf of a composite OpenMP
    construct.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [
    InterfaceMethod<
      /*description=*/[{
        Check whether the operation is representing a leaf of a composite OpenMP
        construct.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"isComposite",
      (ins ), [{}], [{
        return $_op->hasAttr("omp.composite");
      }]
    >,
    InterfaceMethod<
      /*description=*/[{
        Mark the operation as part of an OpenMP composite construct.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setComposite",
      (ins "bool":$val), [{}], [{
        if (val)
          $_op->setDiscardableAttr("omp.composite", mlir::UnitAttr::get($_op->getContext()));
        else
          $_op->removeDiscardableAttr("omp.composite");
      }]
    >
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
          if (::llvm::isa<mlir::BoolAttr>(isTargetDevice))
           return ::llvm::dyn_cast<BoolAttr>(isTargetDevice).getValue();
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
          if (auto isTargetCGVal = ::llvm::dyn_cast<BoolAttr>(isTargetCGAttr))
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
            "uint32_t":$openmpDeviceVersion,
            "bool":$noGPULib), [{}], [{
        $_op->setAttr(("omp." + mlir::omp::FlagsAttr::getMnemonic()).str(),
                  mlir::omp::FlagsAttr::get($_op->getContext(), debugKind,
                      assumeTeamsOversubscription, assumeThreadsOversubscription,
                      assumeNoThreadState, assumeNoNestedParallelism, noGPULib, openmpDeviceVersion));
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
          if (auto requiresVal = ::llvm::dyn_cast<mlir::omp::ClauseRequiresAttr>(requiresAttr))
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
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Get the omp.target_triples attribute on the operator if it's present and
        return its value. If it doesn't exist, return an empty array by default.
      }],
      /*retTy=*/"::llvm::ArrayRef<::mlir::Attribute>",
      /*methodName=*/"getTargetTriples",
      (ins), [{}], [{
        if (Attribute triplesAttr = $_op->getAttr("omp.target_triples"))
          if (auto triples = ::llvm::dyn_cast<::mlir::ArrayAttr>(triplesAttr))
            return triples.getValue();
        return {};
      }]>,
    InterfaceMethod<
      /*description=*/[{
        Set the omp.target_triples attribute on the operation.
      }],
      /*retTy=*/"void",
      /*methodName=*/"setTargetTriples",
      (ins "::llvm::ArrayRef<::std::string>":$targetTriples), [{}], [{
        auto names = ::llvm::to_vector(::llvm::map_range(
            targetTriples, [&](::std::string str) -> ::mlir::Attribute {
              return mlir::StringAttr::get($_op->getContext(), str);
            }));
        $_op->setAttr(
            ::mlir::StringAttr::get($_op->getContext(), "omp.target_triples"),
            ::mlir::ArrayAttr::get($_op->getContext(), names));
      }]>
  ];
}

#endif // OPENMP_OPS_INTERFACES
