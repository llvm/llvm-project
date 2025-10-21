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


// Internal class to hold definitions of BlockArgOpenMPOpInterface methods,
// based on the name of the clause and what clause comes earlier in the list.
//
// The clause order will define the expected relative order between block
// arguments corresponding to each of these clauses.
class BlockArgOpenMPClause<string clauseNameSnake, string clauseNameCamel,
    BlockArgOpenMPClause previousClause> {
  // Default-implemented method, overriden by the corresponding clause. It
  // returns the range of operands passed to the operation associated to the
  // clause.
  //
  // For the override to work, the clause tablegen definition must contain a
  // `Variadic<...> $clause_name_vars` argument.
  //
  // Usage example:
  //
  // ```c++
  // OperandRange reductionVars = op.getReductionVars();
  // ```
  InterfaceMethod varsMethod = InterfaceMethod<
    "Get operation operands associated to `" # clauseNameSnake # "`.",
    "::mlir::OperandRange", "get" # clauseNameCamel # "Vars", (ins), [{}], [{
      return {0, 0};
    }]
  >;

  // It returns the number of entry block arguments introduced by the given
  // clause.
  //
  // By default, it will be the number of operands corresponding to that clause,
  // but it can be overriden by operations where this might not be the case
  // (e.g. `map` clause in `omp.target_update`).
  //
  // Usage example:
  //
  // ```c++
  // unsigned numInReductionArgs = op.numInReductionBlockArgs();
  // ```
  InterfaceMethod numArgsMethod = InterfaceMethod<
    "Get number of block arguments defined by `" # clauseNameSnake # "`.",
    "unsigned", "num" # clauseNameCamel # "BlockArgs", (ins), [{}],
    "return $_op." # varsMethod.name # "().size();"
  >;

  // Unified access method for the start index of clause-associated entry block
  // arguments.
  //
  // Usage example:
  //
  // ```c++
  // auto iface = cast<BlockArgOpenMPOpInterface>(op);
  // unsigned firstMapIndex = iface.getMapBlockArgsStart();
  // ```
  InterfaceMethod startMethod = InterfaceMethod<
    "Get start index of block arguments defined by `" # clauseNameSnake # "`.",
    "unsigned", "get" # clauseNameCamel # "BlockArgsStart", (ins),
    !if(!initialized(previousClause), [{
        auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      }] # "return iface." # previousClause.startMethod.name # "() + iface."
        # previousClause.numArgsMethod.name # "();",
        "return 0;"
    )
  >;

  // Unified access method for clause-associated entry block arguments.
  //
  // Usage example:
  //
  // ```c++
  // auto iface = cast<BlockArgOpenMPOpInterface>(op);
  // ArrayRef<BlockArgument> reductionArgs = iface.getReductionBlockArgs();
  // ```
  InterfaceMethod blockArgsMethod = InterfaceMethod<
    "Get block arguments defined by `" # clauseNameSnake # "`.",
    "::llvm::MutableArrayRef<::mlir::BlockArgument>",
    "get" # clauseNameCamel # "BlockArgs", (ins), [{
      auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
      return $_op->getRegion(0).getArguments().slice(
    }] # "iface." # startMethod.name # "(), iface." # numArgsMethod.name # "());"
  >;
}

def BlockArgHasDeviceAddrClause : BlockArgOpenMPClause<
    "has_device_addr", "HasDeviceAddr", ?>;
def BlockArgHostEvalClause : BlockArgOpenMPClause<
    "host_eval", "HostEval", BlockArgHasDeviceAddrClause>;
def BlockArgInReductionClause : BlockArgOpenMPClause<
    "in_reduction", "InReduction", BlockArgHostEvalClause>;
def BlockArgMapClause : BlockArgOpenMPClause<
    "map", "Map", BlockArgInReductionClause>;
def BlockArgPrivateClause : BlockArgOpenMPClause<
    "private", "Private", BlockArgMapClause>;
def BlockArgReductionClause : BlockArgOpenMPClause<
    "reduction", "Reduction", BlockArgPrivateClause>;
def BlockArgTaskReductionClause : BlockArgOpenMPClause<
    "task_reduction", "TaskReduction", BlockArgReductionClause>;
def BlockArgUseDeviceAddrClause : BlockArgOpenMPClause<
    "use_device_addr", "UseDeviceAddr", BlockArgTaskReductionClause>;
def BlockArgUseDevicePtrClause : BlockArgOpenMPClause<
    "use_device_ptr", "UseDevicePtr", BlockArgUseDeviceAddrClause>;

def BlockArgOpenMPOpInterface : OpInterface<"BlockArgOpenMPOpInterface"> {
  let description = [{
    OpenMP operations that define entry block arguments as part of the
    representation of its clauses.
  }];

  let cppNamespace = "::mlir::omp";

  defvar clauses = [ BlockArgHasDeviceAddrClause, BlockArgHostEvalClause,
    BlockArgInReductionClause, BlockArgMapClause, BlockArgPrivateClause,
    BlockArgReductionClause, BlockArgTaskReductionClause,
    BlockArgUseDeviceAddrClause, BlockArgUseDevicePtrClause ];

  let methods = !listconcat(
    !foreach(clause, clauses, clause.varsMethod),
    !foreach(clause, clauses, clause.numArgsMethod),
    !foreach(clause, clauses, clause.startMethod),
    !foreach(clause, clauses, clause.blockArgsMethod),
    [
      InterfaceMethod<
        "Get the total number of clause-defined entry block arguments",
        "unsigned", "numClauseBlockArgs", (ins),
        "return " # !interleave(
          !foreach(clause, clauses, "$_op." # clause.numArgsMethod.name # "()"),
          " + ") # ";"
      >,
      InterfaceMethod<
        "Populate a vector of pairs representing the matching between operands "
        "and entry block arguments.", "void", "getBlockArgsPairs",
        (ins "::llvm::SmallVectorImpl<std::pair<::mlir::Value, ::mlir::BlockArgument>> &" : $pairs),
        [{
          auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>(*$_op);
          pairs.reserve(pairs.size() + iface.numClauseBlockArgs());
        }] # !interleave(!foreach(clause, clauses, [{
        }] # "if (iface." # clause.numArgsMethod.name # "() > 0) {" # [{
        }] # "  for (auto [var, arg] : ::llvm::zip_equal(" #
                "iface." # clause.varsMethod.name # "()," #
                "iface." # clause.blockArgsMethod.name # "()))" # [{
            pairs.emplace_back(var, arg);
        } }]), "\n")
      >
    ]
  );

  let verify = [{
    auto iface = ::llvm::cast<BlockArgOpenMPOpInterface>($_op);
    unsigned expectedArgs = iface.numClauseBlockArgs();
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

def LoopWrapperInterface : OpInterface<"LoopWrapperInterface"> {
  let description = [{
    OpenMP operations that wrap a single loop nest. They must only contain a
    single region with a single block in which there's a single operation and a
    terminator. That nested operation must be another loop wrapper or an
    `omp.loop_nest`.

    Operation-specific verifiers should make the following checks in their
    verifier, additionally to what the interface itself checks:
      - If `getNestedWrapper() != nullptr`, is the type of the nested wrapper
      allowed in that context? This check might require looking at the parent as
      well.
      - If the operation is a `ComposableOpInterface`, check that it is
      consistent with the potential existence of a `LoopWrapperInterface` parent
      and whether `getNestedWrapper() != nullptr`.
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
    /// Interface verifier implementation.
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
           "mlir::omp::DeclareTargetCaptureClause":$captureClause,
           "bool":$automap), [{}], [{
        $_op->setAttr("omp.declare_target",
                  mlir::omp::DeclareTargetAttr::get(
                      $_op->getContext(),
                      mlir::omp::DeclareTargetDeviceTypeAttr::get(
                          $_op->getContext(), deviceType),
                      mlir::omp::DeclareTargetCaptureClauseAttr::get(
                          $_op->getContext(), captureClause),
                      mlir::BoolAttr::get($_op->getContext(), automap)));
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
      }]>,
      InterfaceMethod<
      /*description=*/[{
        Return true if the DeclareTarget attribute has the AUTOMAP modifier.
      }],
      /*retTy=*/"bool",
      /*methodName=*/"getDeclareTargetAutomap",
      (ins), [{}], [{
        if (mlir::Attribute dTar = $_op->getAttr("omp.declare_target"))
          if (auto dAttr = llvm::dyn_cast_or_null<mlir::omp::DeclareTargetAttr>(dTar))
            if (auto autoVal = dAttr.getAutomap())
              return autoVal.getValue();
         return false;
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

def LoopTransformationInterface : OpInterface<"LoopTransformationInterface"> {
  let description = [{
    Methods that are common for OpenMP loop transformation operations.
  }];

  let cppNamespace = "::mlir::omp";

  let methods = [

    InterfaceMethod<
      /*description=*/[{
        Get the indices for the arguments that represent CanonicalLoopInfo
        applyees, i.e. loops that are transformed/consumed by this operation.
      }],
      /*returnType=*/ "std::pair<unsigned, unsigned>",
      /*methodName=*/ "getApplyeesODSOperandIndexAndLength",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*description=*/[{
        Get the indices for the arguments that represent CanonicalLoopInfo
        generatees, i.e. loops that are emitted by this operation.
      }],
      /*returnType=*/ "std::pair<unsigned, unsigned>",
      /*methodName=*/ "getGenerateesODSOperandIndexAndLength",
      /*args=*/(ins)
    >,

    InterfaceMethod<
      /*description=*/[{
        Return the number of applyees of this loop transformation.
      }],
      /*returnType=*/ "unsigned",
      /*methodName=*/ "getNumApplyees",
      /*args=*/       (ins),
      /*methodBody=*/ "",
      /*defaultImpl=*/[{
        return $_op.getApplyeesODSOperandIndexAndLength().second;
      }]
    >,

    InterfaceMethod<
      /*description=*/[{
        Return the number of generatees of this loop transformation.
      }],
      /*returnType=*/ "unsigned",
      /*methodName=*/ "getNumGeneratees",
      /*args=*/       (ins),
      /*methodBody=*/ "",
      /*defaultImpl=*/[{
        return $_op.getGenerateesODSOperandIndexAndLength().second;
      }]
    >,

    InterfaceMethod<
      /*description=*/[{
        Return whether the provided operand is an applyee of this operation.
      }],
      /*returnType=*/ "unsigned",
      /*methodName=*/ "isApplyee",
      /*args=*/       (ins "unsigned":$opnum),
      /*methodBody=*/ "",
      /*defaultImpl=*/[{
        auto applyeeArgs = $_op.getApplyeesODSOperandIndexAndLength();
        return (applyeeArgs.first <= opnum && opnum < applyeeArgs.first + applyeeArgs.second) ;
      }]
    >,

    InterfaceMethod<
      /*description=*/[{
        Return whether the provided operand is a generatee of this operation.
      }],
      /*returnType=*/ "unsigned",
      /*methodName=*/ "isGeneratee",
      /*args=*/       (ins "unsigned":$opnum),
      /*methodBody=*/ "",
      /*defaultImpl=*/[{
        auto generateeArgs = $_op.getGenerateesODSOperandIndexAndLength();
        return (generateeArgs.first <= opnum && opnum < generateeArgs.first + generateeArgs.second) ;
      }]
    >,

  ];
}

#endif // OPENMP_OPS_INTERFACES
