//===- SPIRVAvailability.td - Op Availability Base file ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_AVAILABILITY
#define MLIR_DIALECT_SPIRV_IR_AVAILABILITY

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Op availability definitions
//===----------------------------------------------------------------------===//

// The base class for defining op availability dimensions.
class Availability {
  // The following are fields for controlling the generated C++ OpInterface.

  // The namespace for the generated C++ OpInterface subclass.
  string cppNamespace = ?;
  // The name for the generated C++ OpInterface subclass.
  string interfaceName = ?;
  // The documentation for the generated C++ OpInterface subclass.
  string interfaceDescription = "";

  // The following are fields for controlling the query function signature.

  // The query function's return type in the generated C++ OpInterface subclass.
  string queryFnRetType = ?;
  // The query function's name in the generated C++ OpInterface subclass.
  string queryFnName = ?;

  // The following are fields for controlling the query function implementation.

  // The logic for merging two availability requirements. This is used to derive
  // the final availability requirement when, for example, an op has two
  // operands and these two operands have different availability requirements.
  //
  // The code should use `$overall` as the placeholder for the final requirement
  // and `$instance` for the current availability requirement instance.
  code mergeAction = ?;
  // The initializer for the final availability requirement.
  string initializer = ?;
  // An availability instance's type.
  string instanceType = ?;

  // The following are fields for a concrete availability instance.

  // The code for preparing a concrete instance. This should be C++ statements
  // and will be generated before the `mergeAction` logic.
  code instancePreparation = "";
  // The availability requirement carried by a concrete instance.
  string instance = ?;
}

class MinVersionBase<string name, EnumAttr scheme, I32EnumAttrCase min>
    : Availability {
  let interfaceName = name;

  let queryFnRetType = "llvm::Optional<" # scheme.returnType # ">";
  let queryFnName = "getMinVersion";

  let mergeAction = "{ "
    "if ($overall) { "
      "$overall = static_cast<" # scheme.returnType # ">("
                  "std::max(*$overall, $instance)); "
    "} else { $overall = $instance; }}";
  let initializer = "::llvm::None";
  let instanceType = scheme.cppNamespace # "::" # scheme.enum.className;

  let instance = scheme.cppNamespace # "::" # scheme.enum.className # "::" #
                 min.symbol;
}

class MaxVersionBase<string name, EnumAttr scheme, I32EnumAttrCase max>
    : Availability {
  let interfaceName = name;

  let queryFnRetType = "llvm::Optional<" # scheme.returnType # ">";
  let queryFnName = "getMaxVersion";

  let mergeAction = "{ "
    "if ($overall) { "
      "$overall = static_cast<" # scheme.returnType # ">("
                  "std::min(*$overall, $instance)); "
    "} else { $overall = $instance; }}";
  let initializer = "::llvm::None";
  let instanceType = scheme.cppNamespace # "::" # scheme.enum.className;

  let instance = scheme.cppNamespace # "::" # scheme.enum.className # "::" #
                 max.symbol;
}

#endif // MLIR_DIALECT_SPIRV_IR_AVAILABILITY
