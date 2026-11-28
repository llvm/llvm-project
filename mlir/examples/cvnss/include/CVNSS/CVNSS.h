// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once
#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "CVNSS/CVNSSOpsDialect.h.inc"
#define GET_OP_CLASSES
#include "CVNSS/CVNSSOps.h.inc"
namespace mlir::cvnss { void registerCVNSSPasses(); }
