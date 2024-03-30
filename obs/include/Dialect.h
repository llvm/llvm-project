#ifndef MLIR_DIALECT_TRAITS_H
#define MLIR_DIALECT_TRAITS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect.h.inc"

#define GET_OP_CLASSES
#include "Ops.h.inc"


#endif //MLIR_DIALECT_TRAITS_H
