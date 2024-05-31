#ifndef MLIR_DIALECT_TRAITS_H
#define MLIR_DIALECT_TRAITS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Dialect.h.inc"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Types.h>

namespace mlir {
namespace obs {
namespace user_types {
    struct OwnTypeStorage;
    struct RefTypeStorage;
}//user_types
}//obs
}//mlir

#define GET_OP_CLASSES
#include "Ops.h.inc"



namespace mlir {
namespace obs {

class OwnType : public mlir::Type::TypeBase<OwnType, mlir::Type, user_types::OwnTypeStorage> {
public:
    using Base::Base;

    //Create an instance of `OwnType`.
    static OwnType get(mlir::MLIRContext *ctx, StringRef resName, ArrayRef<unsigned int> dims);

    //Return the owned resource name.
    StringRef getResName() ;

    //Return the  dims of the owned resource.
    ArrayRef<unsigned int> getDims();

    static constexpr llvm::StringLiteral name = "obs.OWN";
};

class RefType : public mlir::Type::TypeBase<RefType, mlir::Type, user_types::RefTypeStorage> {
public:
    using Base::Base;

    //Create an instance of `OwnType`.
    static RefType get(ArrayRef<mlir::Type> ownerType);

    //Return the owned resource name.
    ArrayRef<mlir::Type> getOwnerType() ;

    static constexpr llvm::StringLiteral name = "obs.REF";
};

} //obs
} //mlir


#endif //MLIR_DIALECT_TRAITS_H
