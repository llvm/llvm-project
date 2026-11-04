#ifndef MLIR_INTERFACES_DIALECTFOLDINTERFACE
#define MLIR_INTERFACES_DIALECTFOLDINTERFACE

include "mlir/IR/Interfaces.td"

def DialectFoldInterface : DialectInterface<"DialectFoldInterface"> {
  let description = [{
    Define a fold interface to allow for dialects to control specific aspects
    of the folding behavior for operations they define.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
        Registered fallback fold for the dialect. Like the fold hook of each
        operation, it attempts to fold the operation with the specified constant
        operand values - the elements in "operands" will correspond directly to
        the operands of the operation, but may be null if non-constant.  If
        folding is successful, this fills in the `results` vector.  If not, this
        returns failure and `results` is unspecified.
      }],
      "::mlir::LogicalResult", "fold",
      (ins "::mlir::Operation *":$op, "::mlir::ArrayRef<Attribute>":$operands,
           "::mlir::SmallVectorImpl<OpFoldResult> &":$results),
      [{
        return failure();
      }]
    >,
    InterfaceMethod<[{
        Registered hook to check if the given region, which is attached to an
        operation that is *not* isolated from above, should be used when
        materializing constants. The folder will generally materialize constants
        into the top-level isolated region, this allows for materializing into a
        lower level ancestor region if it is more profitable/correct.
      }],
      "bool", "shouldMaterializeInto",
      (ins "::mlir::Region *":$region),
      [{
        return false;
      }]
    > 
  ];
}


#endif
