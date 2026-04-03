#ifndef MLIR_INTERFACES_CONVERTTOEMITCPATTERNINTERFACE
#define MLIR_INTERFACES_CONVERTTOEMITCPATTERNINTERFACE

include "mlir/IR/Interfaces.td"

def ConvertToEmitCPatternInterface : DialectInterface<"ConvertToEmitCPatternInterface"> {
  let description = [{}];
  let cppNamespace = "::mlir";

  let methods = [
    PureVirtualInterfaceMethod<[{
        Hook for derived dialect interface to provide conversion patterns
        and mark dialect legal for the conversion target.
      }],
      "void", "populateConvertToEmitCConversionPatterns",
      (ins "::mlir::ConversionTarget &":$target, "::mlir::TypeConverter &":$typeConverter,
           "::mlir::RewritePatternSet &":$patterns)
    >
  ];
}

#endif
