#ifndef MLIR_IR_QUANTSTORAGETYPEINTERFACE
#define MLIR_IR_QUANTSTORAGETYPEINTERFACE

include "mlir/IR/OpBase.td"

def QuantStorageTypeInterface : TypeInterface<"QuantStorageTypeInterface"> {
  let description = [{
    Interface for types that can be used as storage types in Quant dialect.
    This interface provides methods to determine storage characteristics for quantization purposes.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Check if the storage type is signed.
      Returns true if the type represents signed values, false for unsigned.
    }],
    "bool", "isStorageSigned", (ins)>,

    InterfaceMethod<[{
      Get the bit width of this integer type.
      Returns the number of bits used to store values of this type.
    }],
    "unsigned", "getStorageWidth", (ins)>,

    InterfaceMethod<[{
      Get default minimum value for this integer type.
    }],
    "int64_t", "getDefaultMinimum", (ins)>,

    InterfaceMethod<[{
      Get default maximum value for this integer type.
    }],
    "int64_t", "getDefaultMaximum", (ins)>,

    InterfaceMethod<[{
      Get the storage type as a string.
    }],
    "std::string", "getStorageType", (ins)>
  ];

}

#endif // MLIR_IR_QUANTSTORAGETYPEINTERFACE
