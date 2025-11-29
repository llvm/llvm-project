#ifndef MLIR_IR_QUANTSTORAGETYPEINTERFACE
#define MLIR_IR_QUANTSTORAGETYPEINTERFACE

include "mlir/IR/OpBase.td"

def QuantStorageTypeInterface : TypeInterface<"QuantStorageTypeInterface"> {
  let description = [{
    Interface for types that can be used as storage types in Quant dialect.
    This interface provides methods to determine storage characteristics for quantization purposes,
    including packing behavior, and alignment requirements.
  }];
  let cppNamespace = "::mlir";

  let methods = [
    InterfaceMethod<[{
      Check if the storage type is signed.
      Returns true if the type represents signed values, false for unsigned.
    }],
    "bool", "isStorageSigned", (ins)>,

    InterfaceMethod<[{
      Get the bit width of this type.
      Returns the number of bits used to store values of this type.
    }],
    "unsigned", "getStorageWidth", (ins)>,

    InterfaceMethod<[{
      Get default minimum value for this type.
    }],
    "int64_t", "getDefaultMinimum", (ins)>,

    InterfaceMethod<[{
      Get default maximum value for this type.
    }],
    "int64_t", "getDefaultMaximum", (ins)>,

    InterfaceMethod<[{
      Get the storage type as a string.
    }],
    "std::string", "getStorageType", (ins)>,

    InterfaceMethod<[{
      Check if the storage type uses packed representation.
      Returns true if multiple values are packed into one byte (e.g., sub-byte types),
      false if value uses full byte.
    }],
    "bool", "isPacked", (ins)>,

    InterfaceMethod<[{
      Get the logical bit width per value.
      For packed sub-byte types, this may differ from getStorageWidth().
    }],
    "unsigned", "getLogicalBitWidth", (ins)>,

    InterfaceMethod<[{
      Get the number of logical elements that fit in one byte.
      For packed sub-byte types, this returns how many values can be stored per byte.
    }],
    "unsigned", "getElementsPerByte", (ins)>,

    InterfaceMethod<[{
      Returns the preferred alignment for this type, in bytes.
    }],
    "std::optional<unsigned>", "getPreferredAlignmentBytes", (ins)>
  ];

}

#endif // MLIR_IR_QUANTSTORAGETYPEINTERFACE
