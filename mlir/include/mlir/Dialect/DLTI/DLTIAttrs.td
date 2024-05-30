//===- DLTIAttrs.td - DLTI dialect attributes definition --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DLTI_DLTIATTRS_TD
#define MLIR_DIALECT_DLTI_DLTIATTRS_TD

include "mlir/Dialect/DLTI/DLTI.td"
include "mlir/IR/AttrTypeBase.td"

class DLTIAttr<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<DLTI_Dialect, name, traits, baseCppClass> { }

//===----------------------------------------------------------------------===//
// DataLayoutEntryAttr
//===----------------------------------------------------------------------===//

def DataLayoutEntryTrait
    : NativeAttrTrait<"DataLayoutEntryInterface::Trait"> {
  let cppNamespace = "::mlir";
}

def DLTI_DataLayoutEntryAttr :
    DLTIAttr<"DataLayoutEntry", [DataLayoutEntryTrait]> {
  let summary = [{
    An attribute to represent an entry of a data layout specification.
  }];
  let description = [{
    A data layout entry attribute is a key-value pair where the key is a type or
    an identifier and the value is another attribute. These entries form a data
    layout specification.
  }];
  let parameters = (ins
    "DataLayoutEntryKey":$key, "Attribute":$value
  );
  // TODO: We do not generate storage class because llvm::PointerUnion
  // does not work with hash_key method.
  let genStorageClass = 0;
  let mnemonic = "dl_entry";
  let genVerifyDecl = 0;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Returns the entry with the given key and value.
    static DataLayoutEntryAttr get(StringAttr key, Attribute value);
    static DataLayoutEntryAttr get(MLIRContext *context, Type key, Attribute value);
    static DataLayoutEntryAttr get(Type key, Attribute value);
  }];
}

//===----------------------------------------------------------------------===//
// DataLayoutSpecAttr
//===----------------------------------------------------------------------===//
def DataLayoutSpecTrait
    : NativeAttrTrait<"DataLayoutSpecInterface::Trait"> {
  let cppNamespace = "::mlir";
}

def DLTI_DataLayoutSpecAttr :
    DLTIAttr<"DataLayoutSpec", [DataLayoutSpecTrait]> {
  let summary = [{
    An attribute to represent a data layout specification.
  }];
  let description = [{
    A data layout specification is a list of entries that specify (partial) data
    layout information. It is expected to be attached to operations that serve
    as scopes for data layout requests.
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "">:$entries
  );
  let mnemonic = "dl_spec";
  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
  let extraClassDeclaration = [{
    /// Combines this specification with `specs`, enclosing specifications listed
    /// from outermost to innermost. This overwrites the older entries with the
    /// same key as the newer entries if the entries are compatible. Returns null
    /// if the specifications are not compatible.
    DataLayoutSpecAttr combineWith(ArrayRef<DataLayoutSpecInterface> specs) const;
  
    /// Returns the endiannes identifier.
    StringAttr getEndiannessIdentifier(MLIRContext *context) const;

    /// Returns the alloca memory space identifier.
    StringAttr getAllocaMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the program memory space identifier.
    StringAttr getProgramMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the global memory space identifier.
    StringAttr getGlobalMemorySpaceIdentifier(MLIRContext *context) const;

    /// Returns the stack alignment identifier.
    StringAttr getStackAlignmentIdentifier(MLIRContext *context) const;
  }];
}

//===----------------------------------------------------------------------===//
// TargetSystemSpecAttr
//===----------------------------------------------------------------------===//

def TargetSystemSpecTrait
    : NativeAttrTrait<"TargetSystemSpecInterface::Trait"> {
  let cppNamespace = "::mlir";
}

def DLTI_TargetSystemSpecAttr :
    DLTIAttr<"TargetSystemSpec", [TargetSystemSpecTrait]> {
  let summary = [{
    An attribute to represent target system specification.
  }];
  let description = [{
    A system specification describes the overall system containing
    multiple devices, with each device having a unique ID
    and its corresponding TargetDeviceSpec object.

    Example:
          dlti.target_system_spec =
            #dlti.target_device_spec<
              #dlti.dl_entry<"dlti.device_id", 0: ui32>,
              #dlti.dl_entry<"dlti.device_type", "CPU">>,
            #dlti.target_device_spec<
              #dlti.dl_entry<"dlti.device_id", 1: ui32>,
              #dlti.dl_entry<"dlti.device_type", "GPU">,
              #dlti.dl_entry<"dlti.max_vector_op_width", 64 : ui32>>,
            #dlti.target_device_spec<
              #dlti.dl_entry<"dlti.device_id", 2: ui32>,
              #dlti.dl_entry<"dlti.device_type", "XPU">>>
  }];
  let parameters = (ins
    ArrayRefParameter<"TargetDeviceSpecInterface", "">:$entries
  );
  let mnemonic = "target_system_spec";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $entries `>`";
  let extraClassDeclaration = [{
    /// Return the device specification that matches the given device ID
    std::optional<TargetDeviceSpecInterface>
    getDeviceSpecForDeviceID(
      TargetDeviceSpecInterface::DeviceID deviceID);
  }];
  let extraClassDefinition = [{
    std::optional<TargetDeviceSpecInterface>
    $cppClass::getDeviceSpecForDeviceID(
        TargetDeviceSpecInterface::DeviceID deviceID) {
      for (TargetDeviceSpecInterface entry : getEntries()) {
        if (entry.getDeviceID() == deviceID)
          return entry;
      }
      return std::nullopt;
    }
  }];
}

//===----------------------------------------------------------------------===//
// TargetDeviceSpecAttr
//===----------------------------------------------------------------------===//

def TargetDeviceSpecTrait
    : NativeAttrTrait<"TargetDeviceSpecInterface::Trait"> {
  let cppNamespace = "::mlir";
}

def DLTI_TargetDeviceSpecAttr :
    DLTIAttr<"TargetDeviceSpec", [TargetDeviceSpecTrait]> {
  let summary = [{
    An attribute to represent target device specification.
  }];
  let description = [{
    Each device specification describes a single device and its
    hardware properties. Each device specification must have a device_id
    and a device_type. In addition, the specification can contain any number
    of optional hardware properties (e.g., max_vector_op_width below).

    Example:
      #dlti.target_device_spec<
            #dlti.dl_entry<"dlti.device_id", 1: ui32>,
            #dlti.dl_entry<"dlti.device_type", "GPU">,
            #dlti.dl_entry<"dlti.max_vector_op_width", 64 : ui32>>
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "">:$entries
  );
  let mnemonic = "target_device_spec";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $entries `>`";
  let extraClassDeclaration = [{
    /// Returns the device ID identifier.
    StringAttr getDeviceIDIdentifier();

    /// Returns the device type identifier.
    StringAttr getDeviceTypeIdentifier();

    /// Returns max vector op width identifier.
    StringAttr getMaxVectorOpWidthIdentifier();

    /// Returns L1 cache size identifier
    StringAttr getL1CacheSizeInBytesIdentifier();

    /// Returns the interface spec for device ID
    /// Since we verify that the spec contains device ID the function
    /// will return a valid spec.
    DataLayoutEntryInterface getSpecForDeviceID();

    /// Returns the interface spec for device type
    /// Since we verify that the spec contains device type the function
    /// will return a valid spec.
    DataLayoutEntryInterface getSpecForDeviceType();

    /// Returns the interface spec for max vector op width
    /// Since max vector op width is an optional property, this function will
    /// return a valid spec if the property is defined, otherwise it
    /// will return an empty spec.
    DataLayoutEntryInterface getSpecForMaxVectorOpWidth();

    /// Returns the interface spec for L1 cache size
    /// Since L1 cache size is an optional property, this function will
    /// return a valid spec if the property is defined, otherwise it
    /// will return an empty spec.
    DataLayoutEntryInterface getSpecForL1CacheSizeInBytes();

    /// Return the value of device ID
    TargetDeviceSpecInterface::DeviceID getDeviceID();
  }];
}

#endif  // MLIR_DIALECT_DLTI_DLTIATTRS_TD
