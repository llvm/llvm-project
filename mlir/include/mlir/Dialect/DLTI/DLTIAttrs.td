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
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

class DLTIAttr<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
    : AttrDef<DLTI_Dialect, name, traits, baseCppClass> { }

//===----------------------------------------------------------------------===//
// DataLayoutEntryAttr
//===----------------------------------------------------------------------===//

def DLTI_DataLayoutEntryAttr :
    DLTIAttr<"DataLayoutEntry", [DataLayoutEntryInterface]> {
  let summary = "An attribute to represent an entry of a data layout specification.";
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

def DLTI_DataLayoutSpecAttr :
    DLTIAttr<"DataLayoutSpec", [DataLayoutSpecInterface]> {
  let summary = "An attribute to represent a data layout specification.";
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

    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) {
      return llvm::cast<mlir::DataLayoutSpecInterface>(*this).queryHelper(key);
    }
  }];
}

def DLTI_MapAttr : DLTIAttr<"Map", [DLTIQueryInterface]> {
  let summary = "A mapping of DLTI-information by way of key-value pairs";
  let description = [{
    A Data Layout and Target Information map is a list of entries effectively
    encoding a dictionary, mapping DLTI-related keys to DLTI-related values.

    This attribute's main purpose is to facilate querying IR for arbitrary
    key-value associations that encode DLTI. Facility functions exist to perform
    recursive lookups on nested DLTI-map/query interface-implementing
    attributes.

    Consider the following flat encoding of a single-key dictionary
    ```
    #dlti.map<#dlti.dl_entry<"CPU::cache::L1::size_in_bytes", 65536 : i32>>
    ```
    versus nested maps, which make it possible to obtain sub-dictionaries of
    related information (with the following example making use of other
    attributes that also implement the `DLTIQueryInterface`):
    ```
    #dlti.target_system_spec<"CPU":
      #dlti.target_device_spec<#dlti.dl_entry<"cache",
        #dlti.map<#dlti.dl_entry<"L1",
          #dlti.map<#dlti.dl_entry<"size_in_bytes", 65536 : i32>>>,
                  #dlti.dl_entry<"L1d",
          #dlti.map<#dlti.dl_entry<"size_in_bytes", 32768 : i32>>> >>>>
    ```

    With the flat encoding, the implied structure of the key is ignored, that is
    the only successful query (as expressed in the Transform Dialect) is:
    `transform.dlti.query ["CPU::cache::L1::size_in_bytes"] at %op`,
    where `%op` is a handle to an operation which associates the flat-encoding
    `#dlti.map` attribute.

    For querying nested dictionaries, the relevant keys need to be separately
    provided. That is, if `%op` is an handle to an op which has the nesting
    `#dlti.target_system_spec`-attribute from above attached, then
    `transform.dlti.query ["CPU","cache","L1","size_in_bytes"] at %op` gives
    back the first leaf value contained. To access the other leaf, we need to do
    `transform.dlti.query ["CPU","cache","L1d","size_in_bytes"] at %op`.
    ```
   }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "">:$entries
  );
  let mnemonic = "map";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $entries `>`";
  let extraClassDeclaration = [{
    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) {
      for (DataLayoutEntryInterface entry : getEntries())
        if (entry.getKey() == key)
            return entry.getValue();
      return ::mlir::failure();
    }
  }];
}

//===----------------------------------------------------------------------===//
// TargetSystemSpecAttr
//===----------------------------------------------------------------------===//

def DLTI_TargetSystemSpecAttr :
    DLTIAttr<"TargetSystemSpec", [TargetSystemSpecInterface]> {
  let summary = "An attribute to represent target system specification.";
  let description = [{
    A system specification describes the overall system containing
    multiple devices, with each device having a unique ID (string)
    and its corresponding TargetDeviceSpec object.

    Example:
    ```
    dlti.target_system_spec =
     #dlti.target_system_spec<
      "CPU": #dlti.target_device_spec<
              #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096: ui32>>,
      "GPU": #dlti.target_device_spec<
              #dlti.dl_entry<"dlti.max_vector_op_width", 64 : ui32>>,
      "XPU": #dlti.target_device_spec<
              #dlti.dl_entry<"dlti.max_vector_op_width", 4096 : ui32>>>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"DeviceIDTargetDeviceSpecPair", "">:$entries
  );
  let mnemonic = "target_system_spec";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $entries `>`";
  let extraClassDeclaration = [{
    /// Return the device specification that matches the given device ID
    std::optional<TargetDeviceSpecInterface>
    getDeviceSpecForDeviceID(
      TargetSystemSpecInterface::DeviceID deviceID);

    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) const {
      return llvm::cast<mlir::TargetSystemSpecInterface>(*this).queryHelper(key);
    }
  }];
  let extraClassDefinition = [{
    std::optional<TargetDeviceSpecInterface>
    $cppClass::getDeviceSpecForDeviceID(
        TargetSystemSpecInterface::DeviceID deviceID) {
      for (const auto& entry : getEntries()) {
        if (entry.first == deviceID)
          return entry.second;
      }
      return std::nullopt;
    }
  }];
}

//===----------------------------------------------------------------------===//
// TargetDeviceSpecAttr
//===----------------------------------------------------------------------===//

def DLTI_TargetDeviceSpecAttr :
    DLTIAttr<"TargetDeviceSpec", [TargetDeviceSpecInterface]> {
  let summary = "An attribute to represent target device specification.";
  let description = [{
    Each device specification describes a single device and its
    hardware properties. Each device specification can contain any number
    of optional hardware properties (e.g., max_vector_op_width below).

    Example:
    ```
    #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.max_vector_op_width", 64 : ui32>>
    ```
  }];
  let parameters = (ins
    ArrayRefParameter<"DataLayoutEntryInterface", "">:$entries
  );
  let mnemonic = "target_device_spec";
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $entries `>`";

  let extraClassDeclaration = [{
    /// Returns the attribute associated with the key.
    FailureOr<Attribute> query(DataLayoutEntryKey key) const {
      return llvm::cast<mlir::TargetDeviceSpecInterface>(*this).queryHelper(key);
    }
  }];
}

#endif  // MLIR_DIALECT_DLTI_DLTIATTRS_TD
