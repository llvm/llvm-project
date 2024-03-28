//===- SYSTEMDESC.h - class to represent hardware configuration --*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hardware configuration provides commonly used hardware information to
// different users, such as optimization passes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_SYSTEMDESC_H
#define MLIR_SUPPORT_SYSTEMDESC_H

#include <map>
#include <memory>
#include <vector>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

/// Sytem description file contains a list of device descriptions that
/// each describe a device (e.g., CPU, GPU, ASIC, etc.) in the system.
/// Example:
/// [
///  {
///    "ID": 1,
///    "TYPE": "CPU",
///    "DESCRIPTION": "Intel Xeon 8480",
///    "L1_CACHE_SIZE_IN_BYTES": 8192,
///    ...
///  },
///  {
///
///  },
///  ...
/// ]
namespace mlir {

/// Describes the individual device from the system description
class DeviceDesc {
public:
  using DeviceID = uint32_t;
  using DevicePropertiesMapTy = mlir::NamedAttrList;
  typedef enum { CPU, GPU, SPECIAL } DeviceType;

  /// Basic constructor
  DeviceDesc() = delete;
  DeviceDesc(DeviceID id, DeviceType type) : ID(id), type(type) {}
  bool operator==(const mlir::DeviceDesc &rhs) const {
    return ID == rhs.getID() && type == rhs.getType() &&
           deviceProperties == rhs.getProperties();
  }
  bool operator!=(const mlir::DeviceDesc &rhs) const { return !(*this == rhs); }

  /// Type converters
  static DeviceID strToDeviceID(const std::string &id_str) {
    llvm::Expected<int64_t> id = llvm::json::parse<int64_t>(id_str);
    if (!id)
      llvm::report_fatal_error("Value of \"ID\" is not int");
    return static_cast<DeviceID>(id.get());
  }
  static DeviceType strToDeviceType(const std::string &type_str) {
    if (type_str == "CPU")
      return DeviceType::CPU;
    else if (type_str == "GPU")
      return DeviceType::GPU;
    else if (type_str == "SPECIAL")
      return DeviceType::SPECIAL;
    llvm::report_fatal_error("Value of \"Type\" is not CPU, GPU, or SPECIAL");
  }

  /// Set description
  DeviceDesc &setDescription(std::string desc) {
    description = desc;
    return *this;
  }

  /// Set property
  DeviceDesc &setProperty(MLIRContext *context, llvm::StringRef name,
                          int64_t iv) {
    std::optional<NamedAttribute> attr = deviceProperties.getNamed(name);
    if (!attr.has_value()) {
      IntegerType int64Ty = IntegerType::get(context, 64);
      Attribute value = IntegerAttr::get(int64Ty, iv);
      deviceProperties.append(name, value);
    } else
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    return *this;
  }

  DeviceDesc &setProperty(MLIRContext *context, llvm::StringRef name,
                          double dv) {
    std::optional<NamedAttribute> attr = deviceProperties.getNamed(name);
    if (!attr.has_value()) {
      FloatType floatType = FloatType::getF64(context);
      Attribute value = FloatAttr::get(floatType, dv);
      deviceProperties.append(name, value);
    } else
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    return *this;
  }

  DeviceDesc &setProperty(MLIRContext *context, llvm::StringRef name,
                          const std::vector<int64_t> &ivv) {
    std::optional<NamedAttribute> attr = deviceProperties.getNamed(name);
    if (!attr.has_value()) {
      IntegerType int64Ty = IntegerType::get(context, 64);
      RankedTensorType shape =
          RankedTensorType::get({static_cast<long>(ivv.size()), 1}, int64Ty);
      DenseElementsAttr value =
          DenseElementsAttr::get(shape, llvm::ArrayRef(ivv));
      deviceProperties.append(name, value);
    } else
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    return *this;
  }

  DeviceDesc &setProperty(MLIRContext *context, llvm::StringRef name,
                          const std::vector<double> &idv) {
    std::optional<NamedAttribute> attr = deviceProperties.getNamed(name);
    if (!attr.has_value()) {
      FloatType float64Ty = FloatType::getF64(context);
      RankedTensorType shape =
          RankedTensorType::get({static_cast<long>(idv.size()), 1}, float64Ty);
      DenseElementsAttr value =
          DenseElementsAttr::get(shape, llvm::ArrayRef(idv));
      deviceProperties.append(name, value);
    } else
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    return *this;
  }

  // We provide convenience interface to handle int/float value as string
  DeviceDesc &setProperty(MLIRContext *context, llvm::StringRef name,
                          const std::string &json_value) {
    if (json_value.length() > 0 && json_value[0] == '[') {
      // Parse as an array
      llvm::Expected<std::vector<int64_t>> ivv =
          llvm::json::parse<std::vector<int64_t>>(json_value);
      if (ivv) {
        *this = this->setProperty(context, name, ivv.get());
        return *this;
      }

      llvm::Expected<std::vector<double>> idv =
          llvm::json::parse<std::vector<double>>(json_value);
      if (idv) {
        *this = this->setProperty(context, name, idv.get());
        return *this;
      }
    } else {
      // int64_t because llvm::json has int64_t support (not int)
      llvm::Expected<int64_t> iv = llvm::json::parse<int64_t>(json_value);
      if (iv) {
        *this = this->setProperty(context, name, iv.get());
        return *this;
      }

      // Int type failed, try float now.
      // double because llvm::json has double support (not float)
      llvm::Expected<double> dv = llvm::json::parse<double>(json_value);
      if (dv) {
        *this = this->setProperty(context, name, dv.get());
        return *this;
      }
    }

    llvm::report_fatal_error(
        "Neither int/float/vector value in Device Description: key " + name);
  }

  /// Get ID
  DeviceID getID() const { return ID; }
  /// Get device type
  DeviceType getType() const { return type; }
  /// Get device description
  std::string getDescription() const { return description; }
  /// Get all of device properties
  const DevicePropertiesMapTy &getProperties() const {
    return deviceProperties;
  }
  /// Get property value: returns the value of the property with given name, if
  /// it exists. Otherwise returns std::nullopt.
  std::optional<int64_t> getPropertyValueAsInt(llvm::StringRef name) const {
    // check that property with the given name exists
    std::optional<NamedAttribute> attr = deviceProperties.getNamed(name);
    if (attr) {
      if (IntegerAttr intAttr = dyn_cast<IntegerAttr>(attr->getValue()))
        return intAttr.getInt();
    }
    return std::nullopt;
  }
  std::optional<double> getPropertyValueAsFloat(llvm::StringRef name) const {
    // check that property with the given name exists
    std::optional<NamedAttribute> attr = deviceProperties.getNamed(name);
    if (attr) {
      if (FloatAttr floatAttr = dyn_cast<FloatAttr>(attr->getValue()))
        return floatAttr.getValueAsDouble();
    }
    return std::nullopt;
  }

  /// Special functions
  auto getAllDevicePropertyNames() const {
    return llvm::map_range(
        deviceProperties.getAttrs(),
        [](const NamedAttribute &named_attribute) -> llvm::StringRef {
          return named_attribute.getName();
        });
  }

  // -----------------------------------------------------------------------
  //          CPU specific methods
  // -----------------------------------------------------------------------
  static constexpr llvm::StringRef getCPUL1CacheSizeInBytesKeyName() {
    return "L1_CACHE_SIZE_IN_BYTES";
  }
  static constexpr llvm::StringRef getConvAndMatMulBlockingFactorKeyName() {
    return "CONV_AND_MATMUL_BLOCKING_FACTOR";
  }
  static constexpr llvm::StringRef getMatMulTileSizeInBytesKeyName() {
    return "MATMUL_TILE_SIZE_IN_BYTES";
  }
  static constexpr llvm::StringRef getCanonicalizerMaxIterationsKeyName() {
    return "CANONICALIZER_MAX_ITERS";
  }
  static constexpr llvm::StringRef getCanonicalizerMaxNumRewritesKeyName() {
    return "CANONICALIZER_MAX_NUM_REWRITES";
  }
  static constexpr llvm::StringRef getMaxVectorWidthKeyName() {
    return "MAX_VECTOR_WIDTH";
  }

  std::optional<int64_t> getL1CacheSizeInBytes() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getCPUL1CacheSizeInBytesKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setL1CacheSizeInBytes(MLIRContext *context, int64_t value) {
    // Temporarily use int override until we support size_t
    this->setProperty(context, DeviceDesc::getCPUL1CacheSizeInBytesKeyName(),
                      value);
  }
  std::optional<int64_t> getConvAndMatMulBlockingFactor() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getConvAndMatMulBlockingFactorKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setConvAndMatMulBlockingFactor(MLIRContext *context, int64_t value) {
    // Temporarily use int override until we support size_t
    this->setProperty(
        context, DeviceDesc::getConvAndMatMulBlockingFactorKeyName(), value);
  }
  std::optional<int64_t> getMatMulTileSizeInBytes() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getMatMulTileSizeInBytesKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setMatMulTileSizeInBytes(MLIRContext *context, int64_t value) {
    // Temporarily use int override until we support size_t
    this->setProperty(context, DeviceDesc::getMatMulTileSizeInBytesKeyName(),
                      value);
  }
  std::optional<int64_t> getCanonicalizerMaxNumRewrites() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getCanonicalizerMaxNumRewritesKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setCanonicalizerMaxNumRewrites(MLIRContext *context, int64_t value) {
    this->setProperty(
        context, DeviceDesc::getCanonicalizerMaxNumRewritesKeyName(), value);
  }
  std::optional<int64_t> getCanonicalizerMaxIterations() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getCanonicalizerMaxIterationsKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setCanonicalizerMaxIterations(MLIRContext *context, int64_t value) {
    this->setProperty(
        context, DeviceDesc::getCanonicalizerMaxIterationsKeyName(), value);
  }
  std::optional<int64_t> getMaxVectorWidth() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getMaxVectorWidthKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setMaxVectorWidth(MLIRContext *context, uint32_t value) {
    this->setProperty(context, DeviceDesc::getMaxVectorWidthKeyName(),
                      static_cast<int64_t>(value));
  }

private:
  /// Unique device ID for every device
  DeviceID ID;

  /// Type of device
  DeviceType type;

  /// Some description of the device
  std::string description;

  /// Dictionary to store rest of the properties
  DevicePropertiesMapTy deviceProperties;
};

class SystemDesc {
public:
  SystemDesc() = default;
  SystemDesc(const SystemDesc &desc) { this->deviceDescs = desc.deviceDescs; }
  void operator=(const SystemDesc &rhs) { this->deviceDescs = rhs.deviceDescs; }

  /// Insert a new device description
  SystemDesc &addDeviceDesc(const DeviceDesc &desc) {
    auto inserted = deviceDescs.insert(std::make_pair(desc.getID(), desc));
    if (!inserted.second || inserted.first->second != desc) {
      llvm::report_fatal_error("Duplicate device description for ID:" +
                               llvm::StringRef(std::to_string(desc.getID())));
    }
    return *this;
  }

  /// Get a device description
  const DeviceDesc &getDeviceDesc(DeviceDesc::DeviceID deviceID) {
    auto iter = deviceDescs.find(deviceID);
    if (iter != deviceDescs.end()) {
      return iter->second;
    }
    llvm::report_fatal_error("Device description with ID not found:" +
                             llvm::StringRef(std::to_string(deviceID)));
  }

  /// Types
  using DeviceDescsMapTy = std::map<DeviceDesc::DeviceID, DeviceDesc>;

  // Generic functions: TODO
  /// Get number of CPU devices in the system
  static uint32_t getNumCPUDevices() { return 0; }
  static uint32_t getNumGPUDevices() { return 0; }

private:
  /// Map to store all the device descriptions
  DeviceDescsMapTy deviceDescs;
};

// An abstract class that represent device description for an abstract base
// device
//
// This class specifies a set of device properties that could be specified by
// the default device descriptor that will be used in case a user does not
// specify its own properties for the device.
class DefaultBaseDeviceDesc {
public:
  virtual ~DefaultBaseDeviceDesc() {}
  virtual void registerDeviceDesc(MLIRContext *context) const = 0;

  /// -----------------------------------------------------------------------
  /// Device-agnostic parameters of system description
  /// -----------------------------------------------------------------------
  /// Set of common questions asked by various passes
  // Blocking factor and tile size are typically used by tile/block passes.
  virtual void setConvAndMatMulBlockingFactor(MLIRContext *context){};
  virtual void setMatMulTileSize(MLIRContext *context){};

  virtual void setCanonicalizerMaxIterations(MLIRContext *context){};
  virtual void setCanonicalizerMaxNumRewrites(MLIRContext *context){};

  /// -----------------------------------------------------------------------
  /// CPU-specific parameters of system description
  /// -----------------------------------------------------------------------
  virtual void setL1CacheSizeInBytes(MLIRContext *context){};

  /// -----------------------------------------------------------------------
  /// GPU-specific parameters of system description
  /// -----------------------------------------------------------------------
  // Used by Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp#L52
  virtual void setMaxVectorWidth(MLIRContext *context){};
};

// Class that represent device description for a typical CPU device
class DefaultCPUDeviceDesc : public DefaultBaseDeviceDesc {
public:
  // We use default ID of 0 because we are expecting to have only one device so
  // far. Not heterogeneous setup.
  DefaultCPUDeviceDesc(MLIRContext *context)
      : cpu_device_desc(DeviceDesc(/* id */ 0, DeviceDesc::CPU)) {
    // Register all system properties
    this->setL1CacheSizeInBytes(context);
    this->setConvAndMatMulBlockingFactor(context);
    this->setMatMulTileSize(context);
    this->setCanonicalizerMaxNumRewrites(context);
    this->setCanonicalizerMaxIterations(context);
  }

  ~DefaultCPUDeviceDesc() {}

  void registerDeviceDesc(MLIRContext *context) const override {
    context->getSystemDesc().addDeviceDesc(cpu_device_desc);
  }

  // -------------------------------------------------------------------------
  //                    CPU-specific properties
  // -------------------------------------------------------------------------

  void setL1CacheSizeInBytes(MLIRContext *context) override {
    cpu_device_desc.setL1CacheSizeInBytes(context, 8192);
  }
  void setConvAndMatMulBlockingFactor(MLIRContext *context) override {
    cpu_device_desc.setConvAndMatMulBlockingFactor(context, 32);
  }
  void setMatMulTileSize(MLIRContext *context) override {
    cpu_device_desc.setMatMulTileSizeInBytes(context, 32);
  }
  void setCanonicalizerMaxNumRewrites(MLIRContext *context) override {
    // taken from include/mlir/Transforms/Passes.td
    cpu_device_desc.setCanonicalizerMaxNumRewrites(context, -1);
  }
  void setCanonicalizerMaxIterations(MLIRContext *context) override {
    // taken from include/mlir/Transforms/Passes.td
    cpu_device_desc.setCanonicalizerMaxIterations(context, 10);
  }

private:
  DeviceDesc cpu_device_desc;
};

class DefaultGPUDeviceDesc : public DefaultBaseDeviceDesc {
public:
  // We use default ID of 0 because we are expecting to have only one device so
  // far. Not heterogeneous setup.
  DefaultGPUDeviceDesc(MLIRContext *context)
      : gpu_device_desc(DeviceDesc(/* id */ 1, DeviceDesc::GPU)) {
    // GPU device supports default value for MaxVectorWidth so far.
    this->setMaxVectorWidth(context);
  }

  ~DefaultGPUDeviceDesc() {}

  void registerDeviceDesc(MLIRContext *context) const override {
    context->getSystemDesc().addDeviceDesc(gpu_device_desc);
  }

  // -------------------------------------------------------------------------
  //                    GPU-specific properties
  // -------------------------------------------------------------------------

  void setMaxVectorWidth(MLIRContext *context) override {
    // Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp#L52
    gpu_device_desc.setMaxVectorWidth(context, 128);
  }

private:
  DeviceDesc gpu_device_desc;
};

// ---------------------------------------------------------------------------
//                     Config file readers
// ---------------------------------------------------------------------------
namespace impl {
class SystemDescJSONConfigParser {
public:
  /// Build SystemDesc by parsing input config file in JSON format.
  /// Returns a valid SystemDesc if parsing is successful; otherwise
  /// returns std::nullopt.
  static std::optional<SystemDesc>
  buildSystemDescFromConfigFile(MLIRContext *context, llvm::StringRef filename);

private:
  /// We represent DeviceDesc in JSON as a key-value pairs of strings.
  using DeviceDescJSONTy = std::map<std::string, std::string>;

  /// A utility function to parse device description entry in JSON format
  /// Returns valid DeviceDesc if parsing is successful; otherwise returns
  /// std::nullopt.
  static std::optional<DeviceDesc>
  buildDeviceDescFromConfigFile(MLIRContext *context,
                                const DeviceDescJSONTy &device_desc_in_json);
};
} // namespace impl

class SystemDescConfigFileParser {
public:
  /// Build SystemDesc by parsing input config file. Returns valid SystemDesc
  /// if parsing is successful; otherwise returns std::nullopt.
  static std::optional<SystemDesc>
  buildSystemDescFromConfigFile(MLIRContext *context,
                                llvm::StringRef filename) {
    // Once we support more formats, we can accept format as the input argument.
    return impl::SystemDescJSONConfigParser::buildSystemDescFromConfigFile(
        context, filename);
  }
};

} // namespace mlir
#endif // MLIR_SUPPORT_SYSTEMDESC_H
