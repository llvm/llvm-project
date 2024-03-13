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

#include "mlir/IR/MLIRContext.h"
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
  /// Some typedefs
  using DeviceID = uint32_t;
  using DevicePropertyName = std::string;
  struct DevicePropertyValue {
    enum Tag { INT, DOUBLE, INT_VECTOR, DOUBLE_VECTOR } tag;
    struct Data {
      int64_t iValue;
      double dValue;
      std::vector<int64_t> ivValue;
      std::vector<double> dvValue;

      Data() : iValue(0), dValue(0.0), ivValue({0}), dvValue({0.0}) {}
      ~Data() {}
    } data;

    DevicePropertyValue() = default;
    DevicePropertyValue(const mlir::DeviceDesc::DevicePropertyValue &rhs) {
      this->tag = rhs.tag;
      if (this->tag == INT)
        this->data.iValue = rhs.data.iValue;
      else if (this->tag == DOUBLE)
        this->data.dValue = rhs.data.dValue;
      else if (this->tag == INT_VECTOR)
        this->data.ivValue = rhs.data.ivValue;
      else
        this->data.dvValue = rhs.data.dvValue;
    }
    bool operator==(const mlir::DeviceDesc::DevicePropertyValue &rhs) const {
      return tag == rhs.tag &&
             ((tag == INT && data.iValue == rhs.data.iValue) ||
              (tag == DOUBLE && data.dValue == rhs.data.dValue) ||
              (tag == INT_VECTOR && data.ivValue == rhs.data.ivValue) ||
              (tag == DOUBLE_VECTOR && data.dvValue == rhs.data.dvValue));
    }
    bool operator!=(const mlir::DeviceDesc::DevicePropertyValue &rhs) const {
      return !(*this == rhs);
    }
  };
  using DevicePropertiesMapTy =
      std::map<DevicePropertyName, DevicePropertyValue>;

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
  DeviceDesc &setProperty(llvm::StringRef name, int64_t iv) {
    DevicePropertyValue value;
    value.tag = DevicePropertyValue::Tag::INT;
    value.data.iValue = iv;
    auto inserted =
        deviceProperties.insert(std::make_pair(std::string(name), value));
    if (!inserted.second && inserted.first->second != value) {
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    }
    return *this;
  }
  DeviceDesc &setProperty(llvm::StringRef name, double dv) {
    DevicePropertyValue value;
    value.tag = DevicePropertyValue::Tag::DOUBLE;
    value.data.dValue = dv;
    auto inserted =
        deviceProperties.insert(std::make_pair(std::string(name), value));
    if (!inserted.second && inserted.first->second != value) {
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    }
    return *this;
  }
  DeviceDesc &setProperty(llvm::StringRef name,
                          const std::vector<int64_t> &ivv) {
    DevicePropertyValue value;
    value.tag = DevicePropertyValue::Tag::INT_VECTOR;
    value.data.ivValue = ivv;
    auto inserted =
        deviceProperties.insert(std::make_pair(std::string(name), value));
    if (!inserted.second && inserted.first->second != value) {
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    }
    return *this;
  }
  DeviceDesc &setProperty(llvm::StringRef name,
                          const std::vector<double> &idv) {
    DevicePropertyValue value;
    value.tag = DevicePropertyValue::Tag::DOUBLE_VECTOR;
    value.data.dvValue = idv;
    auto inserted =
        deviceProperties.insert(std::make_pair(std::string(name), value));
    if (!inserted.second && inserted.first->second != value) {
      llvm::report_fatal_error("Duplicate device property name found:" + name);
    }
    return *this;
  }
  // We provide convenience interface to handle int/float value as string
  DeviceDesc &setProperty(llvm::StringRef name, const std::string &json_value) {
    if (json_value.length() > 0 && json_value[0] == '[') {
      // Parse as an array
      llvm::Expected<std::vector<int64_t>> ivv =
          llvm::json::parse<std::vector<int64_t>>(json_value);
      if (ivv) {
        *this = this->setProperty(name, ivv.get());
        return *this;
      }

      llvm::Expected<std::vector<double>> idv =
          llvm::json::parse<std::vector<double>>(json_value);
      if (idv) {
        *this = this->setProperty(name, idv.get());
        return *this;
      }
    } else {
      // int64_t because llvm::json has int64_t support (not int)
      llvm::Expected<int64_t> iv = llvm::json::parse<int64_t>(json_value);
      if (iv) {
        *this = this->setProperty(name, iv.get());
        return *this;
      }

      // Int type failed, try float now.
      // double because llvm::json has double support (not float)
      llvm::Expected<double> dv = llvm::json::parse<double>(json_value);
      if (dv) {
        *this = this->setProperty(name, dv.get());
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
  /// it exists. Otherwise throws exception (TODO)
  std::optional<int64_t> getPropertyValueAsInt(llvm::StringRef name) const {
    // check that property with the given name exists
    auto iter = deviceProperties.find(std::string(name));
    if (iter == deviceProperties.end()) {
      return std::nullopt;
    }
    // TODO: we can do a tag check here.
    return iter->second.data.iValue;
  }
  std::optional<double> getPropertyValueAsFloat(llvm::StringRef name) const {
    // check that property with the given name exists
    auto iter = deviceProperties.find(std::string(name));
    if (iter == deviceProperties.end()) {
      return std::nullopt;
    }
    // TODO: we can do a tag check here.
    return iter->second.data.dValue;
  }

  /// Special functions
  auto getAllDevicePropertyNames() const {
    return llvm::map_range(
        deviceProperties,
        [](const DevicePropertiesMapTy::value_type &item) -> llvm::StringRef {
          return item.first;
        });
  }

  /// We use a list of key-value pairs to represent a system description in
  /// JSON.
  using DeviceDescJSONTy = std::map<std::string, std::string>;
  static DeviceDesc
  parseDeviceDescFromJSON(const DeviceDescJSONTy &device_desc);

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
  void setL1CacheSizeInBytes(int64_t value) {
    // Temporarily use int override until we support size_t
    this->setProperty(DeviceDesc::getCPUL1CacheSizeInBytesKeyName(), value);
  }
  std::optional<int64_t> getConvAndMatMulBlockingFactor() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getConvAndMatMulBlockingFactorKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setConvAndMatMulBlockingFactor(int64_t value) {
    // Temporarily use int override until we support size_t
    this->setProperty(DeviceDesc::getConvAndMatMulBlockingFactorKeyName(),
                      value);
  }
  std::optional<int64_t> getMatMulTileSizeInBytes() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getMatMulTileSizeInBytesKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setMatMulTileSizeInBytes(int64_t value) {
    // Temporarily use int override until we support size_t
    this->setProperty(DeviceDesc::getMatMulTileSizeInBytesKeyName(), value);
  }
  std::optional<int64_t> getCanonicalizerMaxNumRewrites() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getCanonicalizerMaxNumRewritesKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setCanonicalizerMaxNumRewrites(int64_t value) {
    this->setProperty(DeviceDesc::getCanonicalizerMaxNumRewritesKeyName(),
                      value);
  }
  std::optional<int64_t> getCanonicalizerMaxIterations() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getCanonicalizerMaxIterationsKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setCanonicalizerMaxIterations(int64_t value) {
    this->setProperty(DeviceDesc::getCanonicalizerMaxIterationsKeyName(),
                      value);
  }
  std::optional<int64_t> getMaxVectorWidth() const {
    if (std::optional<int64_t> v = this->getPropertyValueAsInt(
            DeviceDesc::getMaxVectorWidthKeyName())) {
      return v;
    }
    return std::nullopt;
  }
  void setMaxVectorWidth(uint32_t value) {
    this->setProperty(DeviceDesc::getMaxVectorWidthKeyName(),
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

  /// Read and parse system description from JSON file
  LogicalResult readSystemDescFromJSONFile(llvm::StringRef filename);
  void writeSystemDescToJSONFile(llvm::StringRef filename);

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
  SystemDesc(const SystemDesc &) = delete;
  void operator=(const SystemDesc &) = delete;

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
  virtual void setConvAndMatMulBlockingFactor(){};
  virtual void setMatMulTileSize(){};

  virtual void setCanonicalizerMaxIterations(){};
  virtual void setCanonicalizerMaxNumRewrites(){};

  /// -----------------------------------------------------------------------
  /// CPU-specific parameters of system description
  /// -----------------------------------------------------------------------
  virtual void setL1CacheSizeInBytes(){};

  /// -----------------------------------------------------------------------
  /// GPU-specific parameters of system description
  /// -----------------------------------------------------------------------
  // Used by Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp#L52
  virtual void setMaxVectorWidth(){};
};

// Class that represent device description for a typical CPU device
class DefaultCPUDeviceDesc : public DefaultBaseDeviceDesc {
public:
  // We use default ID of 0 because we are expecting to have only one device so
  // far. Not heterogeneous setup.
  DefaultCPUDeviceDesc()
      : cpu_device_desc(DeviceDesc(/* id */ 0, DeviceDesc::CPU)) {
    // Register all system properties
    this->setL1CacheSizeInBytes();
    this->setConvAndMatMulBlockingFactor();
    this->setMatMulTileSize();
    this->setCanonicalizerMaxNumRewrites();
    this->setCanonicalizerMaxIterations();
  }

  ~DefaultCPUDeviceDesc() {}

  void registerDeviceDesc(MLIRContext *context) const override {
    context->getSystemDesc().addDeviceDesc(cpu_device_desc);
  }

  // -------------------------------------------------------------------------
  //                    CPU-specific properties
  // -------------------------------------------------------------------------

  void setL1CacheSizeInBytes() override {
    cpu_device_desc.setL1CacheSizeInBytes(8192);
  }
  void setConvAndMatMulBlockingFactor() override {
    cpu_device_desc.setConvAndMatMulBlockingFactor(32);
  }
  void setMatMulTileSize() override {
    cpu_device_desc.setMatMulTileSizeInBytes(32);
  }
  void setCanonicalizerMaxNumRewrites() override {
    // taken from include/mlir/Transforms/Passes.td
    cpu_device_desc.setCanonicalizerMaxNumRewrites(-1);
  }
  void setCanonicalizerMaxIterations() override {
    // taken from include/mlir/Transforms/Passes.td
    cpu_device_desc.setCanonicalizerMaxIterations(10);
  }

private:
  DeviceDesc cpu_device_desc;
};

class DefaultGPUDeviceDesc : public DefaultBaseDeviceDesc {
public:
  // We use default ID of 0 because we are expecting to have only one device so
  // far. Not heterogeneous setup.
  DefaultGPUDeviceDesc()
      : gpu_device_desc(DeviceDesc(/* id */ 1, DeviceDesc::GPU)) {
    // GPU device supports default value for MaxVectorWidth so far.
    this->setMaxVectorWidth();
  }

  ~DefaultGPUDeviceDesc() {}

  void registerDeviceDesc(MLIRContext *context) const override {
    context->getSystemDesc().addDeviceDesc(gpu_device_desc);
  }

  // -------------------------------------------------------------------------
  //                    GPU-specific properties
  // -------------------------------------------------------------------------

  void setMaxVectorWidth() override {
    // Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp#L52
    gpu_device_desc.setMaxVectorWidth(128);
  }

private:
  DeviceDesc gpu_device_desc;
};

} // namespace mlir
#endif // MLIR_SUPPORT_SYSTEMDESC_H
