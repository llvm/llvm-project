//===- HardwareConfig.cpp - Hardware configuration
//----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/SystemDesc.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace mlir;

std::optional<DeviceDesc> impl::SystemDescJSONConfigParser::buildDeviceDescFromConfigFile(
    MLIRContext *context, const DeviceDescJSONTy &device_desc_in_json) {
  // ID and Type are mandatory fields.
  auto iter = device_desc_in_json.find("ID");
  if (iter == device_desc_in_json.end()) {
    llvm::errs() << "\"ID\" key missing in Device Description" << "\n";
    return std::nullopt;
  }
  DeviceDesc::DeviceID id = DeviceDesc::strToDeviceID(iter->second);

  iter = device_desc_in_json.find("Type");
  if (iter == device_desc_in_json.end()) {
    llvm::errs() << "\"Type\" key missing in Device Description" << "\n";
    return std::nullopt;
  }
  DeviceDesc::DeviceType type = DeviceDesc::strToDeviceType(iter->second);

  // Now process optional fields: description and properties
  DeviceDesc device_desc(id, type);
  for (const auto &property : device_desc_in_json) {
    // skip ID and Type as we have already processed those mandatory fields.
    if (property.first != "ID" && property.first != "Type") {
      if (property.first == "Description")
        device_desc.setDescription(property.second);
      else
        device_desc.setProperty(context, property.first, property.second);
    }
  }
  return std::optional<DeviceDesc>(device_desc);
}

std::optional<SystemDesc> impl::SystemDescJSONConfigParser::buildSystemDescFromConfigFile(
    MLIRContext *context, llvm::StringRef filename) {
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return std::nullopt;
  }

  // Code to parse here
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) {
    llvm::errs() << parsed.takeError();
    return std::nullopt;
  }

  json::Path::Root NullRoot;
  // System description is a list of Device descriptions.
  using SystemDescJSONTy = std::vector<DeviceDescJSONTy>;
  SystemDescJSONTy system_desc_in_json;
  if (!json::fromJSON(*parsed, system_desc_in_json, NullRoot)) {
    llvm::errs() << "Invalid System Description in JSON" << "\n";
    return std::nullopt;
  }

  SystemDesc system_desc;
  for (auto device_desc_in_json : system_desc_in_json) {
    std::optional<DeviceDesc> device_desc = impl::SystemDescJSONConfigParser::buildDeviceDescFromConfigFile(
      context, device_desc_in_json);
    if (device_desc)
      system_desc.addDeviceDesc(*device_desc);
    else
      return std::nullopt;
  }

  return std::optional<SystemDesc>(system_desc);
}