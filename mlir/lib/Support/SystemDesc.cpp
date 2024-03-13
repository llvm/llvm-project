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

// ManagedStatic<SystemDesc> systemDesc;

DeviceDesc DeviceDesc::parseDeviceDescFromJSON(
    const DeviceDescJSONTy &device_desc_in_json) {
  // ID and Type are mandatory fields.
  auto iter = device_desc_in_json.find("ID");
  if (iter == device_desc_in_json.end())
    llvm::report_fatal_error("\"ID\" key missing in Device Description");
  DeviceID id = DeviceDesc::strToDeviceID(iter->second);

  iter = device_desc_in_json.find("Type");
  if (iter == device_desc_in_json.end())
    llvm::report_fatal_error("\"Type\" key missing in Device Description");
  DeviceType type = DeviceDesc::strToDeviceType(iter->second);

  // Now process optional fields: description and properties
  DeviceDesc device_desc(id, type);
  for (const auto &property : device_desc_in_json) {
    // skip ID and Type as we have already processed those mandatory fields.
    if (property.first != "ID" && property.first != "Type") {
      if (property.first == "Description")
        device_desc.setDescription(property.second);
      else
        device_desc.setProperty(property.first, property.second);
    }
  }
  return device_desc;
}

LogicalResult SystemDesc::readSystemDescFromJSONFile(llvm::StringRef filename) {
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file =
      openInputFile(filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // Code to parse here
  auto parsed = llvm::json::parse(file.get()->getBuffer());
  if (!parsed) {
    report_fatal_error(parsed.takeError());
  }

  json::Path::Root NullRoot;
  using SystemDescJSONTy = std::vector<mlir::DeviceDesc::DeviceDescJSONTy>;
  SystemDescJSONTy system_desc_in_json;
  if (!json::fromJSON(*parsed, system_desc_in_json, NullRoot)) {
    report_fatal_error("Invalid System Description in JSON");
  }
  for (auto device_desc_in_json : system_desc_in_json) {
    auto device_desc = DeviceDesc::parseDeviceDescFromJSON(device_desc_in_json);
    this->addDeviceDesc(device_desc);
  }

  return success();
}
