//===- llvm-offload-device-info.cpp - Print liboffload properties ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that, by using the new liboffload API, prints
// all devices and properties
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include <OffloadAPI.h>
#include <iostream>
#include <vector>

using namespace llvm;

static cl::opt<bool> JSON("json",
                          cl::desc("Dump device information as a JSON map"),
                          cl::init(false));

#define OFFLOAD_ERR(X)                                                         \
  if (auto Err = X) {                                                          \
    return Err;                                                                \
  }

enum class PrintKind {
  NORMAL,
  FP_FLAGS,
};

template <typename T, PrintKind PK = PrintKind::NORMAL>
void doWrite(std::ostream &S, T &&Val) {
  S << Val;
}

std::string toString(ol_platform_backend_t Val) {
  switch (Val) {
  case OL_PLATFORM_BACKEND_UNKNOWN:
    return "UNKNOWN";
  case OL_PLATFORM_BACKEND_CUDA:
    return "CUDA";
  case OL_PLATFORM_BACKEND_AMDGPU:
    return "AMDGPU";
  case OL_PLATFORM_BACKEND_HOST:
    return "HOST";
  default:
    return "INVALID";
  }
}

std::string toString(ol_device_type_t Val) {
  switch (Val) {
  case OL_DEVICE_TYPE_GPU:
    return "GPU";
  case OL_DEVICE_TYPE_CPU:
    return "CPU";
  case OL_DEVICE_TYPE_HOST:
    return "HOST";
  default:
    return "INVALID";
  }
}

template <>
void doWrite<ol_platform_backend_t>(std::ostream &S,
                                    ol_platform_backend_t &&Val) {
  S << toString(Val);
}

template <>
void doWrite<ol_device_type_t>(std::ostream &S, ol_device_type_t &&Val) {
  S << toString(Val);
}
template <>
void doWrite<ol_dimensions_t>(std::ostream &S, ol_dimensions_t &&Val) {
  S << "{x: " << Val.x << ", y: " << Val.y << ", z: " << Val.z << "}";
}
template <>
void doWrite<ol_device_fp_capability_flags_t, PrintKind::FP_FLAGS>(
    std::ostream &S, ol_device_fp_capability_flags_t &&Val) {
  S << Val << " {";

  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT) {
    S << " CORRECTLY_ROUNDED_DIVIDE_SQRT";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST) {
    S << " ROUND_TO_NEAREST";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO) {
    S << " ROUND_TO_ZERO";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF) {
    S << " ROUND_TO_INF";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN) {
    S << " INF_NAN";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_DENORM) {
    S << " DENORM";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_FMA) {
    S << " FMA";
  }
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT) {
    S << " SOFT_FLOAT";
  }

  S << " }";
}

json::Object toJSON(ol_dimensions_t Val) {
  return json::Object{{"x", Val.x}, {"y", Val.y}, {"z", Val.z}};
}

std::vector<std::string> toJSON(ol_device_fp_capability_flags_t Val) {
  std::vector<std::string> Flags;
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_CORRECTLY_ROUNDED_DIVIDE_SQRT)
    Flags.push_back("CORRECTLY_ROUNDED_DIVIDE_SQRT");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_NEAREST)
    Flags.push_back("ROUND_TO_NEAREST");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_ZERO)
    Flags.push_back("ROUND_TO_ZERO");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_ROUND_TO_INF)
    Flags.push_back("ROUND_TO_INF");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_INF_NAN)
    Flags.push_back("INF_NAN");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_DENORM)
    Flags.push_back("DENORM");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_FMA)
    Flags.push_back("FMA");
  if (Val & OL_DEVICE_FP_CAPABILITY_FLAG_SOFT_FLOAT)
    Flags.push_back("SOFT_FLOAT");
  return Flags;
}

template <typename T>
ol_result_t printPlatformValue(std::ostream &S, ol_platform_handle_t Plat,
                               ol_platform_info_t Info, const char *Desc) {
  S << Desc << ": ";

  if constexpr (std::is_pointer_v<T>) {
    std::vector<uint8_t> Val;
    size_t Size;
    OFFLOAD_ERR(olGetPlatformInfoSize(Plat, Info, &Size));
    Val.resize(Size);
    OFFLOAD_ERR(olGetPlatformInfo(Plat, Info, sizeof(Val), Val.data()));
    doWrite(S, reinterpret_cast<T>(Val.data()));
  } else {
    T Val;
    OFFLOAD_ERR(olGetPlatformInfo(Plat, Info, sizeof(Val), &Val));
    doWrite(S, std::move(Val));
  }
  S << "\n";
  return OL_SUCCESS;
}

template <typename T, PrintKind PK = PrintKind::NORMAL>
ol_result_t printDeviceValue(std::ostream &S, ol_device_handle_t Dev,
                             ol_device_info_t Info, const char *Desc,
                             const char *Units = nullptr) {
  S << Desc << ": ";

  if constexpr (std::is_pointer_v<T>) {
    std::vector<uint8_t> Val;
    size_t Size;
    OFFLOAD_ERR(olGetDeviceInfoSize(Dev, Info, &Size));
    Val.resize(Size);
    OFFLOAD_ERR(olGetDeviceInfo(Dev, Info, Size, Val.data()));
    doWrite<T, PK>(S, reinterpret_cast<T>(Val.data()));
  } else {
    T Val;
    OFFLOAD_ERR(olGetDeviceInfo(Dev, Info, sizeof(Val), &Val));
    doWrite<T, PK>(S, std::move(Val));
  }
  if (Units)
    S << " " << Units;
  S << "\n";
  return OL_SUCCESS;
}

ol_result_t printDevice(std::ostream &S, ol_device_handle_t D) {
  ol_platform_handle_t Platform;
  OFFLOAD_ERR(
      olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform), &Platform));

  std::vector<char> Name;
  size_t NameSize;
  OFFLOAD_ERR(olGetDeviceInfoSize(D, OL_DEVICE_INFO_PRODUCT_NAME, &NameSize))
  Name.resize(NameSize);
  OFFLOAD_ERR(
      olGetDeviceInfo(D, OL_DEVICE_INFO_PRODUCT_NAME, NameSize, Name.data()));
  S << "[" << Name.data() << "]\n";

  OFFLOAD_ERR(printPlatformValue<const char *>(
      S, Platform, OL_PLATFORM_INFO_NAME, "Platform Name"));
  OFFLOAD_ERR(printPlatformValue<const char *>(
      S, Platform, OL_PLATFORM_INFO_VENDOR_NAME, "Platform Vendor Name"));
  OFFLOAD_ERR(printPlatformValue<const char *>(
      S, Platform, OL_PLATFORM_INFO_VERSION, "Platform Version"));
  OFFLOAD_ERR(printPlatformValue<ol_platform_backend_t>(
      S, Platform, OL_PLATFORM_INFO_BACKEND, "Platform Backend"));

  OFFLOAD_ERR(
      printDeviceValue<const char *>(S, D, OL_DEVICE_INFO_NAME, "Name"));
  OFFLOAD_ERR(printDeviceValue<const char *>(S, D, OL_DEVICE_INFO_PRODUCT_NAME,
                                             "Product Name"));
  OFFLOAD_ERR(printDeviceValue<const char *>(S, D, OL_DEVICE_INFO_UID, "UID"));
  OFFLOAD_ERR(
      printDeviceValue<ol_device_type_t>(S, D, OL_DEVICE_INFO_TYPE, "Type"));
  OFFLOAD_ERR(printDeviceValue<const char *>(
      S, D, OL_DEVICE_INFO_DRIVER_VERSION, "Driver Version"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(
      S, D, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, "Max Work Group Size"));
  OFFLOAD_ERR(printDeviceValue<ol_dimensions_t>(
      S, D, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
      "Max Work Group Size Per Dimension"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_MAX_WORK_SIZE,
                                         "Max Work Size"));
  OFFLOAD_ERR(printDeviceValue<ol_dimensions_t>(
      S, D, OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION,
      "Max Work Size Per Dimension"));
  OFFLOAD_ERR(
      printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_VENDOR_ID, "Vendor ID"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_NUM_COMPUTE_UNITS,
                                         "Num Compute Units"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(
      S, D, OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY, "Max Clock Frequency", "MHz"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_MEMORY_CLOCK_RATE,
                                         "Memory Clock Rate", "MHz"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_ADDRESS_BITS,
                                         "Address Bits"));
  OFFLOAD_ERR(printDeviceValue<uint64_t>(
      S, D, OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, "Max Mem Allocation Size", "B"));
  OFFLOAD_ERR(printDeviceValue<uint64_t>(S, D, OL_DEVICE_INFO_GLOBAL_MEM_SIZE,
                                         "Global Mem Size", "B"));
  OFFLOAD_ERR(
      printDeviceValue<uint64_t>(S, D, OL_DEVICE_INFO_WORK_GROUP_LOCAL_MEM_SIZE,
                                 "Work Group Shared Mem Size", "B"));
  OFFLOAD_ERR(
      (printDeviceValue<ol_device_fp_capability_flags_t, PrintKind::FP_FLAGS>(
          S, D, OL_DEVICE_INFO_SINGLE_FP_CONFIG,
          "Single Precision Floating Point Capability")));
  OFFLOAD_ERR(
      (printDeviceValue<ol_device_fp_capability_flags_t, PrintKind::FP_FLAGS>(
          S, D, OL_DEVICE_INFO_DOUBLE_FP_CONFIG,
          "Double Precision Floating Point Capability")));
  OFFLOAD_ERR(
      (printDeviceValue<ol_device_fp_capability_flags_t, PrintKind::FP_FLAGS>(
          S, D, OL_DEVICE_INFO_HALF_FP_CONFIG,
          "Half Precision Floating Point Capability")));
  OFFLOAD_ERR(
      printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
                                 "Native Vector Width For Char"));
  OFFLOAD_ERR(
      printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
                                 "Native Vector Width For Short"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(S, D,
                                         OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
                                         "Native Vector Width For Int"));
  OFFLOAD_ERR(
      printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
                                 "Native Vector Width For Long"));
  OFFLOAD_ERR(
      printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
                                 "Native Vector Width For Float"));
  OFFLOAD_ERR(printDeviceValue<uint32_t>(
      S, D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
      "Native Vector Width For Double"));
  OFFLOAD_ERR(
      printDeviceValue<uint32_t>(S, D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
                                 "Native Vector Width For Half"));

  return OL_SUCCESS;
}

template <typename T>
ol_result_t getPlatformValue(ol_platform_handle_t Plat, ol_platform_info_t Info,
                             T &Val) {
  OFFLOAD_ERR(olGetPlatformInfo(Plat, Info, sizeof(Val), &Val));
  return OL_SUCCESS;
}

template <typename T>
ol_result_t getDeviceValue(ol_device_handle_t Dev, ol_device_info_t Info,
                           T &Val) {
  OFFLOAD_ERR(olGetDeviceInfo(Dev, Info, sizeof(Val), &Val));
  return OL_SUCCESS;
}

template <typename T>
ol_result_t getPlatformValueStr(ol_platform_handle_t Plat,
                                ol_platform_info_t Info, std::string &Val) {
  size_t Size;
  OFFLOAD_ERR(olGetPlatformInfoSize(Plat, Info, &Size));
  std::vector<char> Bytes(Size);
  OFFLOAD_ERR(olGetPlatformInfo(Plat, Info, Size, Bytes.data()));
  Val = std::string(Bytes.data());
  return OL_SUCCESS;
}

template <typename T>
ol_result_t getDeviceValueStr(ol_device_handle_t Dev, ol_device_info_t Info,
                              std::string &Val) {
  size_t Size;
  OFFLOAD_ERR(olGetDeviceInfoSize(Dev, Info, &Size));
  std::vector<char> Bytes(Size);
  OFFLOAD_ERR(olGetDeviceInfo(Dev, Info, Size, Bytes.data()));
  Val = std::string(Bytes.data());
  return OL_SUCCESS;
}

json::Object printDeviceJSON(ol_device_handle_t D, uint32_t ID) {
  json::Object Device;
  Device["Device ID"] = ID;
  ol_platform_handle_t Platform = nullptr;
  if (olGetDeviceInfo(D, OL_DEVICE_INFO_PLATFORM, sizeof(Platform), &Platform))
    return Device;

  std::string PlatformName;
  getPlatformValueStr<const char *>(Platform, OL_PLATFORM_INFO_NAME,
                                    PlatformName);
  Device["Platform Name"] = PlatformName;

  std::string PlatformVendor;
  getPlatformValueStr<const char *>(Platform, OL_PLATFORM_INFO_VENDOR_NAME,
                                    PlatformVendor);
  Device["Platform Vendor Name"] = PlatformVendor;

  std::string PlatformVersion;
  getPlatformValueStr<const char *>(Platform, OL_PLATFORM_INFO_VERSION,
                                    PlatformVersion);
  Device["Platform Version"] = PlatformVersion;

  ol_platform_backend_t Backend = OL_PLATFORM_BACKEND_UNKNOWN;
  getPlatformValue(Platform, OL_PLATFORM_INFO_BACKEND, Backend);
  Device["Platform Backend"] = toString(Backend);

  std::string Name;
  getDeviceValueStr<const char *>(D, OL_DEVICE_INFO_NAME, Name);
  Device["Name"] = Name;

  std::string ProductName;
  getDeviceValueStr<const char *>(D, OL_DEVICE_INFO_PRODUCT_NAME, ProductName);
  Device["Product Name"] = ProductName;

  std::string UID;
  getDeviceValueStr<const char *>(D, OL_DEVICE_INFO_UID, UID);
  Device["UID"] = UID;

  ol_device_type_t Type = OL_DEVICE_TYPE_GPU; // Default or unknown
  getDeviceValue(D, OL_DEVICE_INFO_TYPE, Type);
  Device["Type"] = toString(Type);

  std::string DriverVersion;
  getDeviceValueStr<const char *>(D, OL_DEVICE_INFO_DRIVER_VERSION,
                                  DriverVersion);
  Device["Driver Version"] = DriverVersion;

  uint32_t MaxWorkGroupSize = 0;
  getDeviceValue(D, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE, MaxWorkGroupSize);
  Device["Max Work Group Size"] = MaxWorkGroupSize;

  ol_dimensions_t MaxWorkGroupSizePerDim = {0, 0, 0};
  getDeviceValue(D, OL_DEVICE_INFO_MAX_WORK_GROUP_SIZE_PER_DIMENSION,
                 MaxWorkGroupSizePerDim);
  Device["Max Work Group Size Per Dimension"] = toJSON(MaxWorkGroupSizePerDim);

  uint32_t MaxWorkSize = 0;
  getDeviceValue(D, OL_DEVICE_INFO_MAX_WORK_SIZE, MaxWorkSize);
  Device["Max Work Size"] = MaxWorkSize;

  ol_dimensions_t MaxWorkSizePerDim = {0, 0, 0};
  getDeviceValue(D, OL_DEVICE_INFO_MAX_WORK_SIZE_PER_DIMENSION,
                 MaxWorkSizePerDim);
  Device["Max Work Size Per Dimension"] = toJSON(MaxWorkSizePerDim);

  uint32_t VendorID = 0;
  getDeviceValue(D, OL_DEVICE_INFO_VENDOR_ID, VendorID);
  Device["Vendor ID"] = VendorID;

  uint32_t NumComputeUnits = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NUM_COMPUTE_UNITS, NumComputeUnits);
  Device["Num Compute Units"] = NumComputeUnits;

  uint32_t MaxClockFrequency = 0;
  getDeviceValue(D, OL_DEVICE_INFO_MAX_CLOCK_FREQUENCY, MaxClockFrequency);
  Device["Max Clock Frequency"] = MaxClockFrequency;

  uint32_t MemoryClockRate = 0;
  getDeviceValue(D, OL_DEVICE_INFO_MEMORY_CLOCK_RATE, MemoryClockRate);
  Device["Memory Clock Rate"] = MemoryClockRate;

  uint32_t AddressBits = 0;
  getDeviceValue(D, OL_DEVICE_INFO_ADDRESS_BITS, AddressBits);
  Device["Address Bits"] = AddressBits;

  uint64_t MaxMemAllocSize = 0;
  getDeviceValue(D, OL_DEVICE_INFO_MAX_MEM_ALLOC_SIZE, MaxMemAllocSize);
  Device["Max Mem Allocation Size"] = MaxMemAllocSize;

  uint64_t GlobalMemSize = 0;
  getDeviceValue(D, OL_DEVICE_INFO_GLOBAL_MEM_SIZE, GlobalMemSize);
  Device["Global Mem Size"] = GlobalMemSize;

  uint64_t WorkGroupSharedMemSize = 0;
  getDeviceValue(D, OL_DEVICE_INFO_WORK_GROUP_LOCAL_MEM_SIZE,
                 WorkGroupSharedMemSize);
  Device["Work Group Shared Mem Size"] = WorkGroupSharedMemSize;

  ol_device_fp_capability_flags_t SingleFPConfig = 0;
  getDeviceValue(D, OL_DEVICE_INFO_SINGLE_FP_CONFIG, SingleFPConfig);
  Device["Single Precision Floating Point Capability"] = toJSON(SingleFPConfig);

  ol_device_fp_capability_flags_t DoubleFPConfig = 0;
  getDeviceValue(D, OL_DEVICE_INFO_DOUBLE_FP_CONFIG, DoubleFPConfig);
  Device["Double Precision Floating Point Capability"] = toJSON(DoubleFPConfig);

  ol_device_fp_capability_flags_t HalfFPConfig = 0;
  getDeviceValue(D, OL_DEVICE_INFO_HALF_FP_CONFIG, HalfFPConfig);
  Device["Half Precision Floating Point Capability"] = toJSON(HalfFPConfig);

  uint32_t NativeVectorWidthChar = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_CHAR,
                 NativeVectorWidthChar);
  Device["Native Vector Width For Char"] = NativeVectorWidthChar;

  uint32_t NativeVectorWidthShort = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_SHORT,
                 NativeVectorWidthShort);
  Device["Native Vector Width For Short"] = NativeVectorWidthShort;

  uint32_t NativeVectorWidthInt = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_INT,
                 NativeVectorWidthInt);
  Device["Native Vector Width For Int"] = NativeVectorWidthInt;

  uint32_t NativeVectorWidthLong = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_LONG,
                 NativeVectorWidthLong);
  Device["Native Vector Width For Long"] = NativeVectorWidthLong;

  uint32_t NativeVectorWidthFloat = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_FLOAT,
                 NativeVectorWidthFloat);
  Device["Native Vector Width For Float"] = NativeVectorWidthFloat;

  uint32_t NativeVectorWidthDouble = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_DOUBLE,
                 NativeVectorWidthDouble);
  Device["Native Vector Width For Double"] = NativeVectorWidthDouble;

  uint32_t NativeVectorWidthHalf = 0;
  getDeviceValue(D, OL_DEVICE_INFO_NATIVE_VECTOR_WIDTH_HALF,
                 NativeVectorWidthHalf);
  Device["Native Vector Width For Half"] = NativeVectorWidthHalf;

  return Device;
}

ol_result_t printRoot(std::ostream &S) {
  OFFLOAD_ERR(olInit(nullptr));
  S << "Liboffload Version: " << OL_VERSION_MAJOR << "." << OL_VERSION_MINOR
    << "." << OL_VERSION_PATCH << "\n";

  std::vector<ol_device_handle_t> Devices;
  OFFLOAD_ERR(olIterateDevices(
      [](ol_device_handle_t Device, void *UserData) {
        reinterpret_cast<decltype(Devices) *>(UserData)->push_back(Device);
        return true;
      },
      &Devices));

  S << "Num Devices: " << Devices.size() << "\n";

  if (JSON) {
    json::Object Root;
    std::string Version = std::to_string(OL_VERSION_MAJOR) + "." +
                          std::to_string(OL_VERSION_MINOR) + "." +
                          std::to_string(OL_VERSION_PATCH);
    Root["Liboffload Version"] = Version;
    Root["Num Devices"] = (int64_t)Devices.size();

    json::Array JsonDevices;
    for (uint32_t DevIdx = 0; DevIdx < Devices.size(); ++DevIdx) {
      JsonDevices.push_back(printDeviceJSON(Devices[DevIdx], DevIdx));
    }
    Root["Devices"] = std::move(JsonDevices);

    std::string Processed;
    {
      llvm::raw_string_ostream OS(Processed);
      OS << json::Value(std::move(Root));
    }
    S << Processed << "\n";
  } else {
    for (auto &D : Devices) {
      S << "\n";
      OFFLOAD_ERR(printDevice(S, D));
    }
  }

  OFFLOAD_ERR(olShutDown());
  return OL_SUCCESS;
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "llvm-offload-device-info");
  auto Err = printRoot(std::cout);

  if (Err) {
    std::cerr << "[Liboffload error " << Err->Code << "]: " << Err->Details
              << "\n";
    return 1;
  }
  return 0;
}
