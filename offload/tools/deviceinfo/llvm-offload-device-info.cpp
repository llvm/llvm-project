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

#include <OffloadAPI.h>
#include <iostream>
#include <vector>

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

template <>
void doWrite<ol_platform_backend_t>(std::ostream &S,
                                    ol_platform_backend_t &&Val) {
  switch (Val) {
  case OL_PLATFORM_BACKEND_UNKNOWN:
    S << "UNKNOWN";
    break;
  case OL_PLATFORM_BACKEND_CUDA:
    S << "CUDA";
    break;
  case OL_PLATFORM_BACKEND_AMDGPU:
    S << "AMDGPU";
    break;
  case OL_PLATFORM_BACKEND_HOST:
    S << "HOST";
    break;
  default:
    S << "<< INVALID >>";
    break;
  }
}
template <>
void doWrite<ol_device_type_t>(std::ostream &S, ol_device_type_t &&Val) {
  switch (Val) {
  case OL_DEVICE_TYPE_GPU:
    S << "GPU";
    break;
  case OL_DEVICE_TYPE_CPU:
    S << "CPU";
    break;
  case OL_DEVICE_TYPE_HOST:
    S << "HOST";
    break;
  default:
    S << "<< INVALID >>";
    break;
  }
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
    OFFLOAD_ERR(olGetDeviceInfo(Dev, Info, sizeof(Val), Val.data()));
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

ol_result_t printRoot(std::ostream &S) {
  OFFLOAD_ERR(olInit());
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

  for (auto &D : Devices) {
    S << "\n";
    OFFLOAD_ERR(printDevice(S, D));
  }

  OFFLOAD_ERR(olShutDown());
  return OL_SUCCESS;
}

int main(int argc, char **argv) {
  auto Err = printRoot(std::cout);

  if (Err) {
    std::cerr << "[Liboffload error " << Err->Code << "]: " << Err->Details
              << "\n";
    return 1;
  }
  return 0;
}
