//===------------------- offload-arch/OffloadArch.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
///
/// \file  amdgpu/vendor_specific_capabilities.cpp
///
/// Implementiton of getAMDGPUCapabilities() function for offload-arch tool.
/// This is only called with the -r flag to show all runtime capabilities that
/// would satisfy requirements of the compiled image.
///
//===---------------------------------------------------------------------===//

#include "llvm/OffloadArch/OffloadArch.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

// So offload-arch can be built without ROCm installed as a copy of hsa.h
// is stored with the tool in the vendor specific directory.  This combined
// with dynamic loading (at runtime) of "libhsa-runtime64.so" allows
// offload-arch to be built without the ROCm platform installed.  Of course hsa
// (rocr runtime) must be operational at runtime.
//
#include "hsa-subset.h"
#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <sys/stat.h>
#include <vector>

struct amdgpu_features_t {
  char *name_str;
  uint32_t workgroup_max_size;
  hsa_dim3_t grid_max_dim;
  uint64_t grid_max_size;
  uint32_t fbarrier_max_size;
  uint16_t workgroup_max_dim[3];
  bool def_rounding_modes[3];
  bool base_rounding_modes[3];
  bool mach_models[2];
  bool profiles[2];
  bool fast_f16;
};

// static pointers to dynamically loaded HSA functions used in this module.
static hsa_status_t (*_dl_hsa_init)();
static hsa_status_t (*_dl_hsa_shut_down)();
static hsa_status_t (*_dl_hsa_isa_get_info_alt)(hsa_isa_t, hsa_isa_info_t,
                                                void *);
static hsa_status_t (*_dl_hsa_agent_get_info)(hsa_agent_t, hsa_agent_info_t,
                                              void *);
static hsa_status_t (*_dl_hsa_iterate_agents)(
    hsa_status_t (*callback)(hsa_agent_t, void *), void *);
static hsa_status_t (*_dl_hsa_agent_iterate_isas)(
    hsa_agent_t, hsa_status_t (*callback)(hsa_isa_t, void *), void *);

// These two static vectors are created by HSA iterators and needed after
// iterators complete, so we save them statically.
static std::vector<amdgpu_features_t> AMDGPU_FEATUREs;
static std::vector<hsa_agent_t> HSA_AGENTs;

static std::string offload_arch_requested;
static bool first_call = true;

#define _return_on_err(err)                                                    \
  {                                                                            \
    if ((err) != HSA_STATUS_SUCCESS) {                                         \
      return (err);                                                            \
    }                                                                          \
  }

static hsa_status_t get_isa_info(hsa_isa_t isa, void *data) {
  hsa_status_t err;
  amdgpu_features_t isa_i;

  uint32_t name_len;
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &name_len);
  _return_on_err(err);
  isa_i.name_str = new char[name_len];
  if (isa_i.name_str == nullptr)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_i.name_str);
  _return_on_err(err);

  // Following fields are not currently used but offload-arch ABI extensions
  // may want to access them in future.
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_MACHINE_MODELS,
                                 isa_i.mach_models);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_PROFILES, isa_i.profiles);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES,
                                 isa_i.def_rounding_modes);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(
      isa, HSA_ISA_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES,
      isa_i.base_rounding_modes);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FAST_F16_OPERATION,
                                 &isa_i.fast_f16);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_DIM,
                                 &isa_i.workgroup_max_dim);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_WORKGROUP_MAX_SIZE,
                                 &isa_i.workgroup_max_size);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_DIM,
                                 &isa_i.grid_max_dim);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_GRID_MAX_SIZE,
                                 &isa_i.grid_max_size);
  _return_on_err(err);
  err = _dl_hsa_isa_get_info_alt(isa, HSA_ISA_INFO_FBARRIER_MAX_SIZE,
                                 &isa_i.fbarrier_max_size);
  _return_on_err(err);
  AMDGPU_FEATUREs.push_back(isa_i);
  return err;
}

void *_aot_dynload_hsa_runtime() {

  void *dlhandle = nullptr;
#ifndef _WIN32
  // First search in system library paths. Allows user to dynamically
  // load desired version of hsa runtime.
  dlhandle = dlopen("libhsa-runtime64.so", RTLD_NOW);

  // Return null if hsa runtime is not found in system paths and in
  // absolute locations.
  if (!dlhandle)
    return nullptr;

  // We could use real names of hsa functions but the _dl_ makes it clear
  // these are dynamically loaded
  *(void **)&_dl_hsa_init = dlsym(dlhandle, "hsa_init");
  *(void **)&_dl_hsa_shut_down = dlsym(dlhandle, "hsa_shut_down");
  *(void **)&_dl_hsa_isa_get_info_alt = dlsym(dlhandle, "hsa_isa_get_info_alt");
  *(void **)&_dl_hsa_agent_get_info = dlsym(dlhandle, "hsa_agent_get_info");
  *(void **)&_dl_hsa_iterate_agents = dlsym(dlhandle, "hsa_iterate_agents");
  *(void **)&_dl_hsa_agent_iterate_isas =
      dlsym(dlhandle, "hsa_agent_iterate_isas");
#endif
  return dlhandle;
}

std::string getAMDGPUCapabilities(uint16_t vid, uint16_t devid,
                                  std::string oa) {
  std::string amdgpu_capabilities;
  offload_arch_requested = oa;

  if (IsAmdDeviceAvailable()) {
    BindHsaMethodsAndInitHSA();
  }

  // Make sure that previous calls' results don't interfere
  AMDGPU_FEATUREs.clear();
  HSA_AGENTs.clear();

  std::vector<std::string> GPUs;
  hsa_status_t Status = _dl_hsa_iterate_agents(
      [](hsa_agent_t Agent, void *Data) {
        hsa_device_type_t DeviceType;
        hsa_status_t Status =
            _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);
        // continue only if device type is GPU
        if (Status != HSA_STATUS_SUCCESS || DeviceType != HSA_DEVICE_TYPE_GPU) {
          return Status;
        }
        std::vector<std::string> *GPUs =
            static_cast<std::vector<std::string> *>(Data);
        char GPUName[64];
        Status = _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, GPUName);
        if (Status != HSA_STATUS_SUCCESS)
          return Status;
        if (GPUName == offload_arch_requested) {
          GPUs->push_back(GPUName);
          HSA_AGENTs.push_back(Agent);
        }
        return HSA_STATUS_SUCCESS;
      },
      &GPUs);

  if (Status != HSA_STATUS_SUCCESS) {
    amdgpu_capabilities.append(" HSAERROR-AGENT_ITERATION");
    return amdgpu_capabilities;
  }
  if (GPUs.size() == 0) {
    amdgpu_capabilities.append("NOT-VISIBLE");
    return amdgpu_capabilities;
  }

  // Select first detected HSA agent
  // TODO Select the one matching the given PCI ID instead
  hsa_agent_t xagent = *HSA_AGENTs.begin();
  Status = _dl_hsa_agent_iterate_isas(xagent, get_isa_info, nullptr);
  if (Status == HSA_STATUS_ERROR_INVALID_AGENT) {
    amdgpu_capabilities.append(" HSAERROR-INVALID_AGENT");
    return amdgpu_capabilities;
  }

  // parse features from field name_str of last amdgpu_features_t found
  llvm::StringRef Target(AMDGPU_FEATUREs.rbegin()->name_str);
  auto TargetFeatures = Target.split(':');
  auto TripleOrGPU = TargetFeatures.first.rsplit('-');
  auto TargetID = Target.substr(Target.find(TripleOrGPU.second));
  amdgpu_capabilities.append(TargetID.str());

  // We cannot shutdown hsa or close dlhandle because
  // _aot_amd_capabilities could be called multiple times.

  return amdgpu_capabilities;
}

std::string getAMDGPUCapabilitiesForOffloadarch(std::string uuid) {
  std::string amdgpu_capabilities;

  if (HSA_AGENTs.empty())
    return amdgpu_capabilities;

  int isa_number = 0;

  for (auto Agent : HSA_AGENTs) {

    char agent_uuid[24];
    hsa_status_t Stat = _dl_hsa_agent_get_info(
        Agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID, &agent_uuid);

    if (Stat != HSA_STATUS_SUCCESS) {
      amdgpu_capabilities = "HSA_AGENT_GET_INFO_ERROR\n";
      return amdgpu_capabilities;
    }

    if (uuid.compare(agent_uuid) == 0) {
      Stat = _dl_hsa_agent_iterate_isas(Agent, get_isa_info, &isa_number);

      if (Stat != HSA_STATUS_SUCCESS) {
        amdgpu_capabilities = "HSA_AGENT_ITERATE_ISAS_ERROR\n";
        return amdgpu_capabilities;
      }

      llvm::StringRef TargetFeatures(AMDGPU_FEATUREs.back().name_str);
      auto Features = TargetFeatures.split(":");
      auto Triple = Features.first.rsplit('-');
      amdgpu_capabilities.append(Triple.second.data());
      break;
    }
  }
  return amdgpu_capabilities;
}

bool IsAmdDeviceAvailable() {
  // Check status of ROCk
  auto InitstateOrError =
      llvm::MemoryBuffer::getFile("/sys/module/amdgpu/initstate");
  if (std::error_code EC = InitstateOrError.getError()) {
    fprintf(stderr, "unable to open device!\n");
    return false;
  }

  llvm::StringRef FileContent = InitstateOrError.get()->getBuffer();

  if (FileContent.find_insensitive("live") != llvm::StringRef::npos) {
    return true;
  }

  fprintf(stderr, "No AMD Device(s) found!\n");
  return false;
}

void BindHsaMethodsAndInitHSA() {

  if (first_call) {
    void *dlhandle = _aot_dynload_hsa_runtime();
    if (!dlhandle) {
      fprintf(stderr, " HSAERROR - DIDN'T FOUND RUNTIME\n");
      abort();
    }

    assert(_dl_hsa_init != nullptr);
    hsa_status_t Status = _dl_hsa_init();
    if (Status != HSA_STATUS_SUCCESS) {
      fprintf(stderr, " HSAERROR - INITIALIZATION");
      abort();
    }
    first_call = false;
  }
}

std::vector<std::pair<std::string, std::string>> runHsaDetection() {
  std::vector<std::pair<std::string, std::string>> OffloadingGpus;
  HSA_AGENTs.clear();

  auto Err = _dl_hsa_iterate_agents(
      [](hsa_agent_t Agent, void *GpuData) {
        hsa_device_type_t DeviceType;
        hsa_status_t Stat =
            _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);
        if (Stat != HSA_STATUS_SUCCESS)
          return Stat;

        std::vector<std::pair<std::string, std::string>> *GpuVector =
            static_cast<std::vector<std::pair<std::string, std::string>> *>(
                GpuData);

        if (DeviceType == HSA_DEVICE_TYPE_GPU) {
          hsa_agent_feature_t Features;
          Stat =
              _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_FEATURE, &Features);
          if (Stat != HSA_STATUS_SUCCESS)
            return Stat;

          if (Features & HSA_AGENT_FEATURE_KERNEL_DISPATCH) {

            char hsa_name[64];
            Stat = _dl_hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, hsa_name);
            if (Stat != HSA_STATUS_SUCCESS)
              return Stat;

            char uuid[24];
            Stat = _dl_hsa_agent_get_info(
                Agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_UUID, &uuid);
            if (Stat != HSA_STATUS_SUCCESS)
              return Stat;

            GpuVector->emplace_back(hsa_name, uuid);
            HSA_AGENTs.push_back(Agent);
          }
        }
        return HSA_STATUS_SUCCESS;
      },
      &OffloadingGpus);

  if (Err != HSA_STATUS_SUCCESS)
    OffloadingGpus.emplace_back("HSA_ITERATE_AGENT_ERRROR\n", " ");

  return OffloadingGpus;
}
