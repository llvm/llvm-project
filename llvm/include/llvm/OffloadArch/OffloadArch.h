//===-------------- offload-arch/OffloadArch.h ----------*- C++ header -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef __LLVM_OFFLOAD_OFFLOADARCH_H__
#define __LLVM_OFFLOAD_OFFLOADARCH_H__

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <vector>

#define MAXPATHSIZE 512

// These search phrases in /sys/bus/pci/devices/*/uevent are found even if
// the device driver is not running.
#define AMDGPU_SEARCH_PHRASE "DRIVER=amdgpu"
#define AMDGPU_PCIID_PHRASE "PCI_ID=1002:"
#define NVIDIA_SEARCH_PHRASE "DRIVER=nvidia"
#define NVIDIA_PCIID_PHRASE "PCI_ID=10DE:"

/// Get the vendor specified software capabilities of the current runtime
/// The input vendor id selects the vendor function to call.
std::string
getVendorCapabilities(const std::pair<std::string, std::string> &offloadarch);

/// Get the AMD specific software capabilities of the current runtime
std::string getAMDGPUCapabilities(uint16_t vid, uint16_t devid,
                                  const std::string &oa);
std::string getAMDGPUCapabilitiesForOffloadarch(const std::string &uuid);

/// Get the Nvidia specific software capabilities of the current runtime
std::string getNVPTXCapabilities(uint16_t vid, uint16_t devid,
                                 const std::string &oa);

///  return requirements for each offload image in an application binary
std::vector<std::string> getOffloadArchFromBinary(const std::string &fn);

///  return all offloadable pci-ids found in the system
std::vector<std::pair<std::string, std::string>>
getAllPCIIds(bool hsa_detection);
///  return all offloadable pci-ids for a given vendor
std::vector<std::pair<std::string, std::string>>
getPCIIds(const char *driver_search_phrase, const char *pci_id_search_phrase);

/// return vendor specific offloadable GPUs found in the system Detection
/// without using PCIIds
bool IsAmdDeviceAvailable();
void BindHsaMethodsAndInitHSA();
std::vector<std::pair<std::string, std::string>>
getAmdGpuDevices(const char *driver_search_phrase,
                 const char *pci_id_search_phrase, bool hsa_detection);
std::vector<std::pair<std::string, std::string>> runHsaDetection();

///  lookup function to return all pci-ids for an input codename
std::vector<std::string> lookupCodename(const std::string &lookup_codename);

///  lookup function to return all pci-ids for an input offload_arch
std::vector<std::string> lookupOffloadArch(std::string lookup_offload_arch);

/// get the offload arch for VendorId-DeviceId
std::string getOffloadArch(uint16_t VendorID, uint16_t DeviceID);

/// get the vendor specified offloadarch
std::string getCodename(std::string offloadArch);

/// get the compilation triple for offloadarch
std::string getTriple(const std::string &offloadarch);

/// Utility to return contents of a file as a string
std::string getFileContents(const std::string &fname);
#endif
