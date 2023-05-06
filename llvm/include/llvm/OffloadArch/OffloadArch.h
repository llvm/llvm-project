//===-------------- offload-arch/OffloadArch.h ----------*- C++ header -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef __LLVM_OFFLOAD_OFFLOADARCH_H__
#define __LLVM_OFFLOAD_OFFLOADARCH_H__

#include <string>
#include <vector>
#include <cstdint>

#define MAXPATHSIZE 512

// These search phrases in /sys/bus/pci/devices/*/uevent are found even if
// the device driver is not running.
#define AMDGPU_SEARCH_PHRASE "DRIVER=amdgpu"
#define AMDGPU_PCIID_PHRASE "PCI_ID=1002:"
#define NVIDIA_SEARCH_PHRASE "DRIVER=nvidia"
#define NVIDIA_PCIID_PHRASE "PCI_ID=10DE:"

///
/// Called by libomptarget runtime to get runtime capabilities.
int getRuntimeCapabilities(char *offload_arch_output_buffer,
                           size_t offload_arch_output_buffer_size);

/// Get the vendor specified softeare capabilities of the current runtime
/// The input vendor id selects the vendor function to call.
std::string getVendorCapabilities(uint16_t vid, uint16_t devid, std::string oa);

/// Get the AMD specific software capabilities of the current runtime
std::string getAMDGPUCapabilities(uint16_t vid, uint16_t devid, std::string oa);
/// Get the Nvidia specific software capabilities of the current runtime
std::string getNVPTXCapabilities(uint16_t vid, uint16_t devid, std::string oa);

///  return requirements for each offload image in an application binary
std::vector<std::string> getOffloadArchFromBinary(const std::string &fn);

///  return all offloadable pci-ids found in the system
std::vector<std::string> getAllPCIIds();
///  return all offloadable pci-ids for a given vendor
std::vector<std::string> getPCIIds(const char *driver_search_phrase,
                                   const char *pci_id_search_phrase);

///  lookup function to return all pci-ids for an input codename
std::vector<std::string> lookupCodename(std::string lookup_codename);

///  lookup function to return all pci-ids for an input offload_arch
std::vector<std::string> lookupOffloadArch(std::string lookup_offload_arch);

/// get the offload arch for VendorId-DeviceId
std::string getOffloadArch(uint16_t VendorID, uint16_t DeviceID);

/// get the vendor specified codename VendorId-DeviceId
std::string getCodename(uint16_t VendorID, uint16_t DeviceID);

/// get the compilation triple for VendorId-DeviceId
std::string getTriple(uint16_t VendorID, uint16_t DeviceID);

/// Utility to return contents of a file as a string
std::string getFileContents(std::string fname);

///  \return true if the system only has devices with architecture \in arch
///  false otherwise
bool isHomogeneousSystemOf(std::string arch);

#endif
