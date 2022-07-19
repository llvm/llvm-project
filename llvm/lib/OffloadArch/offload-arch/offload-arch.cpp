//===-- offload-arch/OffloadArch.cpp ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of LLVM offload-arch tool and library (libLLVMOffloadArch.a)
/// The offload-arch tool has alias commands "amdgpu-offload-arch" and
/// "nvidia-arch". The alias commands are symbolic links to offload-arch.
///
/// With no options, offload-arch prints the offload-arch for the current
/// system to stdout. offload-arch has options for querying vendor specific
/// runtime capabilities and for querying the offload-arch of offload images
/// that are embedded into application binaries. See help text below for
/// descriptions of these options.  offload-arch uses a two table lookup
/// scheme to determine the offload-arch of supported offload devices.
/// The first table maps pci-id to vendor-specified codenames. Multiple pci-ids
/// can map to a single codename. The 2nd table maps these codenames to the
/// offload-arch needed for compilation using the --offload-arch clang driver
/// options. Multiple codenames can map to a single offload-arch. The
/// offload-arch tool can be used to dump information from these tables.
/// These tables are maintained by vendors and the community as part of LLVM.
/// The tables and source to support querying architecture-specific capabilities
/// are organized in vendor-specific directories such as amdgpu and nvidia,
/// so that offload-arch can be used as a consistent cross-platform tool for
/// managing offloading architectures. offload-arch is both an LLVM tool and
/// an LLVM Library (libLLVMOffloadArch.a). The library is so that runtimes
/// and/or the clang driver have an API to query the current environment
/// without resorting to execv type calls to the offload-arch tool.
///
//===----------------------------------------------------------------------===//

#include "llvm/OffloadArch/OffloadArch.h"

void aot_usage() {
  printf("\n\
   offload-arch: Print offload architecture(s) for current system, or\n\
                 print offload runtime capabilities of current system,\n\
                 or lookup information about offload architectures,\n\
                 or print offload requirements for an application binary\n\
\n\
   Usage:\n\
\n\
     offload-arch [ Options ] [ Optional lookup-value ]\n\
\n\
     With no options, offload-arch prints the value for the first visible\n\
     offload-arch in the system. This can be used by various clang\n\
     frontends. For example, to compile for openmp offloading on your current\n\
     system, invoke clang with the following command:\n\
        clang -fopenmp -fopenmp-targets=`offload-arch` foo.c\n\
\n\
     If an optional lookup-value is specified, offload-arch will\n\
     check if the value is either a valid offload-arch or a codename\n\
     and display associated values with that offload-arch or codename.\n\
     For example, this provides all information for offload-arch gfx906:\n\
\n\
     offload-arch gfx906 -v \n\
\n\
     Options:\n\
     -m  Print device code name (often found in pci.ids file)\n\
     -n  Print numeric pci-id\n\
     -t  Print clang offload triple to use for the offload arch.\n\
     -c  Print offload capabilities of the current system.\n\
	 This option is used by the language runtime to select an image\n\
	 when multiple offload images are availble in the binary.\n\
	 A capability must exist for each requirement of the selected image.\n\
         each compiled offload image built into an application binary file.\n\
     -a  Print values for all devices. Don't stop at first visible device.\n\
     -v  Verbose = -a -m -n -t  \n\
         For all devices, print codename, numeric value and triple\n\
\n\
     The options -a and -v will show the offload-arch for all pci-ids that could\n\
     offload, even if they are not visible. Otherwise, the options -m, -n, -t,\n\
     or no option will only show information for the first visible device.\n\
\n\
     Other Options:\n\
     -h  Print this help message\n\
     -f  <filename> Print offload requirements including offload-arch for\n\
         each offload image compiled into an application binary file.\n\
\n\
     There are aliases (symbolic links) 'amdgpu-offload-arch' and 'nvidia-arch'\n\
     to the offload-arch tool. These aliases return 1\n\
     if respectively, no AMD or no Nvidia GPUs are found.\n\
     These aliases are useful to determine if architecture-specific\n\
     offloading tests should be run, or to conditionally load \n\
     archecture-specific software.\n\
\n\
     Copyright (c) 2021 ADVANCED MICRO DEVICES, INC.\n\
\n\
");
  exit(1);
}

static bool AOT_get_first_capable_device;

int main(int argc, char **argv) {
  bool print_codename = false;
  bool print_numeric = false;
  bool print_runtime_capabilities = false;
  bool amdgpu_arch = false;
  bool nvidia_arch = false;
  AOT_get_first_capable_device = true;
  bool print_triple = false;
  std::string lookup_value;
  std::string a, input_filename;
  for (int argi = 0; argi < argc; argi++) {
    a = std::string(argv[argi]);
    if (argi == 0) {
      // look for arch-specific invocation with symlink
      amdgpu_arch = (a.find("amdgpu-offload-arch") != std::string::npos);
      nvidia_arch = (a.find("nvidia-arch") != std::string::npos);
    } else {
      if (a == "-n") {
        print_numeric = true;
      } else if (a == "-m") {
        print_codename = true;
      } else if (a == "-c") {
        print_runtime_capabilities = true;
      } else if (a == "-h") {
        aot_usage();
      } else if (a == "-a") {
        AOT_get_first_capable_device = false; // get all devices
      } else if (a == "-t") {
        print_triple = true;
      } else if (a == "-v") {
        AOT_get_first_capable_device = false; // get all devices
        print_codename = true;
        print_numeric = true;
        print_triple = true;
      } else if (a == "-f") {
        argi++;
        if (argi == argc) {
          fprintf(stderr, "ERROR: Missing filename for -f option\n");
          return 1;
        }
        input_filename = std::string(argv[argi]);
      } else {
        lookup_value = a;
      }
    }
  }

  std::vector<std::string> PCI_IDS;

  if (!input_filename.empty()) {
    PCI_IDS = getOffloadArchFromBinary(input_filename);
    if (PCI_IDS.empty())
      return 1;
    for (auto PCI_ID : PCI_IDS)
      printf("%s\n", PCI_ID.c_str());
    return 0;

  } else if (lookup_value.empty()) {
    // No lookup_value so get the current pci ids.
    // First check if invocation was arch specific.
    if (amdgpu_arch)
      PCI_IDS = getPCIIds(AMDGPU_SEARCH_PHRASE, AMDGPU_PCIID_PHRASE);
    else if (nvidia_arch)
      PCI_IDS = getPCIIds(NVIDIA_SEARCH_PHRASE, NVIDIA_PCIID_PHRASE);
    else
      PCI_IDS = getAllPCIIds();
  } else {
    if (print_runtime_capabilities) {
      fprintf(stderr, "ERROR: cannot lookup offload-arch/codename AND query\n");
      fprintf(stderr, "       active runtime capabilities (-c).\n");
      return 1;
    }
    PCI_IDS = lookupOffloadArch(lookup_value);
    if (PCI_IDS.empty())
      PCI_IDS = lookupCodename(lookup_value);
    if (PCI_IDS.empty()) {
      fprintf(stderr, "ERROR: Could not find \"%s\" in offload-arch tables\n",
              lookup_value.c_str());
      fprintf(stderr, "       as either an offload-arch or a codename.\n");
      return 1;
    }
  }

  if (PCI_IDS.empty()) {
    return 1;
  }

  int rc = 0;
  bool first_device_printed = false;
  for (auto PCI_ID : PCI_IDS) {
    if (AOT_get_first_capable_device && first_device_printed)
      break;
    unsigned vid32, devid32;
    sscanf(PCI_ID.c_str(), "%x:%x", &vid32, &devid32);
    uint16_t vid = vid32;
    uint16_t devid = devid32;
    std::string offload_arch = getOffloadArch(vid, devid);
    if (offload_arch.empty()) {
      fprintf(stderr, "ERROR: offload-arch not found for %x:%x.\n", vid, devid);
      rc = 1;
    } else {
      std::string xinfo;
      if (print_codename)
        xinfo.append(" ").append(getCodename(vid, devid));
      if (print_numeric)
        xinfo.append(" ").append(PCI_ID);
      if (print_triple)
        xinfo.append(" ").append(getTriple(vid, devid));
      if (print_runtime_capabilities || AOT_get_first_capable_device) {
        std::string caps = getVendorCapabilities(vid, devid, offload_arch);
        std::size_t found_loc = caps.find("NOT-VISIBLE");
        if (found_loc == std::string::npos) {
          if (print_runtime_capabilities) {
            xinfo.clear();
            xinfo = std::move(caps);
            printf("%s\n", xinfo.c_str());
          } else {
            printf("%s%s\n", offload_arch.c_str(), xinfo.c_str());
          }
          first_device_printed = true;
        }
      } else {
        printf("%s%s\n", offload_arch.c_str(), xinfo.c_str());
        first_device_printed = true;
      }
    }
  }
  if (!first_device_printed)
    rc = 1;
  return rc;
}
