//===- llvm-omp-device-info.cpp - Obtain device info as seen from OpenMP --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that, by using Libomptarget, and the device
// plugins, list devices information as seen from the OpenMP Runtime.
//
//===----------------------------------------------------------------------===//

#include "omptarget.h"
#include <cstdio>

int main(int argc, char **argv) {
  __tgt_bin_desc EmptyDesc = {0, nullptr, nullptr, nullptr};
  __tgt_register_lib(&EmptyDesc);
  __tgt_init_all_rtls();

  for (int Dev = 0; Dev < omp_get_num_devices(); Dev++) {
    printf("Device (%d):\n", Dev);
    if (!__tgt_print_device_info(Dev))
      printf("    print_device_info not implemented\n");
    printf("\n");
  }

  __tgt_unregister_lib(&EmptyDesc);
  return 0;
}
