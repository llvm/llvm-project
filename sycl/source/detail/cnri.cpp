//===-- cnri.cpp - SYCL common native runtime interface impl-----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CL/sycl/detail/cnri.h"
#include <CL/sycl/detail/common.hpp>

#include <assert.h>
#include <cstdint>

cl_int cnriSelectDeviceImage(cnri_context ctx, cnri_device_image **images,
                             cl_uint num_images,
                             cnri_device_image **selected_image) {
  // TODO dummy implementation.
  // Real implementaion will use the same mechanism OpenCL ICD dispatcher
  // uses. Somthing like:
  //   CNRI_VALIDATE_HANDLE_RETURN_HANDLE(ctx, CNRI_INVALID_CONTEXT);
  //     return context->dispatch->cnriSelectDeviceImage(
  //       ctx, images, num_images, selected_image);
  // where context->dispatch is set to the dispatch table provided by CNRI
  // plugin for platform/device the ctx was created for.

  *selected_image = num_images > 0 ? images[0] : nullptr;
  return CNRI_SUCCESS;
}
