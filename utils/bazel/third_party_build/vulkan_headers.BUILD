# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

# Exports all headers but defining VK_NO_PROTOTYPES to disable the
# inclusion of C function prototypes. Useful if dynamically loading
# all symbols via dlopen/etc.
# Not all headers are hermetic, so they are just included as textual
# headers to disable additional validation.
cc_library(
    name = "vulkan_headers_no_prototypes",
    defines = ["VK_NO_PROTOTYPES"],
    includes = ["include"],
    textual_hdrs = glob(["include/vulkan/*.h"]),
)

# Exports all headers, including C function prototypes. Useful if statically
# linking against the Vulkan SDK.
# Not all headers are hermetic, so they are just included as textual
# headers to disable additional validation.
cc_library(
    name = "vulkan_headers",
    includes = ["include"],
    textual_hdrs = glob(["include/vulkan/*.h"]),
)
