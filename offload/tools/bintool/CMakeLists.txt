# INTEL_CUSTOMIZATION
#
# INTEL CONFIDENTIAL
#
# Modifications, Copyright (C) 2021 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute, disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the
# License.
#
# end INTEL_CUSTOMIZATION
message(STATUS "Building the llvm-offload-bintool tool")

add_openmp_tool(
  llvm-offload-bintool llvm-offload-bintool.cpp
  ../../plugins-nextgen/common/src/OffloadError.cpp
)

llvm_update_compile_flags(llvm-offload-bintool)

target_link_libraries(llvm-offload-bintool PRIVATE
  LLVMOffload LLVMSupport LLVMObject
)
