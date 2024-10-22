#===--------------------------------------------------------------------===//
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for details.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===--------------------------------------------------------------------===//

add_library(llvm-libc-common-utilities INTERFACE)
# TODO: Reorganize the libc shared section so that it can be included without
# adding the root "libc" directory to the include path.
target_include_directories(llvm-libc-common-utilities INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../../../libc)
target_compile_definitions(llvm-libc-common-utilities INTERFACE LIBC_NAMESPACE=__llvm_libc_common_utils)
target_compile_features(llvm-libc-common-utilities INTERFACE cxx_std_17)
