##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#  Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#  Notified per clause 4(b) of the license.
#
##===----------------------------------------------------------------------===##
#
# Install the FindOpenMPTarget module
#
##===----------------------------------------------------------------------===##


install(FILES "FindOpenMPTarget.cmake"
              DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}/cmake/openmp")
