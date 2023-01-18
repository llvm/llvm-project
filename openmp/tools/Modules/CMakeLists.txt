##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Install the FindOpenMPTarget module
#
##===----------------------------------------------------------------------===##


install(FILES "FindOpenMPTarget.cmake"
              DESTINATION "${DEVEL_PACKAGE}${OPENMP_INSTALL_LIBDIR}/cmake/openmp")
