#===- SPIRVBasicBlock.cpp - SPIR-V Basic Block ------------------*- Bash -*-===#
#
#                     The LLVM/SPIRV Translator
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
# Copyright (c) 2018 Pierre Moreau All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal with the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimers.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimers in the documentation
# and/or other materials provided with the distribution.
# Neither the names of Advanced Micro Devices, Inc., nor the names of its
# contributors may be used to endorse or promote products derived from this
# Software without specific prior written permission.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
# THE SOFTWARE.
#
#===------------------------------------------------------------------------===#

MODIFIED_FILES=$(git diff --name-only master | grep -E ".*\.(cpp|cc|c\+\+|cxx|c|h|hpp)$")
FILES_TO_CHECK=$(echo "${MODIFIED_FILES}" | grep -v -E "Mangler/*|runtime/*|libSPIRV/(OpenCL.std.h|spirv.hpp)$")
CPP_FILES=$(find . -regex "\./\(lib\|tools\)/.*\.cpp" | grep -v -E "Mangler/*|runtime/*")
CPP_FILES="${CPP_FILES//$'\n'/ }"

if [ -z "${FILES_TO_CHECK}" ]; then
  echo "No source code to check for tidying."
  exit 0
fi

TIDY_DIFF=$(git diff -U0 master -- ${FILES_TO_CHECK} | ./utils/clang-tidy-diff.py -p1 -- "${CPP_FILES}" 2> /dev/null)

if [ "${TIDY_DIFF}" = "No relevant changes found." ]; then
  echo "${TIDY_DIFF}"
  exit 0
elif [ -z "${TIDY_DIFF}" ]; then
  echo "All source code in PR properly tidied."
  exit 0
else
  echo "Found tidying errors!"
  echo "${TIDY_DIFF}"
  exit 1
fi
