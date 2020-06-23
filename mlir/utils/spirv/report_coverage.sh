#!/bin/bash

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script for reporting current level of SPIR-V spec instruction coverage in spv
# Dialect. It dumps to standard output a YAML string of current coverage.
#
# Run as:
# ./report_coverage.sh

set -e

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

python3 ${current_dir}/gen_spirv_dialect.py \
  --base-td-path ${current_dir}/../../include/mlir/Dialect/SPIRV/SPIRVBase.td \
  --gen-inst-coverage
