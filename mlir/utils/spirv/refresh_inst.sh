#!/bin/bash
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script for refreshing all defined SPIR-V ops using SPIR-V spec from the
# Internet.
#
# Run as:
# ./refresh_inst.sh

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

spirv_ir_include_dir="${current_dir}/../../include/mlir/Dialect/SPIRV/IR/"

for file in "${spirv_ir_include_dir}"/*; do
  file_name="$(basename $file)"
  if [[ $file_name == "SPIRVOps.td" ||
        $file_name == "SPIRVCLOps.td" ||
        $file_name == "SPIRVGLOps.td" ]]; then
    continue
  fi
  if [[ $file_name =~ SPIRV.*Ops.td ]]; then
    echo "--- refreshing $file_name ---"
    "${current_dir}/define_inst.sh" ${file_name} Op
  fi
done

