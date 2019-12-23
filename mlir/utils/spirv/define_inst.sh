#!/bin/bash
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script for defining a new op using SPIR-V spec from the Internet.
#
# Run as:
# ./define_inst.sh <filename> <inst_category> (<opname>)*

# <filename> is required, which is the file name of MLIR SPIR-V op definitions
# spec.
# <inst_category> is required. It can be one of
# (Op|ArithmeticOp|LogicalOp|ControlFlowOp|StructureOp). Based on the
# inst_category the file SPIRV<inst_category>s.td is updated with the
# instruction definition. If <opname> is missing, this script updates existing
# ones in SPIRV<inst_category>s.td

# For example:
# ./define_inst.sh SPIRVArithmeticOps.td ArithmeticOp OpIAdd
# ./define_inst.sh SPIRVLogicalOps.td LogicalOp OpFOrdEqual
set -e

file_name=$1
inst_category=$2

case $inst_category in
  Op | ArithmeticOp | LogicalOp | CastOp | ControlFlowOp | StructureOp | AtomicUpdateOp | AtomicUpdateWithValueOp)
  ;;
  *)
    echo "Usage : " $0 "<filename> <inst_category> (<opname>)*"
    echo "<filename> is the file name of MLIR SPIR-V op definitions spec"
    echo "<inst_category> must be one of " \
      "(Op|ArithmeticOp|LogicalOp|CastOp|ControlFlowOp|StructureOp|AtomicUpdateOp)"
    exit 1;
  ;;
esac

shift
shift

current_file="$(readlink -f "$0")"
current_dir="$(dirname "$current_file")"

python3 ${current_dir}/gen_spirv_dialect.py \
  --op-td-path \
  ${current_dir}/../../include/mlir/Dialect/SPIRV/${file_name} \
  --inst-category $inst_category --new-inst "$@"

${current_dir}/define_opcodes.sh "$@"

