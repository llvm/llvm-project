#!/usr/bin/bash
# script to generate code for LLVM/SPIR-V translator based on khronos 
# header file spirv.hpp.
#


######################
#
# generate NameMap
#
######################

genNameMap() {
prefix=$1
echo "template<> inline void
SPIRVMap<$prefix, std::string>::init() {"

cat $spirvHeader | sed -n -e "/^ *${prefix}[^a-z]/s:^ *${prefix}\([^= ][^= ]*\)[= ][= ]*\([0x]*[0-9][0-9]*\).*:\1 \2:p"  | while read a b; do
  printf "  add(${prefix}%s, \"%s\");\n" $a $a
done

echo "}
SPIRV_DEF_NAMEMAP($prefix, SPIRV${prefix}NameMap)
"

}

###########################
#
# generate isValid function
#
###########################
genIsValid() {
prefix=$1
echo "inline bool
isValid(spv::$prefix V) {
  switch(V) {"

  cat $spirvHeader | sed -n -e "/^ *${prefix}[^a-z]/s:^ *${prefix}\([^= ][^= ]*\)[= ][= ]*\(.*\).*:\1 \2:p"  | while read a b; do
  if [[ $a == CapabilityNone ]]; then
    continue
  fi
  printf "    case ${prefix}%s:\n" $a
done

echo "      return true;
    default:
      return false;
  }
}
"
}
genMaskIsValid() {
prefix=$1
subprefix=`echo $prefix | sed -e "s:Mask::g"`
echo "inline bool
isValid$prefix(SPIRVWord Mask) {
  SPIRVWord ValidMask = 0u;"

  cat $spirvHeader | sed -n -e "/^ *${subprefix}[^a-z]/s:^ *${subprefix}\([^= ][^= ]*\)Mask[= ][= ]*\(.*\).*:\1 \2:p"  | while read a b; do
  if [[ $a == None ]]; then
    continue
  fi
  printf "  ValidMask |= ${subprefix}%sMask;\n" $a
done

echo "
  return (Mask & ~ValidMask) == 0;
}
"
}

##############################
#
# generate entries for td file
#
##############################
genTd() {
prefix=$1

if [[ $prefix == "Capability" ]]; then
  echo "class SPIRV${prefix}_ {"
else
  echo "def SPIRV${prefix} : Operand<i32> {
  let PrintMethod = \"printSPIRV${prefix}\";
"
fi

cat $spirvHeader | sed -n -e "/^ *${prefix}[^a-z]/s:^ *${prefix}\([^= ][^= ]*\)[= ][= ]*\([0xX]*[0-9a-fA-F][0-9a-fA-F]*\).*:\1 \2:p"  | while read a b; do
  if [[ $a == CapabilityNone ]]; then
    continue
  fi
  printf "  int %s = %s;\n" $a $b
done

if [[ $prefix == "Capability" ]]; then
  echo "}
def SPIRV${prefix} : SPIRV${prefix}_;
"
else 
  echo "}
"
fi
}

gen() {
type=$1
for prefix in SourceLanguage ExecutionModel AddressingModel MemoryModel ExecutionMode StorageClass Dim SamplerAddressingMode SamplerFilterMode ImageFormat \
  ImageChannelOrder ImageChannelDataType FPRoundingMode LinkageType AccessQualifier FunctionParameterAttribute Decoration BuiltIn Scope GroupOperation \
  KernelEnqueueFlags Capability Op; do
  if [[ "$type" == NameMap ]]; then
    genNameMap $prefix
  elif [[ "$type" == isValid ]]; then
    genIsValid $prefix
  elif [[ "$type" == td ]]; then
    genTd $prefix
  else
    echo "invalid type \"$type\"."
    exit
  fi
done
for prefix in ImageOperandsMask FPFastMathModeMask SelectionControlMask LoopControlMask FunctionControlMask MemorySemanticsMask MemoryAccessMask \
  KernelProfilingInfoMask; do
  if [[ "$type" == isValid ]]; then
    genMaskIsValid $prefix
  fi
done
}

####################
#
# main
#
####################

if [[ $# -ne 3 ]]; then
  echo "usage: gen_spirv path_to_spirv.hpp [NameMap|isValid|td] output_file"
  exit
fi

spirvHeader=$1
type=$2
outputFile=$3
includeGuard="`echo ${outputFile} | tr '[:lower:]' '[:upper:]' | sed -e 's/\./_/g'`_"

echo "//===- ${outputFile} - SPIR-V ${type} enums ----------------*- C++ -*-===//
//
//                     The LLVM/SPIRV Translator
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// Copyright (c) 2014 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the \"Software\"),
// to deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimers in the documentation
// and/or other materials provided with the distribution.
// Neither the names of Advanced Micro Devices, Inc., nor the names of its
// contributors may be used to endorse or promote products derived from this
// Software without specific prior written permission.
// THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.
//
//===----------------------------------------------------------------------===//
/// \\file
///
/// This file defines SPIR-V ${type} enums.
///
//===----------------------------------------------------------------------===//
// WARNING:
//
// This file has been generated using \`tools/spirv-tool/gen_spirv.bash\` and
// should not be modified manually. If the file needs to be updated, edit the
// script and any other source file instead, before re-generating this file.
//===----------------------------------------------------------------------===//

#ifndef ${includeGuard}
#define ${includeGuard}

#include \"spirv.hpp\"
#include \"SPIRVEnum.h\"

using namespace spv;

namespace SPIRV {
" > ${outputFile}

gen $type >> ${outputFile}

echo "} /* namespace SPIRV */

#endif /* ${includeGuard} */" >> ${outputFile}
