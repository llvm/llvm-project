# ===- CMakeLists.txt - llvm-elf2bin's build script -----------------------===//
# 
#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
# ===----------------------------------------------------------------------===//

set(LLVM_LINK_COMPONENTS
  Object
  Option
  Support)

set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(Elf2BinOptsTableGen)

add_llvm_tool(llvm-elf2bin
  llvm-elf2bin.cpp
  elf.cpp
  bin.cpp
  hex.cpp
  DEPENDS
  Elf2BinOptsTableGen
  GENERATE_DRIVER
)
