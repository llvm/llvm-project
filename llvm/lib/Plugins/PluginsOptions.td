//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def PluginsOptions : OptionsStruct;

// PassPlugin.cpp
let MetaVarName = "<plugin>,<arg>" in
defm plugin_args : UnsplitListField<"plugin-arg", "StringRef",
  "Pass <arg> to the pass plugin named <plugin>">;
