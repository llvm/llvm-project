//===-- InstallNameToolOpts.td - llvm-install-name-tool options  --------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the command line options of llvm-install-name.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def help : Flag<["--"], "help">;
def h : Flag<["-"], "h">, Alias<help>;

def add_rpath : Option<["-", "--"], "add_rpath", KIND_SEPARATE>,
                HelpText<"Add new rpath">;

def version : Flag<["--"], "version">,
              HelpText<"Print the version and exit.">;
