//===-- FromElfOpts.td - llvm-fromelf options  ---------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the command line options of llvm-fromelf.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

multiclass Eq<string name, string help> {
  def NAME : Separate<["--"], name>;
  def NAME #_eq : Joined<["--"], name #"=">,
                  Alias<!cast<Separate>(NAME)>,
                  HelpText<help>;
}

def help : Flag<["--"], "help">;

def h : Flag<["-"], "h">, Alias<help>;

def version : Flag<["--"], "version">,
              HelpText<"Print the version and exit">;

def V : Flag<["-"], "V">,
        Alias<version>,
        HelpText<"Alias for --version">;

defm output_target : Eq<"output-target", "Format of the output file">,
                     Values<"binary">;
def O : JoinedOrSeparate<["-"], "O">,
        Alias<output_target>,
        HelpText<"Alias for --output-target">;
