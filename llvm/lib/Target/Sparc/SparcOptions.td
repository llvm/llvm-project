//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def SparcOptions : OptionsStruct<prefix = "sparc-">;

defm : BoolField<"disable-sparc-delay-filler", "false",
  "Disable the Sparc delay slot filler">;
defm : BoolField<"disable-sparc-leaf-proc", "false",
  "Disable Sparc leaf procedure optimization">;
defm : ValueField<"sparc-bpcc-offset-bits", "unsigned", "19",
  "Restrict range of BPcc/FBPfcc instructions (DEBUG)">;
defm : ValueField<"sparc-bpr-offset-bits", "unsigned", "16",
  "Restrict range of BPr instructions (DEBUG)">;
defm : BoolField<"sparc-enable-branch-relax", "true",
  "Relax out of range conditional branches">;
defm : BoolField<"sparc-reserve-app-registers", "false",
  "Reserve application registers (%g2-%g4)">;
