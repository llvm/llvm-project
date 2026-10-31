//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Option/LibraryOptions.td"

def DirectXMCOptions : OptionsStruct<prefix = "dx-">;

defm : BoolField<"dx-Zss", "false",
  "Compute Shader Hash considering source information">;
defm : BoolField<"dx-embed-debug", "false",
  "Embed PDB in shader container">;
defm : BoolField<"dx-pdb-in-private", "false",
  "Store PDB in private user data">;
defm : ValueField<"dx-pdb-path", "StringRef", "",
  "Write debug information to the given file, or automatically named file in directory when ending in '/'">,
  MetaVarName<"<filename>">;
// Disabling the pass makes it easier to test other passes related to DXIL
// resources using llc.
defm : BoolField<"dx-remove-unused-resources", "true",
  "Run the dxil-remove-unused-resources pass">;
defm : BoolField<"dx-slim-debug", "false",
  "Generate slim PDB without ILDB part">;
defm : BoolField<"dx-source-in-debug-module", "false",
  "Embed source code into debug module on DirectX target">;
defm : BoolField<"dx-strip-debug", "false",
  "Strip debug information from shader bytecode">;

// TODO: Remove the old spellings after 2027-04.
def : AliasOf<"disable-dxil-remove-unused-resources", "dx-remove-unused-resources=false">;
