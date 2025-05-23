# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

let

  pkgs = import (builtins.fetchTarball {
    url =
      "https://github.com/NixOS/nixpkgs/archive/064b8bf531fbfd5d6a3f46a3b5bcb8e6fd403e45.tar.gz";
    sha256 = "0xrlbh2lc0ijslxyckq50z40hsc234cla0cjhdff9kwiz5s5cmph";
  }) { };

  # Run the compile script from anywhere
  compile-llvm = pkgs.writeShellScriptBin "compile-llvm" ''
    #!/usr/bin/env bash
    set -euo pipefail

    # Get the root directory of the git repository
    ROOT_DIR=$(git rev-parse --show-toplevel)

    # Run the compile-llvm.sh script in the root directory
    "$ROOT_DIR/compile-llvm.sh" "$@"
  '';

  format-parasol = pkgs.writeShellScriptBin "format-parasol" ''
    #!/usr/bin/env bash
    set -euo pipefail

    # Get the root directory of the git repository
    ROOT_DIR=$(git rev-parse --show-toplevel)

    # Run the format-parasol.sh script in the root directory
    "$ROOT_DIR/format-parasol.sh" "$@"
  '';

in pkgs.mkShellNoCC {
  buildInputs = with pkgs; [ ninja diffoscopeMinimal compile-llvm format-parasol ];
}
