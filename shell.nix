let
  nixpkgs = fetchTarball "https://github.com/NixOS/nixpkgs/tarball/nixos-24.05";
  pkgs = import nixpkgs { config = {}; overlays = []; };
in


pkgs.mkShellNoCC {
  packages = with pkgs; [
    cmake
    ninja
    llvmPackages_latest.llvm
  ];
stdenv = pkgs.clangStdenv;
}
