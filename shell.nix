{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    gcc.cc.lib
    binutils
    cmake
    ninja
    gnumake
    pkg-config
    which
    lld
    zlib
    libxml2
    libedit
    libpfm
    ncurses
    python3
  ];

  shellHook = ''
    export CC=${pkgs.gcc}/bin/gcc
    export CXX=${pkgs.gcc}/bin/g++

    export LD_LIBRARY_PATH=${
      pkgs.lib.makeLibraryPath [
        pkgs.gcc.cc.lib
        pkgs.zlib
        pkgs.libxml2
        pkgs.libedit
        pkgs.ncurses
      ]
    }:$LD_LIBRARY_PATH

    echo "LLVM dev shell ready"
  '';
}
