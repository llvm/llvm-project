// RUN: %llvmgcc -xc++ -c -o - %s | llvm-dis | grep _ZN11AccessFlags6strlenEv

struct AccessFlags {
  void strlen();
};

void AccessFlags::strlen() { }

