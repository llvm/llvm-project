// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

struct duration {
 duration operator/=(int c) {
  return *this;
  }
};

void a000090() {
  duration() /= 1;
}
