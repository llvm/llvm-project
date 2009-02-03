// RUN: %llvmgcc -xc++ -S -o - %s | grep {struct.X::Y}
struct X {

  struct Y {
    Y();
  };

};

X::Y::Y() {

}
