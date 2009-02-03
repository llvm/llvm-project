// RUN: %llvmgxx -S %s -o - | llvm-as -f -o /dev/null

// There is a HOLE in the derived2 object due to not wanting to place the two
// baseclass instances at the same offset!

struct baseclass {};

class derived1 : public baseclass {
  void * NodePtr;
};

class derived2 : public baseclass {
  derived1 current;
};

derived2 RI;
