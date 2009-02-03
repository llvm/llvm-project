// RUN: %llvmgcc -S %s -o - | grep {i8 1}
// PR2603

struct A {
  char num_fields;
};

struct B {
  char a, b[1];
};

const struct A Foo = {
  (char *)(&( (struct B *)(16) )->b[0]) - (char *)(16)
};
