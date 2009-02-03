// RUN: %llvmgcc -xc -Os -c %s -o /dev/null
// PR1641

struct A {
  unsigned long l;
};

void bar(struct A *a);

void bork() {
  const unsigned long vcgt = 1234;
  struct A a = { vcgt };
  bar(&a);
}
