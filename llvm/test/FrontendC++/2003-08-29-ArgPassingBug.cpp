
// RUN: %llvmgcc -xc++ -c -o /dev/null %s |& not grep WARNING

struct iterator {
  iterator();
  iterator(const iterator &I);
};

iterator foo(const iterator &I) { return I; }

void test() {
  foo(iterator());
}
