// These are needed to make sure that the linker does not strip the parts of the
// C++ abi library that are necessary to execute the expressions in the
// debugger. It would be great if we did not need to do this, but the fact that
// LLDB cannot conjure up the abi library on demand is not relevant for testing
// top level expressions.
struct DummyA {};
struct DummyB : public virtual DummyA {};

int main() {
  DummyB b;
  (void)b; // Set breakpoint here
  return 0;
}
