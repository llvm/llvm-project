namespace ns {

int i = 1;

namespace ns {

int i = 2;

} // namespace ns

} // namespace ns

namespace {
    int foo = 13;
}

namespace ns1 {
    namespace {
        int foo = 5;
    }
}

namespace {
    namespace ns2 {
        namespace {
            int foo = 7;
        }
    }
}

int main(int argc, char **argv) {
  int foo = 1;

  return foo + ::foo + ns1::foo + ns2::foo; // Set a breakpoint here
}
