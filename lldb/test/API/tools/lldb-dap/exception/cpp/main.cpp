#include <stdexcept>

void throw_some_string() {
  throw "fake message"; // thrown_exception.
}

int main(int argc, char const *argv[]) {
  try {
    throw_some_string();
  } catch (const char *) { // caught_exception.
    if (argc > 1)
      return 0;
  }

  throw std::invalid_argument("throwing exception for testing");
  return 0;
}
