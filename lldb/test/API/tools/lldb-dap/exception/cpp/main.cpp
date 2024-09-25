#include <stdexcept>

int main(int argc, char const *argv[]) {
  throw std::invalid_argument("throwing exception for testing");
  return 0;
}
