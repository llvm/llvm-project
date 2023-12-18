#include <vector>

struct HasVector {
  std::vector<int> v;
};

int main() {
  std::vector<int> vector = {42, 47};
  HasVector has_vector = {vector};
  return 0; // break here
}
