#include <vector>

struct DbgInfoClass {
  std::vector<int> ints;
};

int main(int argc, char **argv) {
  std::vector<int> a = {3, 1, 2};

  // Create a std::vector of a class from debug info with one element.
  std::vector<DbgInfoClass> dbg_info_vec;
  dbg_info_vec.resize(1);
  // And that class has a std::vector of integers that comes from the C++
  // module.
  dbg_info_vec.back().ints.push_back(1);
  return 0; // Set break point at this line.
}
