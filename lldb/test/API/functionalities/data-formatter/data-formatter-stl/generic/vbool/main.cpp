#include <cstdio>
#include <vector>

int main() {
  std::vector<bool> vBool;

  // 0..=7
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 8..=15
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 16..=23
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 24..=31
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 32..=39
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 40..=47
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 48..=55
  vBool.push_back(true);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 56..=63
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 64..=71
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);
  vBool.push_back(true);
  vBool.push_back(true);
  vBool.push_back(false);
  vBool.push_back(true);

  // 72
  vBool.push_back(true);

  std::puts("// Set break point at this line.");
  return 0;
}
