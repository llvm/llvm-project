#include <vector>

int main() {
  std::vector<bool> vBoolEmpty;

  std::vector<bool> vBoolSmall = {true,  false, true,  true, false,
                                  false, true,  false, true, true};

  // Make a bit vector that is larger than 64 bit.
  std::vector<bool> vBool = {
      // 0..=47: alternating false, true
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      // 48..=55: pattern breaks at 48
      true,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      // 56..=63: alternating again
      false,
      true,
      false,
      true,
      false,
      true,
      false,
      true,
      // 64..=71: pattern breaks at 68
      false,
      true,
      false,
      true,
      true,
      true,
      false,
      true,
      // 72
      true,
  };

  return 0; // break here
}
