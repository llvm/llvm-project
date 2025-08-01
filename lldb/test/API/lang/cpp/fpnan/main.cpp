#include <limits>

int main() {
  float fnan = std::numeric_limits<float>::quiet_NaN();
  float fdenorm = std::numeric_limits<float>::denorm_min();

  // Set break point at this line.
}
