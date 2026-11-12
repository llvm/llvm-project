#include "baz.h"

#include <math.h>

int baz(int &j, std::mutex &mutex, std::condition_variable &cv) {
  std::unique_lock<std::mutex> lock(mutex);
  cv.wait(lock, [&j] { return j == 42 * 42; });
  int k = sqrt(j);
  return k; // break here
}
