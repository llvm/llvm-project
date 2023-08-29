#include <chrono>
#include <iostream>

int main() {
  // break here
  std::chrono::nanoseconds ns{1};
  std::chrono::microseconds us{12};
  std::chrono::milliseconds ms{123};
  std::chrono::seconds s{1234};
  std::chrono::minutes min{12345};
  std::chrono::hours h{123456};

  std::chrono::days d{654321};
  std::chrono::weeks w{54321};
  std::chrono::months m{4321};
  std::chrono::years y{321};

  std::cout << "break here\n";
}
