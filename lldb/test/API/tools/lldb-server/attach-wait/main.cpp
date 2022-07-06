#include <thread>

int main()
{
  // Wait to be attached.
  std::this_thread::sleep_for(std::chrono::minutes(1));
  return 0;
}
