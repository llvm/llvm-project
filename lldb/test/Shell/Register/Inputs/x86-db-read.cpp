#include <cstdint>
#include <csignal>

uint8_t g_8w;
uint16_t g_16rw;
uint32_t g_32w;
uint32_t g_32rw;

int main() {
  ::raise(SIGSTOP);
  return 0;
}
