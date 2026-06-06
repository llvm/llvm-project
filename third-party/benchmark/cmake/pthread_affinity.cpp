#include <pthread.h>
int main() {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int i = 0; i < CPU_SETSIZE; ++i) {
    CPU_SET(i, &set);
    CPU_CLR(i, &set);
  }
  pthread_t self = pthread_self();
  int ret;
  ret = pthread_getaffinity_np(self, sizeof(set), &set);
  if (ret != 0) return ret;
  ret = pthread_setaffinity_np(self, sizeof(set), &set);
  if (ret != 0) return ret;
  return 0;
}
