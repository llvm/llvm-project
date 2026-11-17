#include <stdio.h>

// clang-format off
#include <windows.h>
#include <processthreadsapi.h>
// clang-format on

int main() {
  // break here
  HANDLE thread = GetCurrentThread();
  HRESULT hr = SetThreadDescription(thread, L"ThreadName");
  if (FAILED(hr)) {
    fprintf(stderr, "SetThreadDescription failed: 0x%08lx\n", hr);
    return 1;
  }

  printf("Thread name set successfully.\n"); // break here
  return 0;
}
