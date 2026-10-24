// RUN: %liboffload-compilexx %s -o %t
// RUN: env LIBOFFLOAD_DEBUG=Sync %t %target_triple 2>&1 | %fcheck-generic

// The host plugin runs host functions synchronously, so it cannot keep a
// destroyed queue busy.
// REQUIRES: gpu
// REQUIRES: libomptarget-debug

// A queue destroyed while it still has work is kept, and handed to the next
// olCreateQueue once that work is done. Which queue a new handle gets is not
// visible through the API, so check the debug log.
//
// Usage: queue_reuse <target triple> [device index]. Runs on the given device
// (default: the first) of the backend matching the target.

#include <OffloadAPI.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#define CHECK_OL(Call)                                                         \
  do {                                                                         \
    ol_result_t Res = (Call);                                                  \
    if (Res && Res->Code != OL_ERRC_SUCCESS) {                                 \
      fprintf(stderr, "%s failed: %s\n", #Call, Res->Details);                 \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

// Keeps the queue busy until *Release is set.
static void block(void *Release) {
  while (!static_cast<std::atomic<bool> *>(Release)->load())
    std::this_thread::yield();
}

static ol_platform_backend_t getBackend(const char *Triple) {
  if (strstr(Triple, "nvptx"))
    return OL_PLATFORM_BACKEND_CUDA;
  if (strstr(Triple, "amdgcn"))
    return OL_PLATFORM_BACKEND_AMDGPU;
  if (strstr(Triple, "spirv"))
    return OL_PLATFORM_BACKEND_LEVEL_ZERO;
  return OL_PLATFORM_BACKEND_UNKNOWN;
}

struct DeviceQuery {
  ol_platform_backend_t Backend;
  int Index;
  ol_device_handle_t Found = nullptr;
};

static bool findDevice(ol_device_handle_t Device, void *Data) {
  auto *Query = static_cast<DeviceQuery *>(Data);
  ol_platform_handle_t Platform;
  ol_platform_backend_t Backend;
  if (olGetDeviceInfo(Device, OL_DEVICE_INFO_PLATFORM, sizeof(Platform),
                      &Platform) ||
      olGetPlatformInfo(Platform, OL_PLATFORM_INFO_BACKEND, sizeof(Backend),
                        &Backend))
    return true;
  if (Backend != Query->Backend || Query->Index-- > 0)
    return true;
  Query->Found = Device;
  return false;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <target triple> [device index]\n", argv[0]);
    return EXIT_FAILURE;
  }
  DeviceQuery Query{getBackend(argv[1]), argc > 2 ? atoi(argv[2]) : 0};
  CHECK_OL(olInit(nullptr));
  CHECK_OL(olIterateDevices(findDevice, &Query));
  ol_device_handle_t Device = Query.Found;
  if (!Device) {
    fprintf(stderr, "no matching device for %s\n", argv[1]);
    return EXIT_FAILURE;
  }
  ol_context_handle_t Context;
  CHECK_OL(olCreateContext(1, &Device, &Context));

  // Destroy a queue while it is blocked, so it is kept for reuse.
  std::atomic<bool> Release{false};
  ol_queue_handle_t Queue;
  ol_event_handle_t Done;
  CHECK_OL(olCreateQueue(Context, Device, &Queue));
  CHECK_OL(olLaunchHostFunction(Queue, block, &Release));
  CHECK_OL(olCreateEvent(Queue, OL_EVENT_FLAGS_NONE, &Done));
  CHECK_OL(olDestroyQueue(Queue));
  // CHECK: Keeping busy queue [[QUEUE:0x[0-9a-f]+]] for reuse

  // Let it finish; the next new queue must pick it up.
  Release = true;
  CHECK_OL(olSyncEvent(Done));
  CHECK_OL(olDestroyEvent(Done));
  CHECK_OL(olCreateQueue(Context, Device, &Queue));
  // CHECK: Reusing outstanding queue [[QUEUE]]
  CHECK_OL(olDestroyQueue(Queue));

  CHECK_OL(olDestroyContext(Context));
  CHECK_OL(olShutDown());
  // CHECK: PASS
  printf("PASS\n");
  return EXIT_SUCCESS;
}
