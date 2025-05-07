// These defines are from bsd/sys/reason.h:
#include <stdint.h>
#include <string.h>

extern void abort_with_payload(uint32_t reason_namespace, uint64_t reason_code,
                               void *payload, uint32_t payload_size,
                               const char *reason_string,
                               uint64_t reason_flags);

extern void abort_with_reason(uint32_t reason_namespace, uint64_t reason_code,
                              const char *reason_string, uint64_t reason_flags);

#define OS_REASON_FLAG_FROM_USERSPACE 0x4
#define OS_REASON_FLAG_NO_CRASH_REPORT 0x1
#define OS_REASON_FLAG_ONE_TIME_FAILURE 0x80

#define MY_REASON_FLAGS                                                        \
  OS_REASON_FLAG_FROM_USERSPACE | OS_REASON_FLAG_NO_CRASH_REPORT |             \
      OS_REASON_FLAG_ONE_TIME_FAILURE
#define OS_REASON_TEST 5

int main(int argc, char **argv) {
  const char *reason_string = "This is the reason string";
  const char *payload_string = "This is a payload that happens to be a string";
  size_t payload_string_len = strlen(payload_string) + 1;
  if (argc == 1) // Stop here before abort
    abort_with_payload(OS_REASON_TEST, 100, (void *)payload_string,
                       payload_string_len, reason_string, MY_REASON_FLAGS);
  else
    abort_with_reason(OS_REASON_TEST, 100, reason_string, MY_REASON_FLAGS);

  return 0;
}
