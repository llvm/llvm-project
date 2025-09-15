#include <cstdio>
#include <cstdlib>
#include <stdint.h>
#include <string>

// For more information about libc++'s std::string ABI, see:
//
//   https://joellaity.com/2020/01/31/string.html

// A corrupt string which hits the SSO code path, but has an invalid size.
static struct {
#if _LIBCPP_ABI_VERSION == 1
  // Set the size of this short-mode string to 116. Note that in short mode,
  // the size is encoded as `size << 1`.
  unsigned char size = 232;

  // 23 garbage bytes for the inline string payload.
  char inline_buf[23] = {0};
#else  // _LIBCPP_ABI_VERSION == 1
  // Like above, but data comes first, and use bitfields to indicate size.
  char inline_buf[23] = {0};
  unsigned char size : 7 = 116;
  unsigned char is_long : 1 = 0;
#endif // #if _LIBCPP_ABI_VERSION == 1
} garbage_string_short_mode;

// A corrupt libcxx string in long mode with a payload that contains a utf8
// sequence that's inherently too long.
static unsigned char garbage_utf8_payload1[] = {
    250, // This means that we expect a 5-byte sequence, this is invalid. LLDB
         // should fall back to ASCII printing.
    250, 250, 250};
static struct {
#if _LIBCPP_ABI_VERSION == 1
  uint64_t cap = 5;
  uint64_t size = 4;
  unsigned char *data = &garbage_utf8_payload1[0];
#else  // _LIBCPP_ABI_VERSION == 1
  unsigned char *data = &garbage_utf8_payload1[0];
  uint64_t size = 4;
  uint64_t cap : 63 = 4;
  uint64_t is_long : 1 = 1;
#endif // #if _LIBCPP_ABI_VERSION == 1
} garbage_string_long_mode1;

// A corrupt libcxx string in long mode with a payload that contains a utf8
// sequence that's too long to fit in the buffer.
static unsigned char garbage_utf8_payload2[] = {
    240, // This means that we expect a 4-byte sequence, but the buffer is too
         // small for this. LLDB should fall back to ASCII printing.
    240};
static struct {
#if _LIBCPP_ABI_VERSION == 1
  uint64_t cap = 3;
  uint64_t size = 2;
  unsigned char *data = &garbage_utf8_payload2[0];
#else  // _LIBCPP_ABI_VERSION == 1
  unsigned char *data = &garbage_utf8_payload2[0];
  uint64_t size = 2;
  uint64_t cap : 63 = 3;
  uint64_t is_long : 1 = 1;
#endif // #if _LIBCPP_ABI_VERSION == 1
} garbage_string_long_mode2;

// A corrupt libcxx string which has an invalid size (i.e. a size greater than
// the capacity of the string).
static struct {
#if _LIBCPP_ABI_VERSION == 1
  uint64_t cap = 5;
  uint64_t size = 7;
  const char *data = "foo";
#else  // _LIBCPP_ABI_VERSION == 1
  const char *data = "foo";
  uint64_t size = 7;
  uint64_t cap : 63 = 5;
  uint64_t is_long : 1 = 1;
#endif // #if _LIBCPP_ABI_VERSION == 1
} garbage_string_long_mode3;

// A corrupt libcxx string in long mode with a payload that would trigger a
// buffer overflow.
static struct {
#if _LIBCPP_ABI_VERSION == 1
  uint64_t cap = 5;
  uint64_t size = 2;
  uint64_t data = 0xfffffffffffffffeULL;
#else  // _LIBCPP_ABI_VERSION == 1
  uint64_t data = 0xfffffffffffffffeULL;
  uint64_t size = 2;
  uint64_t cap : 63 = 5;
  uint64_t is_long : 1 = 1;
#endif // #if _LIBCPP_ABI_VERSION == 1
} garbage_string_long_mode4;

int main() {
  std::string garbage1, garbage2, garbage3, garbage4, garbage5;
  if (sizeof(std::string) == sizeof(garbage_string_short_mode))
    memcpy((void *)&garbage1, &garbage_string_short_mode, sizeof(std::string));
  if (sizeof(std::string) == sizeof(garbage_string_long_mode1))
    memcpy((void *)&garbage2, &garbage_string_long_mode1, sizeof(std::string));
  if (sizeof(std::string) == sizeof(garbage_string_long_mode2))
    memcpy((void *)&garbage3, &garbage_string_long_mode2, sizeof(std::string));
  if (sizeof(std::string) == sizeof(garbage_string_long_mode3))
    memcpy((void *)&garbage4, &garbage_string_long_mode3, sizeof(std::string));
  if (sizeof(std::string) == sizeof(garbage_string_long_mode4))
    memcpy((void *)&garbage5, &garbage_string_long_mode4, sizeof(std::string));

  std::puts("// Set break point at this line.");
  return 0;
}
