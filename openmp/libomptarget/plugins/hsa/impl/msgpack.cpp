#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>

#include "msgpack.h"

namespace msgpack {

[[noreturn]] void internal_error() {
  printf("internal error\n");
  exit(1);
}

const char *type_name(type ty) {
  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case NAME:                                                                   \
    return #NAME;
#include "msgpack.def"
#undef X
  }
  internal_error();
}

unsigned bytes_used_fixed(msgpack::type ty) {
  using namespace msgpack;
  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case NAME:                                                                   \
    return WIDTH;
#include "msgpack.def"
#undef X
  }
  internal_error();
}

msgpack::type parse_type(unsigned char x) {

#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  if (x >= LOWER && x <= UPPER) {                                              \
    return NAME;                                                               \
  } else
#include "msgpack.def"
#undef X
  { internal_error(); }
}

template <typename T, typename R> R bitcast(T x) {
  static_assert(sizeof(T) == sizeof(R), "");
  R tmp;
  memcpy(&tmp, &x, sizeof(T));
  return tmp;
}
template int64_t bitcast<uint64_t, int64_t>(uint64_t);
} // namespace msgpack

// Helper functions for reading additional payload from the header
// Depending on the type, this can be a number of bytes, elements,
// key-value pairs or an embedded integer.
// Each takes a pointer to the start of the header and returns a uint64_t

namespace {
namespace payload {
uint64_t read_zero(const unsigned char *) { return 0; }

// Read the first byte and zero/sign extend it
uint64_t read_embedded_u8(const unsigned char *start) { return start[0]; }
uint64_t read_embedded_s8(const unsigned char *start) {
  int64_t res = msgpack::bitcast<uint8_t, int8_t>(start[0]);
  return msgpack::bitcast<int64_t, uint64_t>(res);
}

// Read a masked part of the first byte
uint64_t read_via_mask_0x1(const unsigned char *start) { return *start & 0x1u; }
uint64_t read_via_mask_0xf(const unsigned char *start) { return *start & 0xfu; }
uint64_t read_via_mask_0x1f(const unsigned char *start) {
  return *start & 0x1fu;
}

// Read 1/2/4/8 bytes immediately following the type byte and zero/sign extend
// Big endian format.
uint64_t read_size_field_u8(const unsigned char *from) {
  from++;
  return from[0];
}

// TODO: detect whether host is little endian or not, and whether the intrinsic
// is available. And probably use the builtin to test the diy
const bool use_bswap = false;

uint64_t read_size_field_u16(const unsigned char *from) {
  from++;
  if (use_bswap) {
    uint16_t b;
    memcpy(&b, from, 2);
    return __builtin_bswap16(b);
  } else {
    return (from[0] << 8u) | from[1];
  }
}
uint64_t read_size_field_u32(const unsigned char *from) {
  from++;
  if (use_bswap) {
    uint32_t b;
    memcpy(&b, from, 4);
    return __builtin_bswap32(b);
  } else {
    return (from[0] << 24u) | (from[1] << 16u) | (from[2] << 8u) |
           (from[3] << 0u);
  }
}
uint64_t read_size_field_u64(const unsigned char *from) {
  from++;
  if (use_bswap) {
    uint64_t b;
    memcpy(&b, from, 8);
    return __builtin_bswap64(b);
  } else {
    return ((uint64_t)from[0] << 56u) | ((uint64_t)from[1] << 48u) |
           ((uint64_t)from[2] << 40u) | ((uint64_t)from[3] << 32u) |
           (from[4] << 24u) | (from[5] << 16u) | (from[6] << 8u) |
           (from[7] << 0u);
  }
}

uint64_t read_size_field_s8(const unsigned char *from) {
  uint8_t u = read_size_field_u8(from);
  int64_t res = msgpack::bitcast<uint8_t, int8_t>(u);
  return msgpack::bitcast<int64_t, uint64_t>(res);
}
uint64_t read_size_field_s16(const unsigned char *from) {
  uint16_t u = read_size_field_u16(from);
  int64_t res = msgpack::bitcast<uint16_t, int16_t>(u);
  return msgpack::bitcast<int64_t, uint64_t>(res);
}
uint64_t read_size_field_s32(const unsigned char *from) {
  uint32_t u = read_size_field_u32(from);
  int64_t res = msgpack::bitcast<uint32_t, int32_t>(u);
  return msgpack::bitcast<int64_t, uint64_t>(res);
}
uint64_t read_size_field_s64(const unsigned char *from) {
  uint64_t u = read_size_field_u64(from);
  int64_t res = msgpack::bitcast<uint64_t, int64_t>(u);
  return msgpack::bitcast<int64_t, uint64_t>(res);
}
} // namespace payload
} // namespace

namespace msgpack {

payload_info_t payload_info(msgpack::type ty) {
  using namespace msgpack;
  switch (ty) {
#define X(NAME, WIDTH, PAYLOAD, LOWER, UPPER)                                  \
  case NAME:                                                                   \
    return payload::PAYLOAD;
#include "msgpack.def"
#undef X
  }
  internal_error();
}

} // namespace msgpack

const unsigned char *msgpack::skip_next_message(const unsigned char *start,
                                                const unsigned char *end) {
  class f : public functors_defaults<f> {};
  return handle_msgpack({start, end}, f());
}

namespace msgpack {
bool message_is_string(byte_range bytes, const char *needle) {
  bool matched = false;
  size_t needleN = strlen(needle);

  foronly_string(bytes, [=, &matched](size_t N, const unsigned char *str) {
    if (N == needleN) {
      if (memcmp(needle, str, N) == 0) {
        matched = true;
      }
    }
  });
  return matched;
}

void dump(byte_range bytes) {
  struct inner : functors_defaults<inner> {
    inner(unsigned indent) : indent(indent) {}
    const unsigned by = 2;
    unsigned indent = 0;

    void handle_string(size_t N, const unsigned char *bytes) {
      char *tmp = (char *)malloc(N + 1);
      memcpy(tmp, bytes, N);
      tmp[N] = '\0';
      printf("\"%s\"", tmp);
      free(tmp);
    }

    void handle_signed(int64_t x) { printf("%ld", x); }
    void handle_unsigned(uint64_t x) { printf("%lu", x); }

    const unsigned char *handle_array(uint64_t N, byte_range bytes) {
      printf("\n%*s[\n", indent, "");
      indent += by;

      for (uint64_t i = 0; i < N; i++) {
        indent += by;
        printf("%*s", indent, "");
        const unsigned char *next = handle_msgpack<inner>(bytes, {indent});
        printf(",\n");
        indent -= by;
        bytes.start = next;
        if (!next) {
          break;
        }
      }
      indent -= by;
      printf("%*s]", indent, "");

      return bytes.start;
    }

    const unsigned char *handle_map(uint64_t N, byte_range bytes) {
      printf("\n%*s{\n", indent, "");
      indent += by;

      for (uint64_t i = 0; i < 2 * N; i += 2) {
        const unsigned char *start_key = bytes.start;
        printf("%*s", indent, "");
        const unsigned char *end_key =
            handle_msgpack<inner>({start_key, bytes.end}, {indent});
        if (!end_key) {
          break;
        }

        printf(" : ");

        const unsigned char *start_value = end_key;
        const unsigned char *end_value =
            handle_msgpack<inner>({start_value, bytes.end}, {indent});

        if (!end_value) {
          break;
        }

        printf(",\n");
        bytes.start = end_value;
      }

      indent -= by;
      printf("%*s}", indent, "");

      return bytes.start;
    }
  };

  handle_msgpack<inner>(bytes, {0});
  printf("\n");
}

} // namespace msgpack
