// Type punning: reinterpret the bits of a float as an integer. Both members
// span the same 4 bytes, so the observed values don't depend on endianness.
union FloatBits {
  float f;
  unsigned int bits;
};

// Basic aliasing: the same 4 bytes seen as different types and sizes. Reading
// the storage back through the smaller members depends on the host byte order.
union Basic {
  int n;
  unsigned short halves[2];
  unsigned char byte;
};

// A union whose storage is also described by a struct ("view" pattern).
struct Halves {
  short lo;
  short hi;
};
union WithStruct {
  int packed;
  struct Halves view;
};

// A union nested directly inside another union.
union Nested {
  int all;
  union {
    short a;
    short b;
  } inner;
};

// A struct with an anonymous union member; the union's members are accessed as
// if they were fields of the enclosing struct.
struct WithAnonUnion {
  int tag;
  union {
    int as_int;
    unsigned char as_bytes[4];
  };
};

int main() {
  union FloatBits fb;
  fb.f = 1.0f;

  union Basic basic;
  basic.n = 0x11223344;

  union WithStruct ws;
  ws.packed = 0x00020001;

  union Nested nested;
  nested.all = 0x00060005;

  struct WithAnonUnion anon;
  anon.tag = 42;
  anon.as_int = 0x04030201;

  return 0; // break here
}
