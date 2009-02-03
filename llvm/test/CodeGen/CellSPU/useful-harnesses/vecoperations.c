#include <stdio.h>

typedef unsigned char v16i8 __attribute__((ext_vector_type(16))); 
typedef short         v8i16 __attribute__((ext_vector_type(16))); 
typedef int           v4i32 __attribute__((ext_vector_type(4))); 
typedef float         v4f32 __attribute__((ext_vector_type(4))); 
typedef long long     v2i64 __attribute__((ext_vector_type(2))); 
typedef double        v2f64 __attribute__((ext_vector_type(2))); 

void print_v16i8(const char *str, const v16i8 v) {
  union {
    unsigned char elts[16];
    v16i8 vec;
  } tv;
  tv.vec = v;
  printf("%s = { %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, "
                "%hhu, %hhu, %hhu, %hhu, %hhu, %hhu, %hhu, "
		"%hhu, %hhu }\n",
	str, tv.elts[0], tv.elts[1], tv.elts[2], tv.elts[3], tv.elts[4], tv.elts[5],
	tv.elts[6], tv.elts[7], tv.elts[8], tv.elts[9], tv.elts[10], tv.elts[11],
	tv.elts[12], tv.elts[13], tv.elts[14], tv.elts[15]);
}

void print_v16i8_hex(const char *str, const v16i8 v) {
  union {
    unsigned char elts[16];
    v16i8 vec;
  } tv;
  tv.vec = v;
  printf("%s = { 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, "
                "0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, 0x%02hhx, "
		"0x%02hhx, 0x%02hhx }\n",
	str, tv.elts[0], tv.elts[1], tv.elts[2], tv.elts[3], tv.elts[4], tv.elts[5],
	tv.elts[6], tv.elts[7], tv.elts[8], tv.elts[9], tv.elts[10], tv.elts[11],
	tv.elts[12], tv.elts[13], tv.elts[14], tv.elts[15]);
}

void print_v8i16_hex(const char *str, v8i16 v) {
  union {
    short elts[8];
    v8i16 vec;
  } tv;
  tv.vec = v;
  printf("%s = { 0x%04hx, 0x%04hx, 0x%04hx, 0x%04hx, 0x%04hx, "
                "0x%04hx, 0x%04hx, 0x%04hx }\n",
	str, tv.elts[0], tv.elts[1], tv.elts[2], tv.elts[3], tv.elts[4],
	tv.elts[5], tv.elts[6], tv.elts[7]);
}

void print_v4i32(const char *str, v4i32 v) {
  printf("%s = { %d, %d, %d, %d }\n", str, v.x, v.y, v.z, v.w);
}

void print_v4f32(const char *str, v4f32 v) {
  printf("%s = { %f, %f, %f, %f }\n", str, v.x, v.y, v.z, v.w);
}

void print_v2i64(const char *str, v2i64 v) {
  printf("%s = { %lld, %lld }\n", str, v.x, v.y);
}

void print_v2f64(const char *str, v2f64 v) {
  printf("%s = { %g, %g }\n", str, v.x, v.y);
}

/*----------------------------------------------------------------------*/

v16i8 v16i8_mpy(v16i8 v1, v16i8 v2) {
  return v1 * v2;
}

v16i8 v16i8_add(v16i8 v1, v16i8 v2) {
  return v1 + v2;
}

v4i32 v4i32_shuffle_1(v4i32 a) {
  v4i32 c2 = a.yzwx;
  return c2;
}

v4i32 v4i32_shuffle_2(v4i32 a) {
  v4i32 c2 = a.zwxy;
  return c2;
}

v4i32 v4i32_shuffle_3(v4i32 a) {
  v4i32 c2 = a.wxyz;
  return c2;
}

v4i32 v4i32_shuffle_4(v4i32 a) {
  v4i32 c2 = a.xyzw;
  return c2;
}

v4i32 v4i32_shuffle_5(v4i32 a) {
  v4i32 c2 = a.xwzy;
  return c2;
}

v4f32 v4f32_shuffle_1(v4f32 a) {
  v4f32 c2 = a.yzwx;
  return c2;
}

v4f32 v4f32_shuffle_2(v4f32 a) {
  v4f32 c2 = a.zwxy;
  return c2;
}

v4f32 v4f32_shuffle_3(v4f32 a) {
  v4f32 c2 = a.wxyz;
  return c2;
}

v4f32 v4f32_shuffle_4(v4f32 a) {
  v4f32 c2 = a.xyzw;
  return c2;
}

v4f32 v4f32_shuffle_5(v4f32 a) {
  v4f32 c2 = a.xwzy;
  return c2;
}

v2i64 v2i64_shuffle(v2i64 a) {
  v2i64 c2 = a.yx;
  return c2;
}

v2f64 v2f64_shuffle(v2f64 a) {
  v2f64 c2 = a.yx;
  return c2;
}

int main(void) {
  v16i8 v00 = { 0xf4, 0xad, 0x01, 0xe9, 0x51, 0x78, 0xc1, 0x8a,
                0x94, 0x7c, 0x49, 0x6c, 0x21, 0x32, 0xb2, 0x04 };
  v16i8 va0 = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10 };
  v16i8 va1 = { 0x11, 0x83, 0x4b, 0x63, 0xff, 0x90, 0x32, 0xe5,
                0x5a, 0xaa, 0x20, 0x01, 0x0d, 0x15, 0x77, 0x05 };
  v8i16 v01 = { 0x1a87, 0x0a14, 0x5014, 0xfff0,
                0xe194, 0x0184, 0x801e, 0x5940 };
  v4i32 v1 = { 1, 2, 3, 4 };
  v4f32 v2 = { 1.0, 2.0, 3.0, 4.0 };
  v2i64 v3 = { 691043ll, 910301513ll };
  v2f64 v4 = { 5.8e56, 9.103e-62 };

  puts("---- vector tests start ----");

  print_v16i8_hex("v00                        ", v00);
  print_v16i8_hex("va0                        ", va0);
  print_v16i8_hex("va1                        ", va1);
  print_v16i8_hex("va0 x va1                  ", v16i8_mpy(va0, va1));
  print_v16i8_hex("va0 + va1                  ", v16i8_add(va0, va1));
  print_v8i16_hex("v01                        ", v01);

  print_v4i32("v4i32_shuffle_1(1, 2, 3, 4)", v4i32_shuffle_1(v1));
  print_v4i32("v4i32_shuffle_2(1, 2, 3, 4)", v4i32_shuffle_2(v1));
  print_v4i32("v4i32_shuffle_3(1, 2, 3, 4)", v4i32_shuffle_3(v1));
  print_v4i32("v4i32_shuffle_4(1, 2, 3, 4)", v4i32_shuffle_4(v1));
  print_v4i32("v4i32_shuffle_5(1, 2, 3, 4)", v4i32_shuffle_5(v1));

  print_v4f32("v4f32_shuffle_1(1, 2, 3, 4)", v4f32_shuffle_1(v2));
  print_v4f32("v4f32_shuffle_2(1, 2, 3, 4)", v4f32_shuffle_2(v2));
  print_v4f32("v4f32_shuffle_3(1, 2, 3, 4)", v4f32_shuffle_3(v2));
  print_v4f32("v4f32_shuffle_4(1, 2, 3, 4)", v4f32_shuffle_4(v2));
  print_v4f32("v4f32_shuffle_5(1, 2, 3, 4)", v4f32_shuffle_5(v2));

  print_v2i64("v3                         ", v3);
  print_v2i64("v2i64_shuffle              ", v2i64_shuffle(v3));
  print_v2f64("v4                         ", v4);
  print_v2f64("v2f64_shuffle              ", v2f64_shuffle(v4));

  puts("---- vector tests end ----");

  return 0;
}
