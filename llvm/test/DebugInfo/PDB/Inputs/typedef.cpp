// Build with clang -fno-rtti -g -O0 typedef.cpp
// Built with clang (22+) because MSVC does not output lf_alias for typedefs

void *__purecall = 0;

typedef unsigned char u8;
using i64 = long long;

int main() {
    u8 val = 15;
    i64 val2 = -1;

    return 0;
}
