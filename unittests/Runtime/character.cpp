// Basic sanity tests of CHARACTER API; exhaustive testing will be done
// in Fortran.

#include "../../runtime/character.h"
#include "testing.h"
#include <cstring>

using namespace Fortran::runtime;

static void AppendAndPad(std::size_t limit) {
  char x[8];
  std::size_t xLen{0};
  std::memset(x, 0, sizeof x);
  xLen = RTNAME(CharacterAppend1)(x, limit, xLen, "abc", 3);
  xLen = RTNAME(CharacterAppend1)(x, limit, xLen, "DE", 2);
  RTNAME(CharacterPad1)(x, limit, xLen);
  if (xLen > limit) {
    Fail() << "xLen " << xLen << ">" << limit << '\n';
  }
  if (x[limit]) {
    Fail() << "x[" << limit << "]='" << x[limit] << "'\n";
    x[limit] = '\0';
  }
  if (std::memcmp(x, "abcDE   ", limit)) {
    Fail() << "x = '" << x << "'\n";
  }
}

static void TestCharCompare(const char *x, const char *y, std::size_t xBytes,
    std::size_t yBytes, int expect) {
  int cmp{RTNAME(CharacterCompareScalar1)(x, y, xBytes, yBytes)};
  if (cmp != expect) {
    char buf[2][8];
    std::memset(buf, 0, sizeof buf);
    std::memcpy(buf[0], x, xBytes);
    std::memcpy(buf[1], y, yBytes);
    Fail() << "compare '" << buf[0] << "'(" << xBytes << ") to '" << buf[1]
           << "'(" << yBytes << "), got " << cmp << ", should be " << expect
           << '\n';
  }
}

static void Compare(const char *x, const char *y, std::size_t xBytes,
    std::size_t yBytes, int expect) {
  TestCharCompare(x, y, xBytes, yBytes, expect);
  TestCharCompare(y, x, yBytes, xBytes, -expect);
}

int main() {
  StartTests();
  for (std::size_t j{0}; j < 8; ++j) {
    AppendAndPad(j);
  }
  Compare("abc", "abc", 3, 3, 0);
  Compare("abc", "def", 3, 3, -1);
  Compare("ab ", "abc", 3, 2, 0);
  Compare("abc", "abc", 2, 3, -1);
  return EndTests();
}
