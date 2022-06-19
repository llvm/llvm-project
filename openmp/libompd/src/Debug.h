/*
 * Debug.h -- OMP debug
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <ostream>

#ifndef GDB_DEBUG_H_
#define GDB_DEBUG_H_

namespace GdbColor {
enum Code {
  FG_RED = 31,
  FG_GREEN = 32,
  FG_BLUE = 34,
  FG_DEFAULT = 39,
  BG_RED = 41,
  BG_GREEN = 42,
  BG_BLUE = 44,
  BG_DEFAULT = 49
};
inline std::ostream &operator<<(std::ostream &os, Code code) {
  return os << "\033[" << static_cast<int>(code) << "m";
}
} // namespace GdbColor

class ColorOut {
private:
  std::ostream &out;
  GdbColor::Code color;

public:
  ColorOut(std::ostream &_out, GdbColor::Code _color)
      : out(_out), color(_color) {}
  template <typename T> const ColorOut &operator<<(const T &val) const {
    out << color << val << GdbColor::FG_DEFAULT;
    return *this;
  }
  const ColorOut &operator<<(std::ostream &(*pf)(std::ostream &)) const {
    out << color << pf << GdbColor::FG_DEFAULT;
    return *this;
  }
};

static ColorOut dout(std::cout, GdbColor::FG_RED);
static ColorOut sout(std::cout, GdbColor::FG_GREEN);
static ColorOut hout(std::cout, GdbColor::FG_BLUE);

#endif /*GDB_DEBUG_H_*/
