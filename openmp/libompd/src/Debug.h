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
//    std::ostream& operator<<(std::ostream& os, Code code);
std::ostream &operator<<(std::ostream &os, Code code) {
  return os << "\033[" << static_cast<int>(code) << "m";
}
}

// class ColorOut: public std::ostream
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
  /*    template<typename T>
        const ColorOut& operator<< (const T* val) const
          {out << GdbColor::FG_RED << val << GdbColor::FG_DEFAULT;
          return *this;}
      template <class _CharT, class _Traits = std::char_traits<_CharT> >
        const ColorOut& operator<< ( const
              std::basic_ios<_CharT,_Traits>&
     (*pf)(std::basic_ios<_CharT,_Traits>&))const
          {out << GdbColor::FG_RED << pf << GdbColor::FG_DEFAULT;
          return *this;}
  */
  const ColorOut &operator<<(std::ostream &(*pf)(std::ostream &)) const {
    out << color << pf << GdbColor::FG_DEFAULT;
    return *this;
  }
};

static ColorOut dout(std::cout, GdbColor::FG_RED);
static ColorOut sout(std::cout, GdbColor::FG_GREEN);
static ColorOut hout(std::cout, GdbColor::FG_BLUE);

#endif /*GDB_DEBUG_H_*/
