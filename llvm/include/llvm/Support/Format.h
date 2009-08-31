//===- Format.h - Efficient printf-style formatting for streams -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the format() function, which can be used with other
// LLVM subsystems to provide printf-style formatting.  This gives all the power
// and risk of printf.  This can be used like this (with raw_ostreams as an
// example):
//
//    OS << "mynumber: " << format("%4.5f", 1234.412) << '\n';
//
// Or if you prefer:
//
//  OS << format("mynumber: %4.5f\n", 1234.412);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_FORMAT_H
#define LLVM_SUPPORT_FORMAT_H

#include <cstdio>
#ifdef WIN32
#define snprintf _snprintf
#endif

namespace llvm {

/// format_object_base - This is a helper class used for handling formatted
/// output.  It is the abstract base class of a templated derived class.
class format_object_base {
protected:
  const char *Fmt;
  virtual void home(); // Out of line virtual method.

  /// snprint - Call snprintf() for this object, on the given buffer and size.
  virtual int snprint(char *Buffer, unsigned BufferSize) const = 0;

public:
  format_object_base(const char *fmt) : Fmt(fmt) {}
  virtual ~format_object_base() {}

  /// print - Format the object into the specified buffer.  On success, this
  /// returns the length of the formatted string.  If the buffer is too small,
  /// this returns a length to retry with, which will be larger than BufferSize.
  unsigned print(char *Buffer, unsigned BufferSize) const {
    assert(BufferSize && "Invalid buffer size!");

    // Print the string, leaving room for the terminating null.
    int N = snprint(Buffer, BufferSize);

    // VC++ and old GlibC return negative on overflow, just double the size.
    if (N < 0)
      return BufferSize*2;

    // Other impls yield number of bytes needed, not including the final '\0'.
    if (unsigned(N) >= BufferSize)
      return N+1;

    // Otherwise N is the length of output (not including the final '\0').
    return N;
  }
};

/// format_object1 - This is a templated helper class used by the format
/// function that captures the object to be formated and the format string. When
/// actually printed, this synthesizes the string into a temporary buffer
/// provided and returns whether or not it is big enough.
template <typename T>
class format_object1 : public format_object_base {
  T Val;
public:
  format_object1(const char *fmt, const T &val)
    : format_object_base(fmt), Val(val) {
  }

  virtual int snprint(char *Buffer, unsigned BufferSize) const {
    return snprintf(Buffer, BufferSize, Fmt, Val);
  }
};

/// format_object2 - This is a templated helper class used by the format
/// function that captures the object to be formated and the format string. When
/// actually printed, this synthesizes the string into a temporary buffer
/// provided and returns whether or not it is big enough.
template <typename T1, typename T2>
class format_object2 : public format_object_base {
  T1 Val1;
  T2 Val2;
public:
  format_object2(const char *fmt, const T1 &val1, const T2 &val2)
  : format_object_base(fmt), Val1(val1), Val2(val2) {
  }

  virtual int snprint(char *Buffer, unsigned BufferSize) const {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2);
  }
};

/// format_object3 - This is a templated helper class used by the format
/// function that captures the object to be formated and the format string. When
/// actually printed, this synthesizes the string into a temporary buffer
/// provided and returns whether or not it is big enough.
template <typename T1, typename T2, typename T3>
class format_object3 : public format_object_base {
  T1 Val1;
  T2 Val2;
  T3 Val3;
public:
  format_object3(const char *fmt, const T1 &val1, const T2 &val2,const T3 &val3)
    : format_object_base(fmt), Val1(val1), Val2(val2), Val3(val3) {
  }

  virtual int snprint(char *Buffer, unsigned BufferSize) const {
    return snprintf(Buffer, BufferSize, Fmt, Val1, Val2, Val3);
  }
};

/// format - This is a helper function that is used to produce formatted output.
/// This is typically used like:  OS << format("%0.4f", myfloat) << '\n';
template <typename T>
inline format_object1<T> format(const char *Fmt, const T &Val) {
  return format_object1<T>(Fmt, Val);
}

/// format - This is a helper function that is used to produce formatted output.
/// This is typically used like:  OS << format("%0.4f", myfloat) << '\n';
template <typename T1, typename T2>
inline format_object2<T1, T2> format(const char *Fmt, const T1 &Val1,
                                     const T2 &Val2) {
  return format_object2<T1, T2>(Fmt, Val1, Val2);
}

/// format - This is a helper function that is used to produce formatted output.
/// This is typically used like:  OS << format("%0.4f", myfloat) << '\n';
template <typename T1, typename T2, typename T3>
  inline format_object3<T1, T2, T3> format(const char *Fmt, const T1 &Val1,
                                           const T2 &Val2, const T3 &Val3) {
  return format_object3<T1, T2, T3>(Fmt, Val1, Val2, Val3);
}

} // end namespace llvm

#endif
