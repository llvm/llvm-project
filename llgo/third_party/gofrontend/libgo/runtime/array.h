/* array.h -- the open array type for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

#ifndef LIBGO_ARRAY_H
#define LIBGO_ARRAY_H

/* An open array is an instance of this structure.  */

struct __go_open_array
{
  /* The elements of the array.  In use in the compiler this is a
     pointer to the element type.  */
  void* __values;
  /* The number of elements in the array.  Note that this is "int",
     not "size_t".  The language definition says that "int" is large
     enough to hold the size of any allocated object.  Using "int"
     saves 8 bytes per slice header on a 64-bit system with 32-bit
     ints.  */
  intgo __count;
  /* The capacity of the array--the number of elements that can fit in
     the __VALUES field.  */
  intgo __capacity;
};

#endif /* !defined(LIBGO_ARRAY_H) */
