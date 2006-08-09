//===-- remove.c - The remove function for the LLVM libc Library --*- C -*-===//
// 
// This code is a modified form of the remove() function from the GNU C
// library.
//
// Modifications:
//  2005/11/28 - Added to LLVM tree.  Functions renamed to allow compilation.
//               Code to control symbol linkage types removed.
//
//===----------------------------------------------------------------------===//

/* ANSI C `remove' function to delete a file or directory.  POSIX.1 version.
   Copyright (C) 1995,96,97,2002 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, write to the Free
   Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
   02111-1307 USA.  */

#include <errno.h>
#include <stdio.h>
#include <unistd.h>

int
remove (const char * file)
{
  int save;

  save = errno;
  if (rmdir (file) == 0)
    return 0;
  else if (errno == ENOTDIR && unlink (file) == 0)
    {
      errno = (save);
      return 0;
    }

  return -1;
}

