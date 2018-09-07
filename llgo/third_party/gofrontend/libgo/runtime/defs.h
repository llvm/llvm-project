/* defs.h -- runtime definitions for Go.

   Copyright 2009 The Go Authors. All rights reserved.
   Use of this source code is governed by a BSD-style
   license that can be found in the LICENSE file.  */

/* The gc library uses this file for system defines, and generates it
   automatically using the godefs program.  The logical thing to put
   here for gccgo would be #include statements for system header
   files.  We can't do that, though, because runtime.h #define's the
   standard types.  So we #include the system headers from runtime.h
   instead.  */
