/*===-- CodeGenData.inc ----------------------------------------*- C++ -*-=== *\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/
/*
 * This is the main file that defines all the data structure, signature,
 * constant literals that are shared across compiler, host tools (reader/writer)
 * to support codegen data.
 *
\*===----------------------------------------------------------------------===*/

/* Helper macros.  */
#define CG_DATA_SIMPLE_QUOTE(x) #x
#define CG_DATA_QUOTE(x) CG_DATA_SIMPLE_QUOTE(x)

#ifdef CG_DATA_SECT_ENTRY
#define CG_DATA_DEFINED
CG_DATA_SECT_ENTRY(CG_outline, CG_DATA_QUOTE(CG_DATA_OUTLINE_COMMON),
                   CG_DATA_OUTLINE_COFF, "__DATA,")
CG_DATA_SECT_ENTRY(CG_merge, CG_DATA_QUOTE(CG_DATA_MERGE_COMMON),
                   CG_DATA_MERGE_COFF, "__DATA,")

#undef CG_DATA_SECT_ENTRY
#endif

/* section name strings common to all targets other
   than WIN32 */
#define CG_DATA_OUTLINE_COMMON __llvm_outline
#define CG_DATA_MERGE_COMMON __llvm_merge
/* Since cg data sections are not allocated, we don't need to
 * access them at runtime.
 */
#define CG_DATA_OUTLINE_COFF ".loutline"
#define CG_DATA_MERGE_COFF ".lmerge"

#ifdef _WIN32
/* Runtime section names and name strings.  */
#define CG_DATA_OUTLINE_SECT_NAME CG_DATA_OUTLINE_COFF
#define CG_DATA_MERGE_SECT_NAME CG_DATA_MERGE_COFF

#else
/* Runtime section names and name strings.  */
#define CG_DATA_OUTLINE_SECT_NAME CG_DATA_QUOTE(CG_DATA_OUTLINE_COMMON)
#define CG_DATA_MERGE_SECT_NAME CG_DATA_QUOTE(CG_DATA_MERGE_COMMON)

#endif

/* Indexed codegen data format version (start from 1). */
#define CG_DATA_INDEX_VERSION 3
