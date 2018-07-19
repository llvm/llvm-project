//===- DIASupport.h - Common header includes for DIA ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Common defines and header includes for all LLVMDebugInfoPDBDIA.  The
// definitions here configure the necessary #defines and include system headers
// in the proper order for using DIA.
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_DIA_DIASUPPORT_H
#define LLVM_DEBUGINFO_PDB_DIA_DIASUPPORT_H

// Require at least Vista
#define NTDDI_VERSION NTDDI_VISTA
#define _WIN32_WINNT _WIN32_WINNT_VISTA
#define WINVER _WIN32_WINNT_VISTA
#ifndef NOMINMAX
#define NOMINMAX
#endif

// llvm/Support/Debug.h unconditionally #defines DEBUG as a macro.
// DIA headers #define it if it is not already defined, so we have
// an order of includes problem.  The real fix is to make LLVM use
// something less generic than DEBUG, such as LLVM_DEBUG(), but it's
// fairly prevalent.  So for now, we save the definition state and
// restore it.
#pragma push_macro("DEBUG")

// atlbase.h has to come before windows.h
#include <atlbase.h>
#include <windows.h>

// DIA headers must come after windows headers.
#include <cvconst.h>
#include <dia2.h>
#include <diacreate.h>

#pragma pop_macro("DEBUG")

#endif // LLVM_DEBUGINFO_PDB_DIA_DIASUPPORT_H
