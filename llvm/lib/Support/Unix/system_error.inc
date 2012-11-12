//===- llvm/Support/Unix/system_error.inc - Unix error_code ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides the Unix specific implementation of the error_code
// and error_condition classes.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//=== WARNING: Implementation here must contain only generic UNIX code that
//===          is guaranteed to work on *all* UNIX variants.
//===----------------------------------------------------------------------===//

using namespace llvm;

std::string
_system_error_category::message(int ev) const {
  return _do_message::message(ev);
}

error_condition
_system_error_category::default_error_condition(int ev) const {
#ifdef ELAST
  if (ev > ELAST)
    return error_condition(ev, system_category());
#endif  // ELAST
  return error_condition(ev, generic_category());
}
