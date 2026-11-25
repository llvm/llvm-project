// RUN: rm -rf %t && split-file %s %t && cd %t

// Check backslash escaping regardless of the build's default path separators.
// RUN: env LLVM_WINDOWS_PREFER_FORWARD_SLASH=0 llvm-tblgen \
// RUN:   -d out.d -o 'out #$.td' 'main #$.td'
// RUN: FileCheck %s --input-file=out.d --match-full-lines

// CHECK: out\ \#$$.td: main\ \#$$.td file\ with\ escape\\\ before\ spaces.td file\ with\ spaces.td file\#with\#hash.td file$$with$$dollar.td file\with\escape.td normal-file.td

// With forward slashes preferred, Windows converts the backslashes before
// escaping. Other platforms preserve them as literal filename characters.
// RUN: env LLVM_WINDOWS_PREFER_FORWARD_SLASH=1 llvm-tblgen \
// RUN:   -d out.d -o 'out #$.td' 'main #$.td'
// RUN: FileCheck %s --input-file=out.d --match-full-lines \
// RUN:   %if system-windows %{ --check-prefix=FORWARD %}

// FORWARD: out\ \#$$.td: main\ \#$$.td file\ with\ escape/\ before\ spaces.td file\ with\ spaces.td file\#with\#hash.td file$$with$$dollar.td file/with/escape.td normal-file.td

//--- normal-file.td
class NormalClass {}

//--- file with spaces.td
class SpaceClass {}

//--- file#with#hash.td
class HashClass {}

//--- file$with$dollar.td
class DollarClass {}

//--- file\with\escape.td
class EscapeClass {}

//--- file with escape\ before spaces.td
class EscapeBeforeSpacesClass {}

//--- main #$.td
include "normal-file.td"
include "file with spaces.td"
include "file#with#hash.td"
include "file$with$dollar.td"
// backslash itself needs escaping
include "file\\with\\escape.td"
include "file with escape\\ before spaces.td"

def Normal : NormalClass;
def Spaces : SpaceClass;
def Hash : HashClass;
def Dollar : DollarClass;
def Escape : EscapeClass;
def EscapeBeforeSpaces : EscapeBeforeSpacesClass;
