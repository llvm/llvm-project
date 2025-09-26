// RUN: rm -rf %t; split-file %s %t

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

//--- main.td
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

// RUN: llvm-tblgen -I %t -d %t.d -o %t.out %t/main.td
// RUN: FileCheck --input-file=%t.d %s

// CHECK-DAG: normal-file.td
// CHECK-DAG: file\ with\ spaces.td
// CHECK-DAG: file\#with\#hash.td
// CHECK-DAG: file$$with$$dollar.td
// CHECK-DAG: file\with\escape.td
// CHECK-DAG: file\ with\ escape\\\ before\ spaces.td
