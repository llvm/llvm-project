// RUN: not llvm-tblgen -gen-searchable-tables -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/TableGen/SearchableTable.td"

class SmallEnumEntry;

foreach I = 0-299 in {
  def Entry#I : SmallEnumEntry;
}

def SmallScopedEnum : GenericEnumClass {
  let FilterClass = "SmallEnumEntry";
  let Size = 8;
  // CHECK: [[@LINE-1]]:7: error: Scoped enum 'SmallScopedEnum' with size in bits of '8' is too small for the total number of entries '300'
  // CHECK-NEXT: let Size = 8;
}

