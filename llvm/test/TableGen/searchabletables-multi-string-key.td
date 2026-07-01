// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s

// When SearchIndex has multiple string-typed key fields, the emitted
// comparator declares the local helper variables `StringRef LHSStr` and
// `StringRef RHSStr` once per field in the same scope, producing C++ that
// fails to compile due to variable redeclaration. See
// SearchableTableEmitter.cpp's emitComparator lambda.

include "llvm/TableGen/SearchableTable.td"

class MyTableEntry<string first, string second, int val> {
  string First = first;
  string Second = second;
  bits<8> Val = val;
}

def : MyTableEntry<"foo", "baz", 1>;
def : MyTableEntry<"foo", "quux", 2>;
def : MyTableEntry<"bar", "baz", 3>;

def MyTable : GenericTable {
  let FilterClass = "MyTableEntry";
  let Fields = ["First", "Second", "Val"];
}

def lookupMyTableByFirstAndSecond : SearchIndex {
  let Table = MyTable;
  let Key = ["First", "Second"];
}

// CHECK-LABEL: const MyTableEntry *lookupMyTableByFirstAndSecond(StringRef First, StringRef Second)
// CHECK:        struct Comp {
// CHECK-NEXT:     bool operator()(const IndexType &LHS, const KeyType &RHS) const {
// CHECK-NEXT:       StringRef LHSStr = MyTableStrings[LHS.First];
// CHECK-NEXT:       StringRef RHSStr = RHS.First;
// CHECK-NEXT:       int CmpFirst = LHSStr.compare(RHSStr);
// CHECK-NEXT:       if (CmpFirst < 0) return true;
// CHECK-NEXT:       if (CmpFirst > 0) return false;
// CHECK-NEXT:       StringRef LHSStr = MyTableStrings[LHS.Second];
// CHECK-NEXT:       StringRef RHSStr = RHS.Second;
// CHECK-NEXT:       int CmpSecond = LHSStr.compare(RHSStr);
// CHECK-NEXT:       if (CmpSecond < 0) return true;
// CHECK-NEXT:       if (CmpSecond > 0) return false;
// CHECK-NEXT:       return false;
// CHECK-NEXT:     }
// CHECK-NEXT:   };
