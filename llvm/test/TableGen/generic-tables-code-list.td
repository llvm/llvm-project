// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s --implicit-check-not=StringTable --implicit-check-not=getEntryStr
// RUN: llvm-tblgen -gen-searchable-tables -D EXPLICIT_TYPE -I %p/../../include %s | FileCheck %s --implicit-check-not=StringTable --implicit-check-not=getEntryStr

include "llvm/TableGen/SearchableTable.td"

// Encode a list in a code field without generating unused string storage or
// accessors, using either code literals or an explicit type override. Each key
// still has a single table entry.
class Entry<bits<8> key, list<int> values> {
  bits<8> Key = key;
#ifdef EXPLICIT_TYPE
  string Values = "ValueList<" #
      !interleave(!foreach(v, values, !cast<string>(v)), ", ") # ">";
#else
  code Values = [{ValueList<}] #
      !interleave(!foreach(v, values, !cast<string>(v)), ", ") # [{>}];
#endif
}

def : Entry<1, [3]>;
def : Entry<2, [4, 101]>;

def Entries : GenericTable {
  let FilterClass = "Entry";
  let Fields = ["Key", "Values"];
#ifdef EXPLICIT_TYPE
  string TypeOf_Values = "code";
#endif
  let PrimaryKey = ["Key"];
  let PrimaryKeyName = "lookupEntry";
}

// CHECK-LABEL: GET_Entries_DECL
// CHECK-NEXT: const Entry *lookupEntry(uint8_t Key);
// CHECK-NEXT: #endif
// CHECK-LABEL: GET_Entries_IMPL
// CHECK-NEXT: constexpr Entry Entries[] = {
// CHECK-NEXT:   { 0x1, ValueList<3> }, // 0
// CHECK-NEXT:   { 0x2, ValueList<4, 101> }, // 1
// CHECK-NEXT:  };
// CHECK: const Entry *lookupEntry(uint8_t Key) {
