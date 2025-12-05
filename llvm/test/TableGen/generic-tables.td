// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -gen-searchable-tables -I %p/../../include -DERROR1 %s 2>&1 | FileCheck --check-prefix=ERROR1 %s
// XFAIL: vg_leak

include "llvm/TableGen/SearchableTable.td"

// CHECK-LABEL: GET_BValues_DECL
// CHECK: enum BValues {
// CHECK:   BAlice = 172,
// CHECK:   BBob = 20,
// CHECK:   BCharlie = 128,
// CHECK:   BEve = 76,
// CHECK: }

// CHECK-LABEL: GET_CEnum_DECL
// CHECK: enum CEnum {
// CHECK:   CBar
// CHECK:   CBaz
// CHECK:   CFoo
// CHECK: }

// CHECK-LABEL: GET_FEnum_DECL
// CHECK: enum class FEnum {
// CHECK:   FApple = 0,
// CHECK:   FBanana = 1,
// CHECK:   FCherry = 2,
// CHECK: }

// CHECK-LABEL: GET_ATable_DECL
// CHECK: const AEntry *lookupATableByValues(uint8_t Val1, uint16_t Val2);

// CHECK-LABEL: GET_ATable_IMPL
// CHECK: constexpr AEntry ATable[] = {
// CHECK-NOT:   { "aaa"
// CHECK:       { "baz", 0x2, 0x6, 0xFFFFFFFF00000000 },
// CHECK:       { "foo", 0x4, 0x4, 0x100000000 },
// CHECK:       { "foobar", 0x4, 0x5, 0x100000000 },
// CHECK:       { "bar", 0x5, 0x3, 0x100000000 },
// CHECK: };

// CHECK: const AEntry *lookupATableByValues(uint8_t Val1, uint16_t Val2) {
// CHECK:   return &*Idx;
// CHECK: }

class AEntry<string str, int val1, int val2, bits<64> val3> {
  string Str = str;
  bits<8> Val1 = val1;
  bits<10> Val2 = val2;
  bits<64> Val3 = val3;
  bit IsNeeded = 1;
}

def : AEntry<"aaa", 0, 0, 0> { let IsNeeded = 0; }
def : AEntry<"bar",    5, 3, 0x100000000>;
def : AEntry<"baz",    2, 6, 0xFFFFFFFF00000000>;
def : AEntry<"foo",    4, 4, 0b0000000000000000000000000000000100000000000000000000000000000000>;
def : AEntry<"foobar", 4, 5, 4294967296>;

def ATable : GenericTable {
  let FilterClass = "AEntry";
  let FilterClassField = "IsNeeded";
  let Fields = ["Str", "Val1", "Val2", "Val3"];

  let PrimaryKey = ["Val1", "Val2"];
  let PrimaryKeyName = "lookupATableByValues";
}


// CHECK-LABEL: GET_BTable_IMPL
// CHECK: constexpr BTypeName BTable[] = {
// CHECK:   { "BAlice", 0xAC, false,  },
// CHECK:   { "BBob", 0x14, false, Bob == 13 },
// CHECK:   { "BCharlie", 0x80, true, Charlie == 42 },
// CHECK:   { "BEve", 0x4C, true, Eve == 108 },
// CHECK:  };
// CHECK: const BTypeName *lookupBTableByName(StringRef Name) {
// CHECK:   return &BTable[Idx->_index];
// CHECK: }
// CHECK: const BTypeName *lookupBTableByNameAndFlag(StringRef Name, bool Flag) {
// CHECK:   return &BTable[Idx->_index];
// CHECK: }

class BEntry<bits<16> enc, bit flag = 0, code test = [{}]> {
  string Name = NAME;
  bits<16> Encoding = enc;
  bit Flag = flag;
  code Test = test;
}

def BAlice   : BEntry<0xac>;
def BBob     : BEntry<0x14, 0, [{Bob == 13}]>;
def BCharlie : BEntry<0x80, 1, "Charlie == 42">;
def BEve     : BEntry<0x4c, 1, [{Eve == }] # 108>;

def BValues : GenericEnum {
  let FilterClass = "BEntry";
  let NameField = "Name";
  let ValueField = "Encoding";
}

def BTable : GenericTable {
  let FilterClass = "BEntry";
  string CppTypeName = "BTypeName";
  let Fields = ["Name", "Encoding", "Flag", "Test"];
  string TypeOf_Test = "code";
}

def lookupBTableByName : SearchIndex {
  let Table = BTable;
  let Key = ["Name"];
}

def lookupBTableByNameAndFlag : SearchIndex {
  let Table = BTable;
  let Key = ["Name", "Flag"];
}

// CHECK-LABEL: GET_CTable_DECL
// CHECK: const CEntry *lookupCEntryByEncoding(uint16_t Encoding);
// CHECK: const CEntry *lookupCEntry(StringRef Name, unsigned Kind);
// CHECK-LABEL: GET_CTable_IMPL
// CHECK: const CEntry *lookupCEntryByEncoding(uint16_t Encoding) {
// CHECK:   if ((uint16_t)Encoding != std::clamp((uint16_t)Encoding, (uint16_t)0xA, (uint16_t)0xF))
// CHECK:     return nullptr;

// CHECK: const CEntry *lookupCEntry(StringRef Name, unsigned Kind) {
// CHECK: Index[] = {
// CHECK:   { "ALICE", CBar, 1 },
// CHECK:   { "ALICE", CFoo, 0 },
// CHECK:   { "BOB", CBaz, 2 },

class CEnum;

def CFoo : CEnum;
def CBar : CEnum;
def CBaz : CEnum;

def CEnum : GenericEnum {
  let FilterClass = "CEnum";
}

class CEntry<string name, CEnum kind, int enc> {
  string Name = name;
  CEnum Kind = kind;
  bits<16> Encoding = enc;
}

def : CEntry<"alice", CFoo, 10>;
def : CEntry<"alice", CBar, 13>;
def : CEntry<"bob",   CBaz, 15>;

def CTable : GenericTable {
  let FilterClass = "CEntry";
  let Fields = ["Name", "Kind", "Encoding"];

  string TypeOf_Kind = "CEnum";

  let PrimaryKey = ["Encoding"];
  let PrimaryKeyName = "lookupCEntryByEncoding";
  let PrimaryKeyEarlyOut = 1;
}

def lookupCEntry : SearchIndex {
  let Table = CTable;
  let Key = ["Name", "Kind"];
}

#ifdef ERROR1

class DEntry<string str, int val1> {
  string Str = str;
  bits<8> Val1 = val1;
}

def DFoo : DEntry<"foo", 1>;
// ERROR1: [[@LINE+1]]:5: error: Record 'DBar' for table 'DTable' is missing field 'Val1'
def DBar : DEntry<"bar", ?>;

def DTable : GenericTable {
  let FilterClass = "DEntry";
  let Fields = ["Str", "Val1"];
}

#endif // ERROR1

// CHECK-LABEL: GET_EEntryEvenTable_DECL
// CHECK: const EEntry *lookupEEntryEvenTableByValue(uint8_t Value);

// CHECK-LABEL: GET_EEntryEvenTable_IMPL
// CHECK: constexpr EEntry EEntryEvenTable[] = {
// CHECK:   { 0x2
// CHECK:   { 0x4
// CHECK:   { 0x6
// CHECK:   { 0x8
// CHECK:   { 0xA
// CHECK: };

// CHECK: const EEntry *lookupEEntryEvenTableByValue(uint8_t Value) {
// CHECK:   return &*Idx;
// CHECK: }

// CHECK-LABEL: GET_EEntryOddTable_DECL
// CHECK: const EEntry *lookupEEntryOddTableByValue(uint8_t Value);

// CHECK-LABEL: GET_EEntryOddTable_IMPL
// CHECK: constexpr EEntry EEntryOddTable[] = {
// CHECK:   { 0x1
// CHECK:   { 0x3
// CHECK:   { 0x5
// CHECK:   { 0x7
// CHECK:   { 0x9
// CHECK: };

// CHECK: const EEntry *lookupEEntryOddTableByValue(uint8_t Value) {
// CHECK:   return &*Idx;
// CHECK: }

// We can construct two GenericTables with the same FilterClass, so that they
// select from the same overall set of records, but assign them with different
// FilterClassField values so that they include different subsets of the records
// of that class.
class EEntry<bits<8> value> {
  bits<8> Value = value;
  bit IsEven = !eq(!and(value, 1), 0);
  bit IsOdd = !not(IsEven);
}

foreach i = {1-10} in {
  def : EEntry<i>;
}

def EEntryEvenTable : GenericTable {
  let FilterClass = "EEntry";
  let FilterClassField = "IsEven";
  let Fields = ["Value"];
  let PrimaryKey = ["Value"];
  let PrimaryKeyName = "lookupEEntryEvenTableByValue";
}

def EEntryOddTable : GenericTable {
  let FilterClass = "EEntry";
  let FilterClassField = "IsOdd";
  let Fields = ["Value"];
  let PrimaryKey = ["Value"];
  let PrimaryKeyName = "lookupEEntryOddTableByValue";
}

// CHECK-LABEL: GET_FTable_DECL
// CHECK: const FTableEntry *lookupFTableByEnum(FEnum Enum);
// CHECK: const FTableEntry *lookupFTableByEnumAndValue(FEnum Enum, uint32_t Value);

// CHECK-LABEL: GET_FTable_IMPL
// CHECK: constexpr FTableEntry FTable[] = {
// CHECK:   { FEnum::FApple, 0x1 },
// CHECK:   { FEnum::FBanana, 0x2 },
// CHECK:   { FEnum::FCherry, 0x3 },
// CHECK: };
// CHECK: const FTableEntry *lookupFTableByEnum(FEnum Enum) {
// CHECK:   if ((FEnum)Enum != std::clamp((FEnum)Enum, (FEnum)FEnum::FApple, (FEnum)FEnum::FCherry))
// CHECK:     return nullptr;
// CHECK: const FTableEntry *lookupFTableByEnumAndValue(FEnum Enum, uint32_t Value) {
// CHECK:   struct IndexType {
// CHECK:     FEnum Enum;
// CHECK:     uint32_t Value;
// CHECK:     unsigned _index;
// CHECK:   };
// CHECK:   static const struct IndexType Index[] = {
// CHECK:     { FEnum::FApple, 0x1, 0 },
// CHECK:     { FEnum::FBanana, 0x2, 1 },
// CHECK:     { FEnum::FCherry, 0x3, 2 },
// CHECK:   };
// CHECK:   struct KeyType {
// CHECK:     FEnum Enum;
// CHECK:     uint32_t Value;
// CHECK:   };
// CHECK:   return &FTable[Idx->_index];

// CHECK-LABEL: GET_FTableContiguous_DECL
// CHECK: const FTableEntry *lookupFTableContiguousByEnum(FEnum Enum);

// CHECK-LABEL: GET_FTableContiguous_IMPL
// CHECK: constexpr FTableEntry FTableContiguous[] = {
// CHECK:   { FEnum::FApple, 0x1 },
// CHECK:   { FEnum::FBanana, 0x2 },
// CHECK:   { FEnum::FCherry, 0x3 },
// CHECK: };
// CHECK: const FTableEntry *lookupFTableContiguousByEnum(FEnum Enum) {
// CHECK:   if ((FEnum)Enum != std::clamp((FEnum)Enum, (FEnum)FEnum::FApple, (FEnum)FEnum::FCherry))
// CHECK:     return nullptr;
// CHECK:   auto Table = ArrayRef(FTableContiguous);
// CHECK:   size_t Idx = (unsigned)Enum - (unsigned)FEnum::FApple;
// CHECK:   return &Table[Idx];

class FEntry;

def FApple : FEntry;
def FBanana : FEntry;
def FCherry : FEntry;

def FEnum : GenericEnumClass {
  let FilterClass = "FEntry";
}

class FTableEntry<FEntry EnumValue, int val> {
  FEntry Enum = EnumValue;
  bits<32> Value = val;
}

def FAppleEntry : FTableEntry<FApple, 1>;
def FBananaEntry : FTableEntry<FBanana, 2>;
def FCherryEntry : FTableEntry<FCherry, 3>;

def FTable : GenericTable {
  let FilterClass = "FTableEntry";
  let Fields = ["Enum", "Value"];

  string TypeOf_Enum = "FEnum";

  let PrimaryKey = ["Enum"];
  let PrimaryKeyName = "lookupFTableByEnum";
  let PrimaryKeyEarlyOut = 1;
}

def lookupFTableByEnumAndValue : SearchIndex {
  let Table = FTable;
  let Key = ["Enum", "Value"];
}

def FTableContiguous : GenericTable {
  let FilterClass = "FTableEntry";
  let Fields = ["Enum", "Value"];

  string TypeOf_Enum = "FEnum";

  let PrimaryKey = ["Enum"];
  let PrimaryKeyName = "lookupFTableContiguousByEnum";
}
