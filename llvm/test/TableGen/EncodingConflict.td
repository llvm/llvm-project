// RUN: llvm-tblgen -gen-searchable-tables -I %p/../../include %s | FileCheck %s

include "llvm/TableGen/SearchableTable.td"

class SysReg<string name, bits<12> op> {
  string Name = name;
  bits<12> Encoding = op;
  code FeaturesRequired = [{ {} }];
}

def List1 : GenericTable {
  let FilterClass = "SysReg";
  let Fields = [
     "Name", "Encoding", "FeaturesRequired",
  ];

  let PrimaryKey = [ "Encoding" ];
  let PrimaryKeyName = "lookupSysRegByEncoding";
  let PrimaryKeyReturnRange = true;
}

let FeaturesRequired = [{ {Feature1} }] in {
def : SysReg<"csr1", 0x7C0>;
}

let FeaturesRequired = [{ {Feature2} }] in {
def : SysReg<"csr2", 0x7C0>;
}

// CHECK: #ifdef GET_List1_DECL
// CHECK-NEXT: llvm::iterator_range<const SysReg *> lookupSysRegByEncoding(uint16_t Encoding);
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_List1_IMPL
// CHECK-NEXT: constexpr SysReg List1[] = {
// CHECK-NEXT:   { "csr1", 0x7C0,  {Feature1}  }, // 0
// CHECK-NEXT:   { "csr2", 0x7C0,  {Feature2}  }, // 1
// CHECK-NEXT:  };

// CHECK: llvm::iterator_range<const SysReg *> lookupSysRegByEncoding(uint16_t Encoding) {
// CHECK-NEXT:   SysReg Key;
// CHECK-NEXT:   Key.Encoding = Encoding;
// CHECK-NEXT:   auto Table = ArrayRef(List1);
// CHECK-NEXT:   auto It = std::equal_range(Table.begin(), Table.end(), Key,
// CHECK-NEXT:     [](const SysReg &LHS, const SysReg &RHS) {
// CHECK-NEXT:       if (LHS.Encoding < RHS.Encoding)
// CHECK-NEXT:         return true;
// CHECK-NEXT:       if (LHS.Encoding > RHS.Encoding)
// CHECK-NEXT:         return false;
// CHECK-NEXT:       return false;
// CHECK-NEXT:     });

// CHECK:   return llvm::make_range(It.first, It.second);
// CHECK-NEXT: }
// CHECK-NEXT: #endif

def List2 : GenericTable {
  let FilterClass = "SysReg";
  let Fields = [
     "Name", "Encoding", "FeaturesRequired",
  ];
}

def lookupSysRegByName : SearchIndex {
  let Table = List2;
  let Key = [ "Name" ];
  let ReturnRange = true;
}

// CHECK: #ifdef GET_List2_DECL
// CHECK-NEXT: llvm::iterator_range<const SysReg *> lookupSysRegByName(StringRef Name);
// CHECK-NEXT: #endif

// CHECK: #ifdef GET_List2_IMPL
// CHECK-NEXT: constexpr SysReg List2[] = {
// CHECK-NEXT:   { "csr1", 0x7C0,  {Feature1}  }, // 0
// CHECK-NEXT:   { "csr2", 0x7C0,  {Feature2}  }, // 1
// CHECK-NEXT:  };

// CHECK: llvm::iterator_range<const SysReg *> lookupSysRegByName(StringRef Name) {
// CHECK-NEXT:   struct IndexType {
// CHECK-NEXT:     const char * Name;
// CHECK-NEXT:     unsigned _index;
// CHECK-NEXT:   };
// CHECK-NEXT:   static const struct IndexType Index[] = {
// CHECK-NEXT:     { "CSR1", 0 },
// CHECK-NEXT:     { "CSR2", 1 },
// CHECK-NEXT:   };

// CHECK:   IndexType Key;
// CHECK-NEXT:   Key.Name = Name.upper();
// CHECK-NEXT:   auto Table = ArrayRef(Index);
// CHECK-NEXT:   auto It = std::equal_range(Table.begin(), Table.end(), Key,
// CHECK-NEXT:     [](const IndexType &LHS, const IndexType &RHS) {
// CHECK-NEXT:       int CmpName = StringRef(LHS.Name).compare(RHS.Name);
// CHECK-NEXT:       if (CmpName < 0) return true;
// CHECK-NEXT:       if (CmpName > 0) return false;
// CHECK-NEXT:       return false;
// CHECK-NEXT:     });

// CHECK:   return llvm::make_range(It.first, It.second);
// CHECK-NEXT: }
// CHECK-NEXT: #endif

// CHECK: #undef GET_List1_DECL
// CHECK-NEXT: #undef GET_List1_IMPL
// CHECK-NEXT: #undef GET_List2_DECL
// CHECK-NEXT: #undef GET_List2_IMPL
