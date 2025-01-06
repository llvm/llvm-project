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
// CHECK-NEXT: struct KeyType {
// CHECK-NEXT:    uint16_t Encoding;
// CHECK-NEXT:  };
// CHECK-NEXT:  KeyType Key = {Encoding};
// CHECK-NEXT:  struct Comp {
// CHECK-NEXT:    bool operator()(const SysReg &LHS, const KeyType &RHS) const {
// CHECK-NEXT:      if (LHS.Encoding < RHS.Encoding)
// CHECK-NEXT:        return true;
// CHECK-NEXT:      if (LHS.Encoding > RHS.Encoding)
// CHECK-NEXT:        return false;
// CHECK-NEXT:      return false;
// CHECK-NEXT:    }
// CHECK-NEXT:    bool operator()(const KeyType &LHS, const SysReg &RHS) const {
// CHECK-NEXT:      if (LHS.Encoding < RHS.Encoding)
// CHECK-NEXT:        return true;
// CHECK-NEXT:      if (LHS.Encoding > RHS.Encoding)
// CHECK-NEXT:        return false;
// CHECK-NEXT:      return false;
// CHECK-NEXT:    }
// CHECK-NEXT:  };
// CHECK-NEXT:  auto Table = ArrayRef(List1);
// CHECK-NEXT:  auto It = std::equal_range(Table.begin(), Table.end(), Key, Comp());
// CHECK-NEXT:  return llvm::make_range(It.first, It.second);
// CHECK-NEXT: }
// CHECK-NEXT: #endif
