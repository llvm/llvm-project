#include "Type.h"
#include <iostream>
#include <string>

using namespace ABI;

void ABIBuiltinType::dump() const {
    std::cout << "BuiltinType: ";
    // Assuming you're sticking with these only:
    switch (kind) {
        case Void: std::cout << "Void\n"; break;
        case Bool: std::cout << "Bool\n"; break;
        case Integer:
          std::cout << "Integer\n";
          break;
        case Int128: std::cout << "Int128\n"; break;
        case UInt128:
          std::cout << "UInt128\n";
          break;
        case LongLong:
          std::cout << "LongLong\n";
          break;
        case Float:
          std::cout << "Float\n";
          break;
        case Double:
          std::cout << "Double\n";
          break;
        case Float16:
          std::cout << "Float16\n";
          break;
        case BFloat16:
          std::cout << "BFloat16\n";
          break;
        case Float128:
          std::cout << "Float128\n";
          break;
        case LongDouble:
          std::cout << "LongDouble\n";
          break;
    }
}

void ABIRecordType::dump() const {
  std::cout << "Record: " << RecordName << "\n";
  std::cout << "Alignment: " << Alignment << " bits\n";
  for (const auto &F : Fields) {
    std::cout << "  Field: " << F.Name << ", Offset: " << F.OffsetInBits
              << " bits"
              << ", TypeClass: " << F.FieldType->getTypeClass() << "\n";
    }
}
