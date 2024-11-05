enum EnumUChar { UChar = 1 };
enum EnumUShort { UShort = 0x101 };
enum EnumUInt { UInt = 0x10001 };
enum EnumSLong { SLong = 0x100000001 };
enum EnumULong { ULong = 0xFFFFFFFFFFFFFFF0 };
enum EnumNChar { NChar = -1 };
enum EnumNShort { NShort= -0x101 };
enum EnumNInt { NInt = -0x10001 };
enum EnumNLong { NLong = -0x100000001 };

int main() {
  auto UChar_promoted = +EnumUChar::UChar;
  auto UShort_promoted = +EnumUShort::UShort;
  auto UInt_promoted = +EnumUInt::UInt;
  auto SLong_promoted = +EnumSLong::SLong;
  auto ULong_promoted = +EnumULong::ULong;
  auto NChar_promoted = +EnumNChar::NChar;
  auto NShort_promoted = +EnumNShort::NShort;
  auto NInt_promoted = +EnumNInt::NInt;
  auto NLong_promoted = +EnumNLong::NLong;
  return 0; // break here
}
