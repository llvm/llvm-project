enum class SignedEnum : int { min = -2, max = 1 };
enum class UnsignedEnum : unsigned { min = 0, max = 3 };

struct BitfieldStruct {
  SignedEnum signed_min : 2;
  SignedEnum signed_other : 2;
  SignedEnum signed_max : 2;
  UnsignedEnum unsigned_min : 2;
  UnsignedEnum unsigned_other : 2;
  UnsignedEnum unsigned_max : 2;
};

int main() {
  BitfieldStruct bfs;
  bfs.signed_min = SignedEnum::min;
  bfs.signed_other = static_cast<SignedEnum>(-1);
  bfs.signed_max = SignedEnum::max;

  bfs.unsigned_min = UnsignedEnum::min;
  bfs.unsigned_other = static_cast<UnsignedEnum>(1);
  bfs.unsigned_max = UnsignedEnum::max;

  return 0; // break here
}
