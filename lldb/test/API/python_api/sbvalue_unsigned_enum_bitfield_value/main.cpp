#include <stdint.h>

enum class EnumVals : uint16_t { VAL0 = 0 };

struct Foo {
  EnumVals b : 4;
};

int main(int argc, char const *argv[], char const *envp[]) {
  Foo f{.b = static_cast<EnumVals>(8)};
  return 0; //% b = self.frame().FindVariable("f").GetChildMemberWithName("b")
            //% val = b.GetValueAsUnsigned()
            //% self.assertEqual(val, 8, "Bit-field not correctly extracted")
}
