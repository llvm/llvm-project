#include "onetwo.h"

One::~One() = default;
member::One::~One() = default;
array::One::~One() = default;

result::One::One(int member) : member(member) {}
result::One::~One() = default;

void func_shadow::One(int) {}
func_shadow::One::~One() = default;
void func_shadow::One(float) {}
