#include "onetwo.h"

Two::~Two() = default;
member::Two::~Two() = default;
array::Two::~Two() = default;

result::Two::Two(int member) : member(member) {}
result::Two::~Two() = default;
result::One result::Two::one() const { return One(member - 100); }
