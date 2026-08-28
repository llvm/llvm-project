// Second internal data symbol with the SAME name as the one in
// colliding_a.c. Two such symbols in the target's symtab is what trips
// SymbolContext::FindBestGlobalDataSymbol -> "Multiple internal symbols
// found".
static const int colliding_ns __attribute__((used)) = 2;

const int *colliding_b_anchor(void) { return &colliding_ns; }
