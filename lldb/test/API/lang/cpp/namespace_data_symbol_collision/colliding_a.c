// Internal (file-static) data symbol named `colliding_ns`. Built without
// debug info, so lldb sees only the symtab entry, not a DWARF VarDecl.
static const int colliding_ns __attribute__((used)) = 1;

// Anchor referenced from main so the linker keeps the object alive.
const int *colliding_a_anchor(void) { return &colliding_ns; }
