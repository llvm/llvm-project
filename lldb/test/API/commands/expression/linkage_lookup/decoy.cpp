// The primary JIT relocation "linkage_target" must not bind to this C++
// function through the basename alias in the general symbol-name index.
int linkage_target() { return 11; }

int fcl_alias_target() { return 19; }
