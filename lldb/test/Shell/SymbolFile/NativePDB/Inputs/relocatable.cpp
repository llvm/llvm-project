// Test input for relocatable PE/COFF binary with PDB symbols.
// This simulates a DLL or other position-independent code.

extern "C" __declspec(dllexport) int relocatable_function(int x) {
    return x * 2;
}