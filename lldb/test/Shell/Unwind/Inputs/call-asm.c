// Explicit mangling is necessary as on Darwin an underscore is prepended to the symbol.
int asm_main() __asm("asm_main");
int main() { return asm_main(); }
