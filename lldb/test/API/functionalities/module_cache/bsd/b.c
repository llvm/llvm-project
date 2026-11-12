static int __b_global = 2;

int b(int arg) {
    int result = arg + __b_global;
    return result; // Set file and line breakpoint inside b().
}
