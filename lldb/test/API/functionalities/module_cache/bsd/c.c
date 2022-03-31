static int __c_global = 3;

int c(int arg) {
    int result = arg + __c_global;
    return result; // Set file and line breakpoint inside c().
}
